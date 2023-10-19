import torch

from ADTopklib import Compressor
import random
import numpy as np
import horovod.torch as hvd
import math

class ACTopkInterleavingResnetCompressor(Compressor):

    def __init__(self, compress_ratio, rank):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank
        self.epoch=0
        self.iteration=0
        self.index=0
        self.layernumels={}
        self.thres_mean_arr=[]


        self.attributes = {}
        self.tensor={}

        # self.residuals={{}}
        # for i in range(hvd.size()):
        #     self.residuals[str(i)]={}
   
    def initialize(self, named_parameters):
        # if hvd.rank() == 0:
        #     print("=> initializing dgc compressor")
        for name, param in named_parameters:
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]
            
            afa=0.2
            thres_global=None
            compression_global=None
            indices_global= None
            values_global=None
            indices_channel_1=None
            values_channel_1=None
            tensor_original=None
            tensor_mean_global=None
            tensor_mean_channel=None
            tensors_aggregated=None
            scale=None
            tensors_aggregated_mean=None
            tensors_residuals=None
            self.compress_ratio=0.01
            sign=-1
            self.attributes[name] ={'numel':numel,'shape': shape, 'compress_ratio':self.compress_ratio,'rank':self.rank,'thres_global':thres_global,'afa':afa,\
                'compression_global':compression_global,'indices_global':indices_global,'values_global':values_global,\
                    'indices_channel_1':indices_channel_1,'values_channel_1':values_channel_1,\
                        'tensor_original':tensor_original,'tensor_mean_global':tensor_mean_global,'tensor_mean_channel':tensor_mean_channel,\
                            'tensors_aggregated':tensors_aggregated,'scale':scale,'tensors_aggregated_mean':tensors_aggregated_mean,\
                                'tensors_residuals':tensors_residuals,'sign':sign} 
            
            
            # self.residuals[str(hvd.rank())][name]=None
            # for i in range(hvd.size()): 
            #     self.residuals[str(i)]={}

    # tensor稀疏化得到top-k的稀疏值
    def sparsify(self,tensor, compress_ratio,epoch, name):
        # 将tensor展平成一维向量
        compress_ratio_global=1.0
        tensor_flatten = tensor.flatten()
        numel = tensor.numel()
        shape =tensor.shape

        compress_ratio=0.01
        # if True:
        if tensor.dim() >1:
            if self.attributes[name]['sign']==-1 or 'fc' in name:
                # case-1
                k= max(1, int(shape[1] * compress_ratio))
                _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=1,sorted=False,)
                values_flatten_1 = torch.gather(tensor, 1, indices_flatten_1)
                
                # case-2
                # if shape[0]>shape[1]:
                #     k= max(1, int(shape[0] * compress_ratio))
                #     _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=0,sorted=False,)
                #     values_flatten_1 = torch.gather(tensor, 0, indices_flatten_1)
                # else:
                #     k= max(1, int(shape[1] * compress_ratio))
                #     _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=1,sorted=False,)
                #     values_flatten_1 = torch.gather(tensor, 1, indices_flatten_1) 
                # self.attributes[name]['sign']=(-1)*self.attributes[name]['sign']

                return values_flatten_1, indices_flatten_1  
                   
            else:

                k= max(1, int(numel * compress_ratio*compress_ratio_global))
                _, indices_flatten_global = torch.topk(tensor_flatten.abs(), k, sorted=False,)
                values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
                self.attributes[name]['sign']=(-1)*self.attributes[name]['sign']

                return values_flatten_global, indices_flatten_global


        tensor = tensor.flatten().cuda()
        numel = tensor.numel()
        values=tensor
        indices=torch.arange(0,numel).cuda(tensor.device)
        # self.attributes[name]['sign']=(-1)*self.attributes[name]['sign']

        return values, indices


    # tensor反稀疏化
    def desparsify(self,tensors, numel,shape,name):
        values, indices = tensors
        if values.numel()==numel:
        # if tensors.dim()==1:
        # if len(tensors)==2:
            # values, indices = tensors
            return values

        else:
            if values.dim()==1:
                tensor_decompressed = torch.zeros(
                    numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
                tensor_decompressed.scatter_(0, indices, values)
            else:
                # case-1
                tensor_decompressed = torch.zeros(
                    shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
                tensor_decompressed.scatter_(1, indices, values)

                # case-2
                # tensor_decompressed = torch.zeros(
                #     shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
                # if shape[0] > shape[1]:
                #     tensor_decompressed.scatter_(0, indices, values)
                # else:
                #     tensor_decompressed.scatter_(1, indices, values)

            
        return tensor_decompressed

    # 抽象方法重载compress
    def compress(self, tensor, name):

        tensors = self.sparsify(tensor, self.compress_ratio,self.epoch, name)
        self.attributes[name]['sign']=(-1)*self.attributes[name]['sign']
        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def show_sparsify(self, tensor):
        # if self.rank==0:
        print('----------'+str(self.rank)+'----------')
        print(tensor.shape)
        tensor = tensor.flatten()
        print(tensor)


    def decompress(self, tensors, ctx,name):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        
        if ctx==None:
            tensor, = tensors
            return tensor
        numel, shape = ctx
        
        tensor_decompressed = self.desparsify(tensors, numel,shape,name)
        return tensor_decompressed.view(shape)
    
    def decompress_add(self, tensors, ctx, name):
        if ctx==None:
            tensor, = tensors
            return tensor
        
        numel, shape = ctx
        values, indices = tensors
        if values.numel()==numel:
            return values

        else:

            if values.dim()==1:
                tensor_decompressed = torch.zeros(
                    numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()

                tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
            else:

                # case-1
                tensor_decompressed = torch.zeros(
                    shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()

                size = hvd.size()
                # indices_1 = indices.reshape(size, -1, indices.shape[1])
                # values_1 = values.reshape(size, -1, values.shape[1])

                # if hvd.rank() == 0 and tensor_decompressed.shape[0]<5:
                sizes = [tensor_decompressed.shape[0]] * size
                indices_list = indices.split(sizes)
                # indices = torch.concatenate(indices_list,axis = 1)
                indices = torch.cat(indices_list,axis = 1)
                values_list = values.split(sizes)
                # values = torch.concatenate(values_list, axis = 1)
                values = torch.cat(values_list, axis = 1)
                tensor_decompressed = tensor_decompressed.scatter_add(1, indices, values)
                
        return tensor_decompressed.view(shape)
