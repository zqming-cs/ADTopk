import torch

from ADTopklib import Compressor
import random
import horovod.torch as hvd
import numpy as np


class AllChannelTopkCompressor(Compressor):

    def __init__(self, compress_ratio, rank):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank
        self.epoch=0
        self.iteration=0
        self.index=0
        self.layernumels={}
        self.thres_mean_arr=[]

        self.tensor_flatten_np_arr=[]
        self.values_flatten_global_np_arr=[]
        self.values_flatten_channel_np_arr=[]


    # tensor稀疏化得到top-k的稀疏值
    def sparsify(self, tensor, compress_ratio,epoch, name):
        # 将tensor展平成一维向量
        compress_ratio_global=1.0
        
        numel = tensor.numel()
        shape =tensor.shape

        # compress_ratio=0.001
        compress_ratio=0.01
        # compress_ratio=0.05

        # if True:
        if tensor.dim() >1:

            k= max(1, int(shape[1] * compress_ratio))
            _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=1,sorted=False,)
            values_flatten_1 = torch.gather(tensor, 1, indices_flatten_1)
            
            return values_flatten_1, indices_flatten_1


        tensor = tensor.flatten().cuda()
        numel = tensor.numel()
        values=tensor
        indices=torch.arange(0,numel).cuda(tensor.device)
        return values, indices


    # tensor反稀疏化
    def desparsify(self, tensors, numel, shape,name):
        values, indices = tensors
        # if True:
        if values.numel()==numel:
            return values
        else:
            
            tensor_decompressed = torch.zeros(
                shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            tensor_decompressed.scatter_(1, indices, values)

            return tensor_decompressed
  
        return tensor_decompressed

    # 抽象方法重载compress
    def compress(self, tensor, name):

        tensors = self.sparsify(tensor, self.compress_ratio, self.epoch, name)
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
        
        tensor_decompressed =self.desparsify(tensors, numel,shape,name)

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

            tensor_decompressed = torch.zeros(
                shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            size = hvd.size()
            sizes = [tensor_decompressed.shape[0]] * size
            indices_list = indices.split(sizes)
            indices = torch.concatenate(indices_list,axis = 1)
            values_list = values.split(sizes)
            values = torch.concatenate(values_list, axis = 1)
            tensor_decompressed = tensor_decompressed.scatter_add(1, indices, values)

            tensor_decompressed.scatter_add(1, indices, values)

        return tensor_decompressed.view(shape)


