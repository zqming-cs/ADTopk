import torch

from grace_dll.torch import Compressor
import random
import horovod.torch as hvd
import numpy as np


class TopkThroughputCompressor(Compressor):
    def __init__(self, compress_ratio, rank, epoch):
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
        # str_1='residual_function.0.weight'
        # str_2='residual_function.3.weight'  
        # str_3='residual_function.6.weight'

        compress_ratio_global=1.0
        
        numel = tensor.numel()
        shape =tensor.shape

        compress_ratio=0.01


        # if True:
        if tensor.dim() >1:

            # Global Top-k
            # tensor_flatten=tensor.flatten().cuda()
            # k= max(1, int(numel * compress_ratio*compress_ratio_global))
            # _, indices_flatten_global = torch.topk(tensor_flatten.abs(), k, sorted=False,)
            # values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
            # return values_flatten_global, indices_flatten_global

            # All-Channel Top-k
            if shape[0]>shape[1]:
                k= max(1, int(shape[0] * compress_ratio))
                _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=0,sorted=False,)
                values_flatten_1 = torch.gather(tensor, 0, indices_flatten_1)
            else:
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
    def desparsify(self, tensors, numel, shape):
        values, indices = tensors
        # if True:
        if values.numel()==numel:
            return values
        else:

            # All-Channel Top-k
            tensor_decompressed = torch.zeros(
                shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            if shape[0] > shape[1]:
                tensor_decompressed.scatter_(0, indices, values)
            else:
                tensor_decompressed.scatter_(1, indices, values)
            
            # Global Top-k
            # tensor_decompressed = torch.zeros(
            #     numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
            # tensor_decompressed.scatter_(0, indices, values)

            return tensor_decompressed
    

    # 抽象方法重载compress
    def compress(self, tensor, name):

        tensors = self.sparsify(tensor, self.compress_ratio,self.epoch, name)
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
        numel, shape = ctx
        
        tensor_decompressed =self.desparsify(tensors, numel,shape)
        # if self.rank==0:
            

        return tensor_decompressed.view(shape)


