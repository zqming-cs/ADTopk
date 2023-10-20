import torch

from ADTopklib import Compressor
import random


def sparsify(tensor, compress_ratio):

    tensor = tensor.flatten().cuda()

    len = tensor.numel()
    compress_ratio=0.001
    compress_ratio=0.01
    compress_ratio=0.05
    
    k = max(1, int(len * compress_ratio))

    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    return values, indices


def desparsify(tensors, numel):

    values, indices = tensors
    
    # Awareness
    if values.numel()==numel:
        return values
    

    tensor_decompressed = torch.zeros(
        numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()

    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed


class TopKCompressor(Compressor):

    def __init__(self, compress_ratio, rank):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank

    def compress(self, tensor, name):

        tensors = sparsify(tensor, self.compress_ratio)


        ctx = tensor.numel(), tensor.size()
        return tensors, ctx

    def show_sparsify(self, tensor):

        print('----------'+str(self.rank)+'----------')
        print(tensor.shape)
        tensor = tensor.flatten()
        print(tensor)

    def decompress(self, tensors, ctx, name):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        if ctx==None:
            tensor, = tensors
            return tensor
        numel, shape = ctx
        
        tensor_decompressed = desparsify(tensors, numel)

            

        return tensor_decompressed.view(shape)

    def decompress_add(self, tensors, ctx, name):
        numel, shape = ctx
        values, indices = tensors
        if values.numel()==numel:
            return values

        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()

        # if hvd.rank() == 0:
        #     print('values: ', values, 'indices: ', indices)
        # [a,b,    c,d]  [0,1,    0,2]
        # [c, b ,d ][a+c, b,d ]
        tensor_decompressed = tensor_decompressed.scatter_add(0, indices, values)
        return tensor_decompressed.view(shape)

