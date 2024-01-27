import torch

from ADTopklib import Compressor
import random
import numpy as np
import horovod.torch as hvd
import math
import scipy.stats as stats
import time

class ADTopkCompressor(Compressor):

    def __init__(self, compress_ratio, rank):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank
        self.training_epochs=0
        self.training_iterations=0
        
        self.epoch=0
        self.iteration=0
        self.index=0
        self.layernumels={}
        self.thres_mean_arr=[]
        
        self.train_type=None
        

        self.tau=5
        self.warm_train_iterations=5
        
        self.threshold = {}
        self.epsilon = {}
        
        
        self.alldimension= {}
        self.tensor={}
   
    def initialize(self, named_parameters):
        if hvd.rank() == 0:
            print("=> initializing adtopk compressor")

        if self.train_type=='resnet':
            self.tau=5
            self.warm_train_iterations=15
        elif self.train_type=='vgg':
            self.tau=10
            self.warm_train_iterations=15
        elif self.train_type=='lstm':
            self.tau=20
            self.warm_train_iterations=15
        elif self.train_type=='bert':
            self.tau=20
            self.warm_train_iterations=15
        else:
            self.tau=5
            self.warm_train_iterations=15
        
        for name, param in named_parameters:
            
            if param.dim() >1:
                self.threshold[name]=None
                self.epsilon = None
                
                self.alldimension[name]=False

            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]


    def ThresholdTopk(self, name, tensor_flatten, tensor, shape, numel, compress_ratio, epoch, iteration):
        
        thres_global = self.threshold[name]
        epsilon = self.threshold[name]
        
        if iteration < self.warm_train_iterations:
            k= max(1, int(numel * compress_ratio))
            values_flatten, indices_flatten_global = torch.topk(tensor_flatten.abs(), k, sorted=False,)
            values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
            
            thres = values_flatten.min()
            self.threshold[name]=thres*1.0
            self.epsilon[name]=thres/10.0
            return values_flatten_global, indices_flatten_global

        elif iteration > 3/4*self.training_iterations and self.alldimension[name]:
            k= max(1, int(shape[1] * compress_ratio))
            _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=1,sorted=False,)
            values_flatten_1 = torch.gather(tensor, 1, indices_flatten_1)
            return values_flatten_1, indices_flatten_1 

        elif iteration % self.tau==0:
            k= max(1, int(numel * compress_ratio))
            values_flatten, indices_flatten_global = torch.topk(tensor_flatten.abs(), k, sorted=False,)
            values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
                
            thres = values_flatten.min()
            self.threshold[name] = thres*1.0
            self.epsilon[name]=thres/10.0

            # update step tau
            if torch.abs(thres_global-thres) < epsilon:
                self.tau=self.tau + 1
                self.alldimension[name]=True
            elif torch.abs(thres_global-thres) < epsilon/2:
                self.tau=self.tau + 2
                self.alldimension[name]=True
            elif torch.abs(thres_global-thres) > epsilon:
                self.tau=max(self.tau - 1, 5)
                self.alldimension[name]=False
            elif torch.abs(thres_global-thres) > 2*epsilon:
                self.tau=max(self.tau - 2, 5)
                self.alldimension[name]=False

            return values_flatten_global, indices_flatten_global
        
        else:
            mask = tensor_flatten > thres_global
            indices_flatten_global, = torch.where(mask)   
            # values_flatten_global=tensor_flatten[indices_flatten_global]            
            values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
            return values_flatten_global, indices_flatten_global
        
    
    def sparsify(self,tensor, compress_ratio,epoch, iteration, name):
        tensor_flatten = tensor.flatten()
        numel = tensor.numel()
        shape =tensor.shape
        compress_ratio=0.01


        if self.attributes[name]['sign']==-1:
            k= max(1, int(shape[1] * compress_ratio))
            _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=1,sorted=False,)
            values_flatten_1 = torch.gather(tensor, 1, indices_flatten_1)

            return values_flatten_1, indices_flatten_1  
                   
        else:
            # Traditional Top-k                
            # k= max(1, int(numel * compress_ratio*compress_ratio_global))
            # _, indices_flatten_global = torch.topk(tensor_flatten.abs(), k, sorted=False,)
            # values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
            
            # Threshold Top-k  
            values_flatten_global, indices_flatten_global= self.ThresholdTopk(name,tensor_flatten,tensor, shape,numel,compress_ratio,epoch,iteration)
                   
            return values_flatten_global, indices_flatten_global


    def desparsify(self,tensors, numel,shape,name):
        values, indices = tensors
        if values.numel()==numel:
            return values

        else:
            if values.dim()==1 :
                tensor_decompressed = torch.zeros(
                    numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
                tensor_decompressed.scatter_(0, indices, values)
            else:
                tensor_decompressed = torch.zeros(
                    shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
                tensor_decompressed.scatter_(1, indices, values)

        return tensor_decompressed


    def compress(self, tensor, name):
        tensors = self.sparsify(tensor, self.compress_ratio,self.epoch,self.iteration, name)
        ctx = tensor.numel(), tensor.size()        
        self.attributes[name]['sign']=(-1)*self.attributes[name]['sign']     
        return tensors, ctx


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
                tensor_decompressed = torch.zeros(
                    shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()

                size = hvd.size()
                sizes = [tensor_decompressed.shape[0]] * size
                indices_list = indices.split(sizes)
                indices = torch.concatenate(indices_list,axis = 1)
                values_list = values.split(sizes)
                values = torch.concatenate(values_list, axis = 1)
                tensor_decompressed = tensor_decompressed.scatter_add(1, indices, values)

        return tensor_decompressed.view(shape)
