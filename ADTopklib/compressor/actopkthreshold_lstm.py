import torch

from grace_dll.torch import Compressor
import random
import numpy as np
import horovod.torch as hvd
import math
import scipy.stats as stats
import time


class ACTopkThresholdLSTMCompressor(Compressor):

    def __init__(self, compress_ratio, rank):
        super().__init__()

        self.compress_ratio = compress_ratio
        self.rank = rank
        self.training_epochs=0
        self.epoch=0
        self.iteration=0
        self.index=0
        self.layernumels={}
        self.thres_mean_arr=[]
        
        self.tau_=150

        self.attributes = {}
        
        self.threshold = {}
        self.tensor={}
   
    def initialize(self, named_parameters):
        if hvd.rank() == 0:
            print("=> initializing actopk compressor")
        for name, param in named_parameters:
            
            if param.dim() >1 and 'fc' not in name:
                self.threshold[name]=None
            
            if torch.is_tensor(param):
                numel = param.numel()
                shape = list(param.size())
            else:
                assert isinstance(param, (list, tuple))
                numel, shape = param[0], param[1]
            
            thres_global=None
            # 初始化阈值更新步长
            tau =150
            epsilon =0
            
            # tensors_aggregated=None
            # scale=None
            # tensors_aggregated_mean=None
            # tensors_residuals=None
            self.compress_ratio=0.01
            sign=-1
            # self.attributes[name] ={'numel':numel,'shape': shape, 'compress_ratio':self.compress_ratio,'rank':self.rank,'thres_global':thres_global,'afa':afa,\
            #     'tensors_aggregated':tensors_aggregated,'scale':scale,'tensors_aggregated_mean':tensors_aggregated_mean,\
            #         'tensors_residuals':tensors_residuals,'sign':sign,'thres_global_accuracy_arr':thres_global_accuracy_arr,'thres_global_actopk_arr':thres_global_actopk_arr,\
            #             'thres_global_gaussiank_arr':thres_global_gaussiank_arr,'thres_global_dgc_arr':thres_global_dgc_arr,'thres_global_redsync_arr':thres_global_redsync_arr,\
            #             'tau':tau, 'epsilon':epsilon}
            
            self.attributes[name] ={'thres_global':thres_global,'sign':sign, 'tau':tau, 'epsilon':epsilon} 

    # 自适应更新阈值ResNet-50
    def Adaptive_Threshold_Estimation(self, name, tensor_flatten, numel, compress_ratio, epoch, iteration):
        # Iteration的前期不进行阈值更新,直接用Topk计算Value和Indicate
        # if epoch <self.training_epochs/4:
        
    
        
        # tau = self.attributes[name]['tau']
        if epoch < 2 :
        # if epoch < 32 or numel < 300000 :
        # if epoch < 2 or numel<150000 :
            k= max(1, int(numel * compress_ratio))
            values_flatten, indices_flatten_global = torch.topk(tensor_flatten.abs(), k, sorted=False,)
            values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
            
            
            thres = values_flatten.min()
            self.threshold[name]=thres*1.0
            return values_flatten_global, indices_flatten_global
        
        else:

            thres_global = self.threshold[name]
            mask = tensor_flatten > thres_global
            indices_flatten_global, = torch.where(mask) 
            
            k= max(1, int(numel * compress_ratio))
            # if indices_flatten_global.numel()>k:
            #     indices_flatten_global=indices_flatten_global[:k]
                
            values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
            
            return values_flatten_global, indices_flatten_global
        
       


    # 自适应更新幅度
    def AdAdjust(self,name,epoch,afa_global,compress_ratio,iteration):
        afa=afa_global
        ratio=compress_ratio
        if epoch % 15==0 and iteration==5 :
            afa=min(0.8,afa_global*1.2)
            ratio=max(0.01,ratio*0.8)
            
            self.attributes[name]['afa']=afa
            self.attributes[name]['compress_ratio']=ratio
        return afa,ratio
    
    # tensor稀疏化得到top-k的稀疏值
    def sparsify(self,tensor, compress_ratio,epoch, iteration, name):
        # 将tensor展平成一维向量
        compress_ratio_global=1.0
        tensor_flatten = tensor.flatten()
        numel = tensor.numel()
        shape =tensor.shape

        compress_ratio=0.01
        # if True:
        if tensor.dim() >1:
            
            # Traditional Global Top-k
            # k= max(1, int(numel * compress_ratio*compress_ratio_global))
            # values_flatten_global_abs, indices_flatten_global = torch.topk(tensor_flatten.abs(), k, sorted=False,)
            # values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
            # return values_flatten_global, indices_flatten_global


            if self.attributes[name]['sign']==-1 or 'fc' in name:
            # if self.attributes[name]['sign']==-1 or 'classifier.6' in name:
                # case-1
                k= max(1, int(shape[1] * compress_ratio))
                _, indices_flatten_1 = torch.topk(tensor.abs(), k, dim=1,sorted=False,)
                values_flatten_1 = torch.gather(tensor, 1, indices_flatten_1)
                
 
                return values_flatten_1, indices_flatten_1  
                   
            else:
                # Global
                # k= max(1, int(numel * compress_ratio*compress_ratio_global))
                # _, indices_flatten_global = torch.topk(tensor_flatten.abs(), k, sorted=False,)
                # values_flatten_global = torch.gather(tensor_flatten, 0, indices_flatten_global)
                
                values_flatten_global, indices_flatten_global= self.Adaptive_Threshold_Estimation(name,tensor_flatten,numel,compress_ratio,epoch,iteration)

                return values_flatten_global, indices_flatten_global


        tensor = tensor.flatten().cuda()
        numel = tensor.numel()
        values=tensor
        indices=torch.arange(0,numel).cuda(tensor.device)
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
            if values.dim()==1 :
                tensor_decompressed = torch.zeros(
                    numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
                tensor_decompressed.scatter_(0, indices, values)
            else:
                # case-1
                tensor_decompressed = torch.zeros(
                    shape, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
                tensor_decompressed.scatter_(1, indices, values)


            # self.attributes[name]['sign']=(-1)*self.attributes[name]['sign']

        return tensor_decompressed

    # 抽象方法重载compress
    def compress(self, tensor, name):


        tensors = self.sparsify(tensor, self.compress_ratio,self.epoch,self.iteration, name)
        ctx = tensor.numel(), tensor.size()
        
        self.attributes[name]['sign']=(-1)*self.attributes[name]['sign']
        
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
                sizes = [tensor_decompressed.shape[0]] * size
                indices_list = indices.split(sizes)
                indices = torch.concatenate(indices_list,axis = 1)
                values_list = values.split(sizes)
                values = torch.concatenate(values_list, axis = 1)
                tensor_decompressed = tensor_decompressed.scatter_add(1, indices, values)

            # self.attributes[name]['sign']=(-1)*self.attributes[name]['sign']
            
        return tensor_decompressed.view(shape)
    
    # 高斯阈值估计
    def gaussiank_threshold_estimation(self, tensor):
        numel = tensor.numel()
        shape = tensor.size()
        k = max(int(numel * self.compress_ratio), 1)

        std = torch.std(tensor)
        mean = torch.mean(tensor)
        left_thres, right_thres = self.gen_threshold_from_normal_distribution(1-self.compress_ratio, float(mean), float(std))

        tensor_flatten = tensor.flatten().cuda()
        abs_tensor_tensor_flatten = torch.abs(tensor_flatten)
        one_indexes = abs_tensor_tensor_flatten > right_thres
        loops = 0
        while loops < 5:
            one_indexes = abs_tensor_tensor_flatten > right_thres
            selected = one_indexes.sum()
    
            if selected < 2*k/3:
                right_thres *= 0.5
            elif selected > 4*k/3:
                right_thres *= 1.5
            else:
                break
            loops += 1

    
        return right_thres
    
    def gen_threshold_from_normal_distribution(self, p_value, mu, sigma):
        zvalue = stats.norm.ppf((1-p_value)/2)
        return mu+zvalue*sigma, mu-zvalue*sigma
    
    
    # DGC阈值估计
    def dgc_threshold_estimation(self, tensor):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        sample_shape = [max(1, int(numel * 0.01))]
        sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long)
        sample_tensor = tensor[sample_index]

        k = max(1, int(numel * self.compress_ratio * 0.01))
        vals, indices = torch.topk(sample_tensor.abs(), k)

        thr = vals.min()
        # thr = vals.max()
        mask = tensor.abs() >= thr
        selected = mask.sum()

        for _ in range(2):
            if selected > 1.3 * numel * self.compress_ratio:
                thr = 1.3 * thr
            elif selected < 0.7 * numel * self.compress_ratio:
                thr = 0.7 * thr
            else:
                break
            mask = tensor.abs() >= thr
            selected = mask.sum()

        return thr
    
    # Redsync阈值估计
    def redsync_threshold_estimation(self, tensor):
        numel = tensor.numel()
        shape = tensor.size()
        k = max(int(numel * self.compress_ratio), 1)

        tensor_flatten = tensor.flatten().cuda()

        l = 0.0
        r = 1.0
        thres = 0.0
        eps = 0.2
        abs_tensor = torch.abs(tensor_flatten)
        mean_val = torch.mean(abs_tensor)
        max_val = torch.max(abs_tensor)

        one_indexes = abs_tensor > thres
        while r - l > eps:
            tmp_ratio = l + (r-l)/2
            thres = mean_val + tmp_ratio * (max_val - mean_val)
            one_indexes = abs_tensor > thres
            # indexes = one_indexes.nonzero().data.squeeze().view(-1)
            # nnz = indexes.numel()
            nnz = one_indexes.sum()

            if nnz > k and 2*k > nnz:
                break
            elif nnz < k/2:
                r = tmp_ratio
            else:
                l = tmp_ratio
        
        return thres
