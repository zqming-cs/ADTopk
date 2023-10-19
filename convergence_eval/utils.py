import torch

def topk(tensor, compress_ratio=0.01):
    tensor = tensor.flatten().cuda()
    numel = tensor.numel()
    k = max(1, int(numel * compress_ratio))
    # 获取top-k值的索引
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    tensors = values, indices
    tensor_decompressed = torch.zeros(
        numel, dtype=values.dtype, layout=values.layout, device=values.device).cuda()
    # 填充稀疏值
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed