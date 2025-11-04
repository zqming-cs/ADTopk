# baseline
comm_params = {
    'comm_mode':'allreduce',
    'compressor':'none',
    'memory':'none',
    'send_size_aresame':True
}




# Allreduce
# when allreduce are used, the value of send_size_aresame does not matters. 
# Sparsify compression methods do not support allreduce. 

comm_params = {
    'comm_mode':'allreduce',
    'compressor':'topk',
    'memory':'residual',
    'send_size_aresame':True
}

# Fast_Allgather: does not transfer tensor_size and not split aggregated tensors
# does not care about the value of send_size_aresame
# compressor must support decompress_add method
comm_params = {
    'comm_mode':'fast_allgather',
    'compressor':'topk',
    'memory':'residual',
    'send_size_aresame':True
}

# Allgather: original implementation
# when the size of compressed tensor in different rank are not the same, send_size_aresame := False
comm_params = {
    'comm_mode':'allgather',
    'compressor':'topk',
    'memory':'residual',
    'send_size_aresame':True
}










