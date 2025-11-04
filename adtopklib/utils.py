import horovod.torch as hvd

def find_duplicates(lst):
    seen = set()
    dups = set()
    for el in lst:
        if el in seen:
            dups.add(el)
        seen.add(el)
    return dups

# return communication mode: allreduce, allgather, allgather_fast
def get_comm(params):
    comm_name = params.get('comm_mode', 'allreduce')
    return comm_name

def get_compressor(params):
    compress_name = params.get('compressor', 'none')
    
    if compress_name == 'none':
        from ADTopklib.compressor.none import NoneCompressor
        compressor = NoneCompressor()

    elif compress_name == 'dgc':
        from ADTopklib.compressor.dgc import DgcCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        compressor = DgcCompressor(compress_ratio)
    
    elif compress_name == 'topk':
        from ADTopklib.compressor.topk import TopKCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        compressor = TopKCompressor(compress_ratio,rank=hvd.rank())
    
    elif compress_name == 'gaussiank':
        from ADTopklib.compressor.gaussiank import GaussiankCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        compressor = GaussiankCompressor(compress_ratio,rank=hvd.rank())
    
    elif compress_name == 'redsync':
        from ADTopklib.compressor.redsync import RedSyncCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        compressor = RedSyncCompressor(compress_ratio,rank=hvd.rank())
    
    elif compress_name == 'redsynctrim':
        from ADTopklib.compressor.redsync import RedSyncTrimCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        compressor = RedSyncTrimCompressor(compress_ratio,rank=hvd.rank())
   
    elif compress_name == 'sidcoexp':
        from ADTopklib.compressor.sidco import ExpCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        compressor = ExpCompressor(compress_ratio)
  
    elif compress_name == 'sidcogp':
        from ADTopklib.compressor.sidco import GParetoCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        compressor = GParetoCompressor(compress_ratio)
  
    elif compress_name == 'sidcogam':
        from ADTopklib.compressor.sidco import GammaGParetoCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        compressor = GammaGParetoCompressor(compress_ratio)
    
    elif compress_name == 'adtopk':
        from ADTopklib.compressor.adtopk import ADTopkCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        model_named_parameters = params.get('model_named_parameters')
        compressor = ADTopkCompressor(compress_ratio,rank=hvd.rank())
        compressor.initialize(model_named_parameters)
        
    elif compress_name == 'adtopk-i':
        from ADTopklib.compressor.adtopk_i import ADTopkInterleavingCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        model_named_parameters = params.get('model_named_parameters')
        compressor = ADTopkInterleavingCompressor(compress_ratio,rank=hvd.rank())
        compressor.initialize(model_named_parameters)
  
    elif compress_name == 'tradtopk':
        from ADTopklib.compressor.tradtopk import TraditionalTopkCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        model_named_parameters = params.get('model_named_parameters')
        compressor = TraditionalTopkCompressor(compress_ratio,rank=hvd.rank())
    
    elif compress_name == 'powersgd':
        from ADTopklib.compressor.powersgd import PowerSGDCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        model_named_parameters = params.get('model_named_parameters')
        compressor = PowerSGDCompressor(compress_ratio,rank=hvd.rank())

    elif compress_name == 'threshold':
        from ADTopklib.compressor.threshold import ThresholdCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        model_named_parameters = params.get('model_named_parameters')
        compressor = ThresholdCompressor(compress_ratio,rank=hvd.rank())
    
    elif compress_name == 'randomk':
        from ADTopklib.compressor.randomk import RandomKCompressor
        compress_ratio = params.get('compress_ratio', 0.01)
        model_named_parameters = params.get('model_named_parameters')
        compressor = RandomKCompressor(compress_ratio,rank=hvd.rank())

    else:
        raise NotImplementedError(compressor)

    return compressor


def get_memory(params):
    memory_name = params.get('memory', 'none')

    if memory_name == 'dgc':
        from ADTopklib.memory.dgc import DgcMemory
        momentum = params.get('momentum', 0.9)
        gradient_clipping = params.get('gradient_clipping', False)
        memory = DgcMemory(momentum, gradient_clipping)

    elif memory_name == 'none':
        from ADTopklib.memory.none import NoneMemory
        memory = NoneMemory()

    elif memory_name == 'residual':
        from ADTopklib.memory.residual import ResidualMemory
        memory = ResidualMemory()

    elif memory_name == 'residualgtopk':
        from ADTopklib.memory.residualgtopk import ResidualGlobalTopkMemory
        memory = ResidualGlobalTopkMemory()
    else:
        raise NotImplementedError(memory)

    return memory


def get_config(params):
    send_size_aresame = params.get('send_size_aresame', True)
    return send_size_aresame


# Special case:
# All dim==1 tensor should not be compressed
# ResNet: EF on the 'fc' will harm the performance of ADTOPK and AllchannelTopK
# VGG16: 'features.0' should not be compressed
# VGG16: EF on the 'classifier.6' will harm the performance
# LSTM: 'rnn.weight_hh' should not be compressed
def check_not_compress(params, name, tensor):    
    if tensor.dim() == 1:
        return True
    if 'features.0' in name:
        return True
    if 'rnn.weight_hh' in name:
        return True

    return False

def check_not_ef(params, name, tensor):
    compressor_name = params.get('compressor', 'none')
    if 'adtopk' in compressor_name or 'alldimensiontopk' in compressor_name:
        if 'fc' in name:
            return True
    
    if 'classifier.6' in name:
        return True

    return False 