# coding: utf-8
import argparse
import time
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import horovod.torch as hvd

import datahelper
import model
from torch.optim import lr_scheduler


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='/home/user/mzq/workspaces/project/grace/examples/torch/nlp/data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='Transformer',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')

parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')


args = parser.parse_args()

# horovod 0
hvd.init()


# horovod 1
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())
    # print(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)


# if args.cuda:
device = torch.device("cuda")

###############################################################################
# Load data
###############################################################################

corpus = datahelper.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)
    # return data.to(device)

size = hvd.size()

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)

minibatch_size = train_data.size()[0] // size
train_datas = torch.chunk(train_data, size , dim=0)
train_data = train_datas[hvd.rank()]

val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    # model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
    # model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
print(model)
criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train(optimizer, communicator):
    # Turn on training mode which enables dropout.

    communicator.clear_iter()     
    adjust_learning_rate(epoch, communicator) 
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-lr)
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0 and hvd.local_rank()==0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        
        communicator.update_iter()

lr =  1.0 

best_val_loss = None
optimizer = optim.SGD(model.parameters(), lr=lr)


from convergence_eval.compressor import TopKCompressor, AllChannelTopkCompressor 
from convergence_eval.memory import ResidualMemory 
from convergence_eval.comm import AllgatherEval  

compressor = TopKCompressor(0.01) 
compressor = AllChannelTopkCompressor(0.01) 
memory = ResidualMemory() 
communicator = AllgatherEval(compressor=compressor, memory=memory, world_size=hvd.size()) 


# Horovod: wrap optimizer with DistributedOptimizer.
# 得到一个分布式的SGD优化器
optimizer = hvd.DistributedOptimizer(
    optimizer, communicator=communicator, named_parameters=model.named_parameters(),op=hvd.Adasum if args.use_adasum else hvd.Average)

if hvd.rank() == 0:
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())          

# def adjust_learning_rate(epoch, communicator):   
#     warmup_factor = 1. / 1000      
#     warmup_iters = 10     
#     lr_adj = 1.0     
#     if epoch >= warmup_iters:         
#         lr_adj = 0.95**(epoch-warmup_iters)     
#     else:         
#         alpha = float(epoch) / warmup_iters         
#         lr_adj = warmup_factor * (1 - alpha) + alpha             
#     for param_group in optimizer.param_groups:                  
#         param_group['lr'] = 1.0 * lr_adj
    
#     communicator.lr = 1.0 * lr_adj


def adjust_learning_rate(epoch, communicator):   
    lr_adj = 0.98**epoch   
    for param_group in optimizer.param_groups:                  
        param_group['lr'] = 1.0 * lr_adj
    
    communicator.lr = 1.0 * lr_adj




hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)


# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        if hvd.rank() == 0:
            communicator.clear_klist()
        train(optimizer, communicator)
        val_loss = evaluate(val_data)
        if hvd.rank() == 0:
            print('-' * 89)
            communicator.update_eklist()
            communicator.print_eklist() 
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
        
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

import numpy as np           
ans = np.array(communicator.e_klist)           
np.savetxt("./data/transformer_001.txt", ans)   

# Run on test data.
if hvd.rank() == 0:
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
    print('=' * 89)
