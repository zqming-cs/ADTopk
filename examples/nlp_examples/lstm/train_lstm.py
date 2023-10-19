# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import horovod.torch as hvd

import datahelper
import model
from torch.optim import lr_scheduler
import numpy as np 
# is it faster?
torch.backends.cudnn.benchmark = True

# same hyperparameter scheme as word-language-model
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')

parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=500,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=500,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.1,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', default=False,
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--cudnn', action='store_true',
                    help='use cudnn optimized version. i.e. use RNN instead of RNNCell with for loop')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--resume', type=int, default=None,
                    help='if specified with the 1-indexed global epoch, loads the checkpoint and resumes training')

# parameters for adaptive softmax
parser.add_argument('--adaptivesoftmax', action='store_true',
                    help='use adaptive softmax during hidden state to output logits.'
                         'it uses less memory by approximating softmax of large vocabulary.')
parser.add_argument('--cutoffs', nargs="*", type=int, default=[10000, 50000, 100000],
                    help='cutoff values for adaptive softmax. list of integers.'
                         'optimal values are based on word frequencey and vocabulary size of the dataset.')

parser.add_argument('--compressor', default='actopk',type=str, help='compressor type')
parser.add_argument('--dataset', default='wikitext-2',type=str, help='dataset type')



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

dataset_path = '/data/dataset/nlp/lstm/' + args.dataset

corpus = datahelper.Corpus(dataset_path) 

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

size = hvd.size()

eval_batch_size = 10

train_datas = batchify(corpus.train, args.batch_size*size)

num_columns = train_datas.shape[1] 
column_indices = np.arange(num_columns)  

val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()


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
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].contiguous().view(-1)
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


def train(optimizer, train_data):

    # Turn on training mode which enables dropout.
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        optimizer.step()

        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0 and hvd.local_rank()==0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            # batches = train_data[0] // bptt
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
best_val_loss = None
lr=args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr*hvd.size(), weight_decay=1.2e-6) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

hvd.broadcast_parameters(model.state_dict(), root_rank=0) 
hvd.broadcast_optimizer_state(optimizer, root_rank=0) 


if args.compressor == 'none' or args.compressor == 'baseline':
    comm_params = {
        'comm_mode': 'allreduce',
        'compressor': 'none',
        'memory': 'none',
        'send_size_aresame':True,
        'model_named_parameters': model.named_parameters()
    }
else:
     comm_params = {
        'comm_mode': 'allgather_fast',
        'compressor': args.compressor,
        'memory': 'residual',
        'send_size_aresame':True,
        'model_named_parameters': model.named_parameters()
    }

import ADTopklib

# Horovod: wrap optimizer with DistributedOptimizer.
# 得到一个分布式的SGD优化器
optimizer = ADTopklib.DistributedOptimizer(
    optimizer, comm_params=comm_params, named_parameters=model.named_parameters())

if hvd.rank() == 0:
    print('===============model_named_parameters===============')
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size()) 


try:     
    for epoch in range(1, args.epochs+1):         
        epoch_start_time = time.time()

        np.random.seed(epoch)
        np.random.shuffle(column_indices)
        train_data = train_datas[:, column_indices]
        
        train_data = torch.chunk(train_data, size , dim=1)
        train_data = train_data[hvd.rank()]
        train(optimizer, train_data)
        val_loss = evaluate(val_data)
        

        scheduler.step(val_loss)         
        if hvd.rank() == 0:             
            print('-' * 89)
            tmp = time.time() - epoch_start_time           
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | ' 'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
            print('-' * 89)  
        
except KeyboardInterrupt:
    print('-' * 89)     
    print('Exiting from training early') 


if hvd.rank() == 0:     
    test_loss = evaluate(test_data)     
    print('=' * 89)     
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))     
    print('=' * 89) 
    ppl_test = [math.exp(test_loss)]
