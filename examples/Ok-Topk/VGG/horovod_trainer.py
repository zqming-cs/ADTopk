# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import torch
import numpy as np
import argparse, os
import settings
import utils
import logging
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# comm.Set_errhandler(MPI.ERRORS_RETURN)
import torch.optim as optim
from grace_dll.torch.helper import grace_from_params
from torchvision import datasets, transforms, models

from dl_trainer import DLTrainer, _support_datasets, _support_dnns
import horovod.torch as hvd
from tensorboardX import SummaryWriter
writer = None

from settings import logger, formatter
import numpy as np
import matplotlib.pyplot as plt
import time



y_loss = {}  # Loss history
y_loss['train'] = []
y_loss['test'] = []
y_acc = {}
y_acc['train'] = []
y_acc['test'] = []
x_test_epoch_time = []
x_train_epoch_time = []
x_epoch = []


def ssgd_with_horovod(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps = 1):
    rank = hvd.rank()
    # torch.cuda.set_device(rank%nwpernode)
    torch.cuda.set_device(rank)
    
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix='allreduce', pretrain=pretrain, num_steps=num_steps, tb_writer=writer)

    init_epoch = torch.ones(1) * trainer.get_train_epoch()
    init_iter = torch.ones(1) * trainer.get_train_iter()
    trainer.set_train_epoch(int(hvd.broadcast(init_epoch, root_rank=0)[0]))
    trainer.set_train_iter(int(hvd.broadcast(init_iter, root_rank=0)[0]))
    
    # model_state=hvd.broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    # trainer.net.load_state_dict(model_state)
    
    params = {'compressor': 'adtopk', 'memory': 'residual', 'communicator': 'allgather','model_named_parameters':trainer.net.named_parameters()}
   
    # Allreduce
    # params = {'compressor': 'topk', 'memory': 'residual', 'communicator': 'allreduce'}

    # Broadcast
    # params = {'compressor': 'topk', 'memory': 'residual', 'communicator': 'broadcast'}
    grc = grace_from_params(params)

    # Horovod: wrap optimizer with DistributedOptimizer.
    # 得到一个分布式的SGD优化器
    # optimizer = hvd.DistributedOptimizer(
    #     trainer.optimizer, grc, named_parameters=model.named_parameters())
    
    optimizer = hvd.DistributedOptimizer(trainer.optimizer, grace=grc, named_parameters=trainer.net.named_parameters())
    # optimizer = dopt.DistributedOptimizer(trainer.optimizer, trainer.net.named_parameters(), compression=compressor, is_sparse=is_sparse, err_handler=_error_handler, layerwise_times=None, sigma_scale=sigma_scale, density=density, norm_clip=norm_clip, writer=writer)



    trainer.update_optimizer(optimizer)
    iters_per_epoch = trainer.get_num_of_training_samples() // (nworkers * batch_size * nsteps_update)
    
    hvd.broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    
    times = []
    display = 20 if iters_per_epoch > 20 else iters_per_epoch-1
    for epoch in range(1, max_epochs+1):
        
        times_array=[]
        
        
        hidden = None
        if dnn == 'lstm':
            hidden = trainer.net.init_hidden()
        for i in range(1, iters_per_epoch+1):
            s = time.time()
            optimizer.zero_grad()
            for j in range(nsteps_update):
                if j < nsteps_update - 1 and nsteps_update > 1:
                    optimizer.local = True
                else:
                    optimizer.local = False
                if dnn == 'lstm':
                    _, hidden = trainer.train_adtopk(1, hidden=hidden)
                else:
                    trainer.train_adtopk(1,curr_iter=i)
            if dnn == 'lstm':
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
            trainer.update_model()
            times.append(time.time()-s)
            
            if i % display == 0 and i > 0 and hvd.rank()==0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f. Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
            
            times_array.append(time.time()-s)
        
        # Per Epoch Training Time
        if rank == 0:
            print('Epcoh:',epoch)
            per_epoch_train_time=np.sum(times_array)
            print('Training Time:',per_epoch_train_time)
            
            if len(x_train_epoch_time)==0:                
                x_train_epoch_time.append(per_epoch_train_time)
            else:
                x_train_epoch_time.append(x_train_epoch_time[-1]+per_epoch_train_time)
    
    if rank == 0:
        y_acc['train']=trainer.avg_train_acc_array
        y_acc['test']=trainer.test_acc_array
        
        y_loss['train']=trainer.avg_train_loss_array
        y_loss['test']=trainer.test_loss_array
        
        draw_curve(max_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AllReduce trainer")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nsteps-update', type=int, default=1)
    parser.add_argument('--nworkers', type=int, default=1, help='Just for experiments, and it cannot be used in production')
    parser.add_argument('--nwpernode', type=int, default=1, help='Number of workers per node')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=_support_datasets, help='Specify the dataset for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=_support_dnns, help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    parser.add_argument('--max-epochs', type=int, default=90, help='Default maximum epochs to train')
    parser.add_argument('--pretrain', type=str, default=None, help='Specify the pretrain path')
    parser.add_argument('--num-steps', type=int, default=35)
    parser.set_defaults(compression=False)
    args = parser.parse_args()
    batch_size = args.batch_size * args.nsteps_update
    prefix = settings.PREFIX
    logdir = 'allreduce-%s/%s-n%d-bs%d-lr%.4f-ns%d' % (prefix, args.dnn, args.nworkers, batch_size, args.lr, args.nsteps_update) 
    relative_path = './logs/%s'%logdir
    utils.create_path(relative_path)
    rank = 0
    if args.nworkers > 1:
        hvd.init()
        rank = hvd.rank()
    if rank == 0:
        tb_runs = './runs/%s'%logdir
        writer = SummaryWriter(tb_runs)
    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    
    hdlr.setFormatter(formatter)
    if rank == 0:
        logger.addHandler(hdlr) 
        logger.info('Configurations: %s', args)
    ssgd_with_horovod(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps)
