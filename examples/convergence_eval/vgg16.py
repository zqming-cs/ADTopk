import sys
sys.path.append("../cv/")  
import torch
import argparse
import os
import matplotlib.pyplot as plt

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import math
from tqdm import tqdm
from utils import get_network
import numpy as np

import time
import matplotlib
matplotlib.use('agg')
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Cifar100 Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--net', default='resnet50',type=str, required=True, help='net type')
parser.add_argument('--model-net', default='vgg16',type=str, help='net type')
parser.add_argument('--train-dir', default=os.path.expanduser('~/cifar100/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/cifar100/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./save_model/cifar100_vgg16/logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./save_model/cifar100_vgg16/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')

parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or not')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['test'] = []
y_err = {}
y_err['train'] = []
y_err['test'] = []
x_test_epoch_time = []
x_train_epoch_time = []
x_epoch = []


def train(epoch, communicator):

    communicator.clear_iter() 

    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx, communicator)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            
            output = model(data)
            train_accuracy.update(accuracy(output, target))
            loss = F.cross_entropy(output, target)
            train_loss.update(loss)
            loss.backward()
            
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)
            communicator.update_iter()
    
    # train_average_loss = metric_average(train_loss, 'train_loss')
    # train_average_accuracy = metric_average(train_accuracy, 'train_accuracy')
    train_average_loss = train_loss.avg.item()
    train_average_accuracy = train_accuracy.avg.item()

    if hvd.rank() == 0:
        print('\nTrain set: Average loss: {:.4f}, Train Accuracy: {:.2f}%\n'.format(
                train_average_loss, 100. * train_average_accuracy))

def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    # if log_writer:
    #     log_writer.add_scalar('val/loss', val_loss.avg, epoch)
    #     log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)

    # Horovod: average metric values across workers.
    test_loss = val_loss.avg.item()
    test_accuracy = val_accuracy.avg.item()

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * test_accuracy))
    

# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx, communicator):

    if epoch < 40:
        lr_adj = 1e-1
    else:
        lr_adj = 1e-2

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj
    communicator.lr = lr_adj * 1.0


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

def get_start_time():
        # torch.cuda.synchronize()
        start_time = time.time()
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('start_time_str=', start_time_str)
        # result = model(input)
        return start_time

def get_end_time(start_time):
        # torch.cuda.synchronize()
        end_time = time.time()
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('end_time_str=', end_time_str)
        print('end_time-start_time=', end_time-start_time)



if __name__ == '__main__':
    current_epoch=0
    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.

    resume_from_epoch = 0

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    # log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'
    
    CIFAR10_TRAIN_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    CIFAR10_TRAIN_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    # CIFAR100
    train_dataset = \
        datasets.CIFAR100(args.train_dir,
                             train=True,
                             download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=CIFAR10_TRAIN_MEAN,
                                                      std=CIFAR10_TRAIN_STD)
                             ]))
    
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)
    # CIFAR100
    val_dataset = \
        datasets.CIFAR100(args.val_dir,
                             train=False,
                             download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=CIFAR10_TRAIN_MEAN,
                                                      std=CIFAR10_TRAIN_STD)
                             ]))    
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)

    # Set up standard ResNet-50 model.
    # model = models.resnet50()
    model=get_network(args)

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(),
                          lr=(args.base_lr *
                              lr_scaler),
                          momentum=args.momentum, weight_decay=args.wd)

    # Horovod: (optional) compression algorithm.
    # compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    
    # from ADTopklib.helper import get_communicator
    # params = {'compressor': 'allchanneltopk', 'memory': 'residual', 'communicator': 'allgather'}
    # communicator = get_communicator(params)

    from convergence_eval.compressor import TopKCompressor, AllChannelTopkCompressor     
    from convergence_eval.memory import ResidualMemory     
    from convergence_eval.comm import AllgatherEval      
    compressor = AllChannelTopkCompressor(0.05)     
    memory = ResidualMemory()     
    communicator = AllgatherEval(compressor=compressor, memory=memory, world_size=hvd.size())  



    # Horovod: wrap optimizer with DistributedOptimizer.
    # 得到一个分布式的SGD优化器
    optimizer = hvd.DistributedOptimizer(
        optimizer, communicator, named_parameters=model.named_parameters(),op=hvd.Adasum if args.use_adasum else hvd.Average,
    gradient_predivide_factor=args.gradient_predivide_factor)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    for epoch in range(0, args.epochs):
        if hvd.rank() == 0:
            communicator.clear_klist()
        train(epoch, communicator)
        validate(epoch)
        if hvd.rank() == 0:
            communicator.update_eklist()
            communicator.print_eklist()
    
    if hvd.rank() == 0:         
        import numpy as np         
        ans = np.array(communicator.e_klist)         
        np.savetxt("./data/vgg16_005.txt", ans)         