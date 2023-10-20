""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(dnn_name):
    """ return given network
    """
    if dnn_name == 'resnet18':
        from models_cv.resnet import resnet18
        net = resnet18()
    elif dnn_name == 'resnet34':
        from models_cv.resnet import resnet34
        net = resnet34()
    elif dnn_name == 'resnet50':
        from models_cv.resnet import resnet50
        net = resnet50()
    elif dnn_name == 'resnet101':
        from models_cv.resnet import resnet101
        net = resnet101()
    elif dnn_name == 'resnet152':
        from models_cv.resnet import resnet152
        net = resnet152()
    
    elif dnn_name == 'vgg16':
        from models_cv.vgg import vgg16_bn
        net = vgg16_bn()
    elif dnn_name == 'vgg13':
        from models_cv.vgg import vgg13_bn
        net = vgg13_bn()
    elif dnn_name == 'vgg11':
        from models_cv.vgg import vgg11_bn
        net = vgg11_bn()
    elif dnn_name == 'vgg19':
        from models_cv.vgg import vgg19_bn
        net = vgg19_bn()
    elif dnn_name == 'densenet121':
        from models_cv.densenet import densenet121
        net = densenet121()
    elif dnn_name == 'densenet161':
        from models_cv.densenet import densenet161
        net = densenet161()
    elif dnn_name == 'densenet169':
        from models_cv.densenet import densenet169
        net = densenet169()
    elif dnn_name == 'densenet201':
        from models_cv.densenet import densenet201
        net = densenet201()
    elif dnn_name == 'googlenet':
        from models_cv.googlenet import googlenet
        net = googlenet()
    elif dnn_name == 'inceptionv3':
        from models_cv.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif dnn_name == 'inceptionv4':
        from models_cv.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif dnn_name == 'inceptionresnetv2':
        from models_cv.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif dnn_name == 'xception':
        from models_cv.xception import xception
        net = xception()
        
    elif dnn_name == 'preactresnet18':
        from models_cv.preactresnet import preactresnet18
        net = preactresnet18()
    elif dnn_name == 'preactresnet34':
        from models_cv.preactresnet import preactresnet34
        net = preactresnet34()
    elif dnn_name == 'preactresnet50':
        from models_cv.preactresnet import preactresnet50
        net = preactresnet50()
    elif dnn_name == 'preactresnet101':
        from models_cv.preactresnet import preactresnet101
        net = preactresnet101()
    elif dnn_name == 'preactresnet152':
        from models_cv.preactresnet import preactresnet152
        net = preactresnet152()
    elif dnn_name == 'resnext50':
        from models_cv.resnext import resnext50
        net = resnext50()
    elif dnn_name == 'resnext101':
        from models_cv.resnext import resnext101
        net = resnext101()
    elif dnn_name == 'resnext152':
        from models_cv.resnext import resnext152
        net = resnext152()
    elif dnn_name == 'shufflenet':
        from models_cv.shufflenet import shufflenet
        net = shufflenet()
    elif dnn_name == 'shufflenetv2':
        from models_cv.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif dnn_name == 'squeezenet':
        from models_cv.squeezenet import squeezenet
        net = squeezenet()
    elif dnn_name == 'mobilenet':
        from models_cv.mobilenet import mobilenet
        net = mobilenet()
    elif dnn_name == 'mobilenetv2':
        from models_cv.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif dnn_name == 'nasnet':
        from models_cv.nasnet import nasnet
        net = nasnet()
    elif dnn_name == 'attention56':
        from models_cv.attention import attention56
        net = attention56()
    elif dnn_name == 'attention92':
        from models_cv.attention import attention92
        net = attention92()
    elif dnn_name == 'seresnet18':
        from models_cv.senet import seresnet18
        net = seresnet18()
    elif dnn_name == 'seresnet34':
        from models_cv.senet import seresnet34
        net = seresnet34()
    elif dnn_name == 'seresnet50':
        from models_cv.senet import seresnet50
        net = seresnet50()
    elif dnn_name == 'seresnet101':
        from models_cv.senet import seresnet101
        net = seresnet101()
    elif dnn_name == 'seresnet152':
        from models_cv.senet import seresnet152
        net = seresnet152()
    elif dnn_name == 'wideresnet':
        from models_cv.wideresidual import wideresnet
        net = wideresnet()
    elif dnn_name == 'stochasticdepth18':
        from models_cv.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif dnn_name == 'stochasticdepth34':
        from models_cv.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif dnn_name == 'stochasticdepth50':
        from models_cv.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif dnn_name == 'stochasticdepth101':
        from models_cv.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]