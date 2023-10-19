#!/bin/bash

func() {
    echo "Usage:"
    echo "run.sh [-d dataset] [-m model] [-c compressor] [-e epochs]"
    echo "dataset:      cifar100, imagenet"
    echo "model:        resnet50, resnet101, vgg16, vgg19"
    echo "compressor:   none, actopk, allchanneltopk, globaltopk, dgc, gaussiank, redsync, sidco"
    exit -1
}


while getopts 'h:d:m:c:e:' OPT; do
    case $OPT in
        d) dataset="$OPTARG";;
        m) model="$OPTARG";;
        c) compressor="$OPTARG";;
        e) epochs="$OPTARG";;
        h) func;;
        ?) func;;
    esac
done

dataset="${dataset:-cifar100}"
model="${model:-vgg16}"
compressor="${compressor:-actopk}"
epochs="${epochs:-80}"


# example
# ./run.sh -d cifar100 -m vgg16 -c actopk

# single node with 8 GPUs
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H localhost:8 python $(pwd)/cv_run.py --dataset ${dataset} --model-net ${model} --compressor ${compressor} --epochs ${epochs}

# single node with 1 GPU
# HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 1 -H localhost:1 python $(pwd)/cv_run.py --dataset ${dataset} --model-net ${model} --compressor ${compressor} --epochs ${epochs}

# two nodes 
# HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 2 -H node1:1,node2:1 python $(pwd)/cv_run.py --dataset ${dataset} --model-net ${model} --compressor ${compressor} --epochs ${epochs}

