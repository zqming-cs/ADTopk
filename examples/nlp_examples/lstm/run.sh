#!/bin/bash

func() {
    echo "Usage:"
    echo "run.sh [-d dataset] [-c compressor] [-e epochs]"
    echo "dataset:      wikitext-2, wikitext-103"
    echo "compressor:   none, adtopk, allchanneltopk, globaltopk, dgc, gaussiank, redsync, sidco"
    exit -1
}

while getopts 'h:d:c:e:' OPT; do
    case $OPT in
        d) dataset="$OPTARG";;
        c) compressor="$OPTARG";;
        e) epochs="$OPTARG";;
        h) func;;
        ?) func;;
    esac
done

dataset="${dataset:-wikitext-2}"
compressor="${compressor:-adtopk}"
epochs="${epochs:-80}"


# example

HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_CACHE_CAPACITY=0 horovodrun -np 8 -H localhost:1,n16:1,n17:1,n18:1,n19:1,n20:1,n21:1,n22:1 python $(pwd)/train_lstm.py --dataset ${dataset} --compressor ${compressor} --epochs ${epochs}




