# ADTopk
ADTopk is an an all-dimension Top-k sparsification scheme for Distributed DNN Training Systems, which selects the largest *k*
gradient elements from *all* dimensions, meaning that each dimension must provide some elements, so as to avoid the
dimension missing. Further, ADTopk enables each dimension to perform sorting locally within the elements of the dimension, and thus all dimensions can perform multiple local sorting independently and parallelly, instead of a single global sorting.

On top of ADTopk, we further propose an interleaving compression scheme and an efficient threshold estimation algorithm so as to enhance the performance of ADTopk. We build a sparsification compression distributed DNN training framework and implement a compression library containing state-of-the-art sparsification algorithms.

## Introduction
This repository contains the codes for the paper: ADTopk: All-Dimension Top-k Sparsification Compression for Distributed DNN Training Systems. 
In this paper, we observe that there exist two challenges when considering dimensions in the traditional Top-k sparsification:
- **Challenge #1 (Dimension missing).** 
Our observation shows that the traditional Top-k sparsification compression is likely to select elements which are centered in some of all dimensions, while some other dimensions are often missing (dimension missing). Dimension missing causes the DNN model to lose the opportunity to learn more representations from the missing dimensions, which may reduce the convergence accuracy of the model.
- **Challenge #2 (Single global sorting).** Our observation also shows that the traditional Top-k sparsification typically
selects the *k* largest elements of the gradient by a single global sorting operation on its entire elements, which causes
a low GPU core parallelism, thus limiting the training throughput.

In this paper, to address the above two challenges, we propose an All-Dimension Top-k sparsification scheme, called ADTopk. Our contributions include:

- We conduct measurement analysis and give two observations to show the traditional Top-k sparsification’s two
challenges: the dimension missing and the single global sorting. We propose ADTopk which performs sparsification on all dimensions rather than only some dimensions in the traditional Top-k sparsification to address the above two challenges.
- We design ADTopk, which leverages its all dimension based gradient sparsification to realize matrix-based sparsification and multiple local sorting which handles the dimension missing and the single global sorting, respectively. Atop ADTopk, we also propose an interleaving compression scheme that accelerates the convergence of the model and an efficient threshold estimation algorithm that reduces the compression overhead. We finally integrate ADTopk into Stochastic Gradient Descent (SGD) algorithm and prove its convergence.
- We implement a distributed DNN training framework with a compression library containing our ADTopk and state-of-the-art sparsification algorithms (DGC, Gaussiank, Redsync, and OkTopk). We also implement communication libraries (Allreduce, Allgather, and our proposed AllgatherFast) to support different threshold-based sparsification communication schemes.

For more details about the algorithm, please refer to our paper.

## Implementation
We implement our prototype system based on Horovod, a popular distributed deep-learning training framework. Based
on the framework, we implement ADTopk in PyTorch, which consists of an interleaving compression module, an
efficient threshold sparsification module, a communication and aggregation module, and a residual gradient error feedback module.

### Interleaving compression module
We implement two compression APIs: `AllDimensionTopk` and `TradTopk`. 

- The `AllDimensionTopk` takes as input the original gradient matrix, and it leverages torch.topk function as the basic compressor to select the absolute largest k elements from all input dimension. The number of elements selected on each dimension is k divided by the number of input dimensions, which is at least one. 
- The `TradTopk` first flattens the original gradient into a single tensor, and then it leverages torch.topk to select the absolute largest k elements (including values and indices) in the gradient vector.

### Efficient threshold sparsification module
In this module, we implement a `ThresholdTopk` function to adjust the frequency of interleaving compression and replace the `TradTopk`, which consists of a `monitor` and an `updater`. 

- The `monitor` is designed to monitor changes in the threshold and iteration.
- The `updater` periodically updates the threshold.

### Communication and aggregation module
To enable communication and aggregation of the compressed gradient, we rewrite the core component `DistributedOptimizer`
of Horovod. In `DistributedOptimizer`, we provide three collective communication primitives APIs for `Allreduce`, `Allgather`, and `AllgatherFast`.
- we leverage the `allreduce_async_` function to design `Allreduce` collective communication primitive to execute the communication and aggregation in the dense non-compression baseline. Allreduce is the most efficient operation, but it is not readily suitable for several scenarios.
- We use `allgather_async_` function to design `Allgather` primitive to execute the communication and aggregation in sparse ADTopk and other state-off-the-art sparsification compression methods. It does not perform any aggregation, only supports input gradients of different forms, and is well suited for sparsification when different nodes select gradient elements at non-overlapping indices.
- We also implement another alternative primitive called `AllgatherFast`, which speeds up communication by eliminating the gradient split step compared to `Allgather`.

### Residual gradient error feedback module
In this module, we implement an error feedback API, including includes a `memory.compensate` function that accumulates residual into a locally generated gradient and a `memory.update` function that calculates the difference between the compensated gradient and the compressed gradient to update the residual and store it in memory


## Installation
### Prerequisites
- CUDA-11.6
- NCCL-2.8.3
- PyTorch-1.3.+
- [OpenMPI-4.0.+](https://www-lb.open-mpi.org/software/ompi/v4.0/)
- [Horovod-0.27.+](https://github.com/horovod/horovod)

### Install ADTopk
```
git clone https://github.com/User/ADTopk.git
cd ADTopk
pip install -e .
```

## Training with ADTopk
To run CV jobs:
```
./examples/cv_examples/run.sh -d cifar100 -m vgg16 -c actopk -e 80
```
Assume that you have 8 GPUs on a single node and everything works well, you will see that there are 8 workers running at a single node training the VGG16 model for 80 epochs with the Cifar-100 dataset using ADTopk sparsification scheme.

To run NLP jobs:
```
./examples/nlp_examples/lstm/run.sh -d wikitext-2 -c actopk -e 80
```

## Papers
- ADTopk: All-Dimension Top-k Sparsification Compression for Distributed DNN Training Systems

## Referred Datasets
<!-- - Deep speech: [https://github.com/SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)
- PyTorch examples: [https://github.com/pytorch/examples](https://github.com/pytorch/examples) -->
- CIFAR-100: [https://www.cs.utoronto.ca/~kriz/cifar.html](https://www.cs.utoronto.ca/~kriz/cifar.html)
- ImageNet: [https://www.image-net.org/](https://www.image-net.org/)


## License
See [LICENSE](LICENSE).
