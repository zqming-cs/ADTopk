# ADTopk
ADTopk is an all-dimension-based sparsification compression scheme for Distributed DNN Training Systems

## Introduction
This repository contains the codes for the paper: ADTopk: All-Dimension Top-k Sparsification Compression for Distributed DNN Training Systems. Key features include
- Distributed DNN training with various gradient sparsification methods including ADTopk, DGC, Gaussian-k, RedSync, OkTopk, and SIDCo.
- Evaluation of six widely-used DNN models including three image classification models (VGG-16, ResNet-50 and ResNet-101) and three natural language processing models (BERT-base, LSTM and Transformer).
- An all-dimension-based sparsification compression scheme (ADTopk) for Distributed DNN Training Systems

For more details about the algorithm, please refer to our papers.

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
