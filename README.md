# DLSys
This repository contains my solutions to the [DLSys](https://dlsyscourse.org/) online course

## HW0
- Basic implementation of softmax regression algorithm and a two-layer neural net
- Implemented MNIST loader

## HW1
- Implemented forward/backward computation of various ops
- Implemented topo-sort algorithm and reverse-mode auto-differentation

## HW2
- Implemented various components of NN library such as initialization, common layers (`Linear`, `Batchnorm`), and optimizers (`SGD`, `Adam`)
- Implemented image transformations used in data augmentation such as `RandomFlipHorizontal` and `RandomCrop`
- Created dataset and dataloader classes
- Trained simple MLP ResNet on CIFAR10

## HW3
- Built NDArray library (with CPU and CUDA backend) supporting various ops such as `reshape`, `permute`, `reductions` and `matmul`

## HW4
- Implemented other ops useful in ConvNets such as padding, dialate and im2col Conv
- TODO: Test Conv on GPU, implement `LSTM` and `Penn Treebank dataset`