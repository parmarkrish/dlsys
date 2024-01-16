# DLSys
This repository contains my solutions to the [DLSys](https://dlsyscourse.org/) online course

## HW0: Simple Nets in Numpy
[Notebook](https://github.com/parmarkrish/dlsys/blob/main/hw0/hw0.ipynb) | [Implementations](https://github.com/parmarkrish/dlsys/blob/main/hw0/src/simple_ml.py)
- MNIST loader/parser
- SGD softmax regression  and a two-layer neural net and respective training code in numpy (including backward pass :0)

## HW1: Autograd
[Notebook](https://github.com/parmarkrish/dlsys/blob/main/hw1/hw1.ipynb) | [Ops Implementations](https://github.com/parmarkrish/dlsys/blob/main/hw1/python/needle/ops.py) | [Autograd Implementation (relevant functions: compute_gradient_of_variables, find_topo_sort and topo_sort_dfs)](https://github.com/parmarkrish/dlsys/blob/main/hw1/python/needle/autograd.py)
- Forward/backward computation of Ops such as `PowerScalar`, `EWiseDiv`, `DivScalar`, `Matmul`, `Summation`, `BroadcastTo`, `Reshape`, `Negate`, `Transpose`
- Reverse mode differentation (autograd) and topological sort

## HW2: Neural Network Library
[Notebook](https://github.com/parmarkrish/dlsys/blob/main/hw2/hw2.ipynb) | [Init Implementations](https://github.com/parmarkrish/dlsys/blob/main/hw2/python/needle/init.py) | [NN Modules Implementation](https://github.com/parmarkrish/dlsys/blob/main/hw2/python/needle/nn.py)
- Neural Network Initalization schemes (`xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`)
- Various NN modules such as `Linear`, `Sequential`, `SoftmaxLoss`, `BatchNorm`, `LayerNorm` and `Dropout` 
- Optimizers such as `SGD` and `Adam`
- Data primitives such as `DataLoader` and `Dataset`
- Computer vision transforms such as `flip_horizontal` and `random_crop`
- Implementation and training of simple MLP ResNet on CIFAR10

## HW3: Building an NDArray Library
[Notebook](https://github.com/parmarkrish/dlsys/blob/main/hw3/hw3.ipynb) | [NDArray Python Implementation](https://github.com/parmarkrish/dlsys/blob/main/hw3/python/needle/backend_ndarray/ndarray.py) | [NDArray CPU Backend](https://github.com/parmarkrish/dlsys/blob/main/hw3/src/ndarray_backend_cpu.cc) | [NDArray CUDA Backend](https://github.com/parmarkrish/dlsys/blob/main/hw3/src/ndarray_backend_cuda.cu) \
Numpy-like array library with **CPU** and **CUDA** backend that supports:
1. Movement ops such as `reshape`, `permute`, `broadcast`, `__getitem__` 
2. Elementwise and scalar ops (`mul`, `div`, `power`, `max`, `log`, `exp`, `tanh`)
3. Reduction such as `ReduceSum` and `ReduceMax`
4. Matrix multplication

## HW4: Convolutional Neural Networks & Language Modeling
[Notebook](https://github.com/parmarkrish/dlsys/blob/main/hw4/hw4.ipynb) | [NDArray Methods](https://github.com/parmarkrish/dlsys/blob/main/hw4/python/needle/backend_ndarray/ndarray.py) | [Conv Op Implementation](https://github.com/parmarkrish/dlsys/blob/main/hw4/python/needle/ops.py)
- More NDArray methods such as `stack` and `split`
- CNN related ops such as `pad`, `flip`, `dilate` 
- Forward and backward pass implemention of im2col `Conv`
- ResNet9 model implementation
- NOTE: Some sections were completed without GPU due to Colab limits, hence the failed test cases involving the GPU.  I've skipped sections on language modeling as it is redundent with my RNN & LSTM work in CS231n and CS224n
