"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        print(f'init linear with dim: {in_features}, {out_features}')
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
          self.bias = Parameter(ops.transpose(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)))
        else:
          self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Z = ops.matmul(X, self.weight)
        if self.bias:
          Z += ops.broadcast_to(self.bias, shape=Z.shape)
        return Z
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        return X.reshape((batch_size, np.prod(X.shape)//batch_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def __init__(self):
      super().__init__()
      print('init relu')

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
          #print(m) # DEBUG
          x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        #print('softmax input dtype: ', logits.dtype)
        y_one_hot = init.one_hot(logits.shape[1], y)
        #print('y_one_hot dtype: ', y_one_hot.dtype)
        temp = ops.summation(ops.logsumexp(logits, axes=(1,)) - ops.summation(logits * y_one_hot, axes=(1,)))
        #print('in softmax loss without divide: ', temp.dtype)
        #print('logits.shape[0] dtype: ', logits.shape[0].dtype)
        ret = temp / logits.shape[0]
        #print('RET DTYPE', ret.dtype)
        return ret
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        print('init batchnorm')
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        # NOTE: comment this out
        ### BEGIN YOUR SOLUTION
        if self.training:
          N = x.shape[0]
          mean = 1/N * x.sum(axes=(0,))
          x_mean = x - mean.broadcast_to(x.shape)
          var = 1/N * (x_mean**2).sum(axes=(0,))
        
          self.running_mean.data = (1-self.momentum)*self.running_mean.data + self.momentum*(mean.data)
          self.running_var.data = (1-self.momentum)*self.running_var.data + self.momentum*var.data

          std = (var + self.eps)**0.5
          x_norm = x_mean / std.broadcast_to(x.shape)
        else:
          x_running_mean = x - self.running_mean.broadcast_to(x.shape)
          running_std = (self.running_var + self.eps)**0.5
          x_norm = x_running_mean / running_std.broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * x_norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION
        '''
        ### BEGIN YOUR SOLUTION
        batch = x.shape[0]
        if self.training:
            mean = ops.summation(x, 0) / batch
            self.running_mean = self.momentum * mean + \
                (1 - self.momentum) * self.running_mean
            mean = ops.broadcast_to(ops.reshape(mean, (1, self.dim)), x.shape)

            std = ops.summation(ops.power_scalar(x - mean, 2), 0) / batch
            self.running_var = self.momentum * std + \
                (1 - self.momentum) * self.running_var
            std = ops.broadcast_to(ops.reshape(std, (1, self.dim)), x.shape)

            x = (x - mean) / ops.power_scalar(std + self.eps, 0.5) * \
                ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape) \
                + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
            return x
        else:
            x = (x - self.running_mean) / ((self.running_var + self.eps) ** 0.5)
            return x
        ### END YOUR SOLUTION
        '''


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        print('init layer norm')
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N = x.shape[0]
        mean = ops.reshape(1/self.dim * ops.summation(x, axes=(1,)), (N, 1))
        x_mean = x - ops.broadcast_to(mean, x.shape)
        var = self.eps + ops.reshape(1/self.dim * ops.summation(x_mean**2, axes=(1,)), (N, 1))
        std = var**0.5
        x_norm = x_mean / ops.broadcast_to(std, x.shape)
        return ops.broadcast_to(self.weight, x.shape) * x_norm + ops.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        print('init dropout')
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training and self.p > 0.0:
          mask = init.randb(*x.shape, p=1-self.p) / (1-self.p)
          x = x * mask
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        print('init res')
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
