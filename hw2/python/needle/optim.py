"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for w in self.params:
          if self.momentum:
            if not self.u.get(w):
              self.u[w] = ndl.init.zeros(*w.shape) # init if empty
            self.u[w].data = self.momentum * self.u.get(w).data + (1 - self.momentum) * (w.grad.data + self.weight_decay * w.data)
            w.data = w.data - self.lr * self.u[w].data
          else: # if no momentum
            w.data = (1 - self.lr * self.weight_decay) * w.data - self.lr * w.grad.data
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for w in self.params:
          if not self.m.get(w):
              # init if empty
              self.m[w] = ndl.init.zeros(*w.shape)
              self.v[w] = ndl.init.zeros(*w.shape)
          w_grad_reg = w.grad.data + self.weight_decay * w.data # add regularization

          self.m[w].data = self.beta1 * self.m[w].data + (1-self.beta1) * w_grad_reg
          self.v[w].data = self.beta2 * self.v[w].data + (1-self.beta2) * w_grad_reg**2

          # bias correction
          m_hat = self.m[w].data / (1-self.beta1**self.t)
          v_hat = self.v[w].data / (1-self.beta2**self.t)

          w.data = w.data - (self.lr * m_hat) / (v_hat**0.5 + self.eps)
        ### END YOUR SOLUTION
