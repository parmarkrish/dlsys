import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)
# For some reason ResNet training test was failing, so here I was comparing my implementation 
# to another implementation. It is still failing, weird.
'''
class ResBlock(nn.Module):
  def __init__(self, a, b, device=None, dtype="float32"):
    super().__init__()
    self.resblock = nn.Sequential(
      nn.ConvBN(a, b, 3, 2, device=device),
      nn.Residual(
        nn.Sequential(
          nn.ConvBN(b, b, 3, 1, device=device), 
          nn.ConvBN(b, b, 3, 1, device=device)
        )
      )
    )
  
  def forward(self, x):
    return self.resblock(x)
  


class ResNet9(nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.convBN1 = nn.ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.resblock1 = ResBlock(16, 32, device=device, dtype=dtype)
        self.convBN2 = nn.ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.resblock2 = ResBlock(64, 128, device=device, dtype=dtype)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.convBN1(x)
        x = self.resblock1(x)
        x = self.convBN2(x)
        x = self.resblock2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        ### END YOUR SOLUTION
'''

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.block1 = nn.ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.block2 = nn.ConvBN(16, 32, 3, 2, device=device, dtype=dtype)
        self.res1 = nn.Residual(
            nn.Sequential(
                nn.ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
                nn.ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
            )
        )
        self.block3 = nn.ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.block4 = nn.ConvBN(64, 128, 3, 2, device=device, dtype=dtype)
        self.res2 = nn.Residual(
            nn.Sequential(
                nn.ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
                nn.ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
            )
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.block1(x)
        x = self.block2(x)
        x = self.res1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.res2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        ### END YOUR SOLUTION

class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)