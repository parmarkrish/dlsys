'''
import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    print('in res block')
    res = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    return nn.Sequential(nn.Residual(res), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # Order of init matter for tests (by making it into one line, last layer is the second to init for some reason)
    beginning = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU())
    res_sequence = nn.Sequential(*(ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)))
    return nn.Sequential(beginning, res_sequence, nn.Linear(hidden_dim, num_classes))
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.train() if opt else model.eval()
    softmax_loss = nn.SoftmaxLoss()
    err_history = []
    loss_history = []
    for X, y in dataloader:
      if model.training:
        opt.reset_grad()
      X = X.reshape((-1, 28 * 28))
      h = model(X)
      loss = softmax_loss(h, y)

      err = np.mean(np.argmax(h.numpy(), axis=1) != y.numpy())
      #print(err)
      err_history.append(np.mean(np.argmax(h.numpy(), axis=1) != y.numpy())) # error rate
      #print(loss.numpy())
      loss_history.append(loss.numpy()) 
      if model.training:
        loss.backward()
        opt.step()
    
    return sum(err_history) / len(err_history), sum(loss_history) / len(loss_history)
    



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset("./" + data_dir + "/" + "train-images-idx3-ubyte.gz", "./" + data_dir + "/" + "train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset("./" + data_dir + "/" + "t10k-images-idx3-ubyte.gz", "./" + data_dir + "/" + "t10k-labels-idx1-ubyte.gz")

    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size)

    model = MLPResNet(28 * 28, hidden_dim)
    model.train() # set to train mode

    adam = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)


    for i in range(epochs):
      epoch_train_err, epoch_train_loss = epoch(train_dataloader, model, opt=adam)
      print(f"Epoch {i+1}: train err: {epoch_train_err}, train loss: {epoch_train_loss}")

      if i == epochs - 1:
        model.eval() # set to eval mode
        print('in eval')
        epoch_test_err, epoch_test_loss = epoch(test_dataloader, model)
        print(f"Epoch {i+1}: test err: {epoch_test_err},  test loss: {epoch_test_loss}")

    # NOTE: might need to change err to acc
    return epoch_train_err, epoch_train_loss, epoch_test_err, epoch_test_loss
    ### END YOUR SOLUTION



'''
import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    print('in res block')
    res = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    return nn.Sequential(nn.Residual(res), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # Order of init matter for tests (by making it into one line, last layer is the second to init for some reason)
    beginning = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU())
    res_sequence = nn.Sequential(*(ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)))
    return nn.Sequential(beginning, res_sequence, nn.Linear(hidden_dim, num_classes))
    ### END YOUR SOLUTION

def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.train() if opt else model.eval()
    softmax_loss = nn.SoftmaxLoss()
    err_history = []
    loss_history = []
    for X, y in dataloader:
      if model.training:
        opt.reset_grad()
      X = X.reshape((-1, 28 * 28))
      h = model(X)
      loss = softmax_loss(h, y)

      err = np.mean(np.argmax(h.numpy(), axis=1) != y.numpy())
      #print(err)
      err_history.append(np.mean(np.argmax(h.numpy(), axis=1) != y.numpy())) # error rate
      #print(loss.numpy())
      loss_history.append(loss.numpy()) 
      if model.training:
        loss.backward()
        opt.step()
    
    return sum(err_history) / len(err_history), sum(loss_history) / len(loss_history)


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset("./" + data_dir + "/" + "train-images-idx3-ubyte.gz", "./" + data_dir + "/" + "train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset("./" + data_dir + "/" + "t10k-images-idx3-ubyte.gz", "./" + data_dir + "/" + "t10k-labels-idx1-ubyte.gz")

    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size)

    model = MLPResNet(28 * 28, hidden_dim)
    model.train() # set to train mode

    adam = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)


    for i in range(epochs):
      epoch_train_err, epoch_train_loss = epoch(train_dataloader, model, opt=adam)
      print(f"Epoch {i+1}: train err: {epoch_train_err}, train loss: {epoch_train_loss}")

      if i == epochs - 1:
        model.eval() # set to eval mode
        print('in eval')
        epoch_test_err, epoch_test_loss = epoch(test_dataloader, model)
        print(f"Epoch {i+1}: test err: {epoch_test_err},  test loss: {epoch_test_loss}")

    # NOTE: might need to change err to acc
    return epoch_train_err, epoch_train_loss, epoch_test_err, epoch_test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")