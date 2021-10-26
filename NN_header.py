import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NNFilter(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super(NNFilter, self).__init__()
        self.lrelu = F.leaky_relu
        self.relu = nn.ReLU()  # activation function
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, u, y):
        """
        filter out higher nonlinearity, put it before PEM
        """
        in_nn = torch.cat((u, y), -1)
        out = self.l1(in_nn)
        out = self.relu(out)
        out = self.l2(out)
        return out

    def forward1(self, in_nn):
        out = self.l1(in_nn)
        out = self.relu(out)
        out = self.l2(out)
        return out


def mse(y, yhat):
    """
    mean squared error
    :param y: system measurement
    :param yhat: prediction
    :return:
    """
    N = len(y)
    E = y - yhat
    sqE = (E*E).sum()
    msqE = sqE/N
    return msqE


def normalize(x, r=1):
    """
    normalize an array
    :param x: array
    :param r: new array of [-r, r]
    :return: new array
    """
    out = []
    mini = np.amin(x)
    maxi = np.amax(x)
    for j in range(len(x)):
        # norm = (x[i] - mini) / (maxi - mini)  # [0, 1]
        norm = 2 * r * (x[j] - mini) / (maxi - mini) - r
        out.append(norm)
    return np.array(out)
