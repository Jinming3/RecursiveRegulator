"""
import sys, os
sys.path.append(os.path.join("F:/Project/head/"))
"""
import numpy as np


def R2(Y_sys, Yhat):
    """
    R-square metrics
    :param Y_sys: size N sequence
    :param Yhat: size N sequence
    :return:
    """
    s1 = np.sum((Y_sys - Yhat) ** 2)
    mean = np.mean(Y_sys)
    s2 = np.sum((Y_sys - mean) ** 2)

    return 1.0 - s1 / s2


def mse(Y_sys, Yhat):
    s = np.sum((Y_sys - Yhat) ** 2)
    m = s/len(Y_sys)
    return m



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


def split(Y, batch=1):

    length = len(Y)
    number = int(length/batch)
    Y_return = np.array_split(Y, number)
    return Y_return

























