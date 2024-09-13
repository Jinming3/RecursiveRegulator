"""
recursive regulator callable package
"""
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.jit import Final
from typing import List, Tuple, Any
from pem import PEM, mse


def R2(Y_sys, Yhat):
    s1 = np.sum((Y_sys - Yhat) ** 2)
    mean = np.mean(Y_sys)
    s2 = np.sum((Y_sys - mean) ** 2)
    return 1.0 - s1 / s2


def normalize(x, r=1):  # 1-dimension
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
    out = np.array(out, dtype=np.float32)
    return out


class NeuralStateSpaceModel_i(nn.Module):
    n_x: Final[int]
    n_u: Final[int]
    n_feat: Final[int]

    def __init__(self, n_x=2, n_u=1, n_feat=64, scale_dx=1.0, init_small=True, activation='relu'):
        super(NeuralStateSpaceModel_i, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = n_feat
        self.scale_dx = scale_dx

        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'softplus':
            activation = nn.Softplus()
        elif activation == 'tanh':
            activation = nn.Tanh()

        self.phi_k = nn.Sequential(nn.Linear(n_x + n_u, self.n_feat),  # 3*1
                                   activation)

        self.inv_phi = nn.Linear(self.n_feat, n_x)



        if init_small:
            for m in self.phi_k.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)
            for m in self.inv_phi.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, in_x, in_u):
        in_xu = torch.cat((in_x, in_u), -1)  # concatenate x and u over the last dimension to create the [xu] input
        self.out_k = self.phi_k(in_xu)
        dx = self.inv_phi(self.out_k)
        dx = dx * self.scale_dx
        return dx


class MechanicalSystem_i(nn.Module):

    def __init__(self, dt, n_x=2, init_small=True):
        super(MechanicalSystem_i, self).__init__()
        self.dt = dt  # sampling time
        self.hidden = 64
        self.phi_k = nn.Sequential(nn.Linear(n_x + 1, self.hidden),  # 3*1
                                 # nn.LeakyReLU(negative_slope=0.4),
                                 nn.ReLU())

        self.inv_phi = nn.Linear(self.hidden, 1)


        if init_small:
            for i in self.phi_k.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
                    nn.init.constant_(i.bias, val=0)
            for i in self.inv_phi.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
                    nn.init.constant_(i.bias, val=0)

    def forward(self, x1, u1):
        list_dx: List[torch.Tensor]
        in_xu = torch.cat((x1, u1), -1)
        self.out_k = self.phi_k(in_xu)
        out_inv = self.inv_phi(self.out_k)
        dv = out_inv / self.dt  # v, dv = net(x, v)

        list_dx = [x1[..., [1]], dv]  # [dot x=v, dot v = a]
        dx = torch.cat(list_dx, -1)
        return dx


class ForwardEuler_i(nn.Module):

    def __init__(self, model, dt):
        super(ForwardEuler_i, self).__init__()
        self.model = model
        self.dt = dt

    def forward(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        xhat_list = list()
        x_step = x0
        for u_step in u.split(1):
            u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
            dx = self.model(x_step, u_step)
            x_step = x_step + dx * self.dt
            xhat_list += [x_step]

        xhat = torch.stack(xhat_list, 0)
        return xhat



class ForwardEulerPEM(nn.Module):  # use steps or R2 as switch

    def __init__(self, model,
                 factor,
                 dt, N, update, threshold1=1, threshold2=1,
                 sensitivity=600, train=0):

        super(ForwardEulerPEM, self).__init__()
        self.factor = factor
        self.model = model
        self.dt = dt
        self.N = N

        self.update = update  # choose case
        if train == 0:
            self.train = int(N)
        else:
            self.train = train  # stop update state space model

        self.threshold1 = threshold1  # start update
        self.threshold2 = threshold2  # stop update
        self.sensitivity = sensitivity  # a sequence to monitor R2
        self.stop = []  # time and r2
        self.correction = [] # time and r2
        self.xhat_data = np.zeros((N, 2))

    def forward(self, x0: torch.Tensor, u: torch.Tensor, y):
        x_step = x0
        
        # ------------------
        q = 0

        while q < self.N:

            if self.update == 1200:
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt + torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect output of NN
                q = q + 1
                while q < self.train:
                    u_step = u[q]
                    dx = self.model(x_step, u_step)
                    x_step = x_step + dx * self.dt
                    y_nn = x_step[:, 0].clone().detach().numpy()
                    u_in = self.model.out_k.clone().detach().numpy().T
                    self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                    x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                    self.xhat_data[q, :] = x_out
                    x_step = torch.tensor(x_out, dtype=torch.float32)
                    q = q+1

                u_in = self.model.out_k.clone().detach().numpy().T

                self.factor.pem_one(0, u_in, on=False)

        return self.xhat_data




