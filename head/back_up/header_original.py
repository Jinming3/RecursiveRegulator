"""
original, do not put test here

"""

import torch
import torch.nn as nn
import numpy as np
from torch.jit import Final
from typing import List


# -------- use this model original  ----------
class MechanicalSystem(nn.Module):

    def __init__(self, dt, n_x=2, init_small=True):
        super(MechanicalSystem, self).__init__()
        self.dt = dt  # sampling time
        self.hidden = 64
        self.net = nn.Sequential(nn.Linear(n_x + 1, self.hidden),  # 3*1
                                 # nn.LeakyReLU(negative_slope=0.4),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden, 1))

        if init_small:
            for i in self.net.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
                    nn.init.constant_(i.bias, val=0)

    def forward(self, x1, u1):
        list_dx: List[torch.Tensor]
        in_xu = torch.cat((x1, u1), -1)
        dv = self.net(in_xu) / self.dt  # v, dv = net(x, v)
        list_dx = [x1[..., [1]], dv]  # [dot x=v, dot v = a]
        dx = torch.cat(list_dx, -1)
        return dx


# --------- simulator original  -----------------------
class ForwardEuler(nn.Module):

    def __init__(self, model, dt=1.0):
        super(ForwardEuler, self).__init__()
        self.model = model
        self.dt = dt

    def forward(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        xhat_list: List[torch.Tensor] = []
        x_step = x0
        for u_step in u.split(1):
            u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
            dx = self.model(x_step, u_step)
            x_step = x_step + self.dt * dx
            xhat_list += [x_step]

        xhat = torch.stack(xhat_list, 0)
        return xhat


class RK4(nn.Module):
    def __init__(self, model, dt):
        super(RK4, self).__init__()
        self.model = model
        self.dt = dt

    def forward(self, x0, u):
        xhat_list: List[torch.Tensor] = []
        # xhat_list = []
        x_step = x0
        for u_step in u.split(1):
            u_step = u_step.squeeze(0)
            xhat_list += [x_step]
            k1 = self.model(x_step, u_step)
            k2 = self.model(x_step + self.dt * k1 / 2.0, u_step)
            k3 = self.model(x_step + self.dt * k2 / 2.0, u_step)
            k4 = self.model(x_step + self.dt * k3, u_step)
            x_step = x_step + self.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            # xhat_list += [x_step]  #  give result, but far
        xhat = torch.stack(xhat_list, 0)
        return xhat


# --------- other --------------------
def R2(Y_sys, Yhat):
    s1 = np.sum((Y_sys - Yhat) ** 2)
    mean = np.mean(Y_sys)
    s2 = np.sum((Y_sys - mean) ** 2)
    return 1.0 - s1 / s2


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


def vel(pos, dt):
    v_est = np.concatenate((np.array([0]), np.diff(pos[:, 0])))
    v_est = v_est.reshape(-1, 1) / dt
    return v_est
