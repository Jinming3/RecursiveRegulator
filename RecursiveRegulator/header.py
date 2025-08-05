"""
contains all callable functions for NN
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


def fit_index(y_true, y_pred, time_axis=0):
    """ Computes the per-channel fit index.

    The fit index is commonly used in System Identification. See the definitionin the System Identification Toolbox
    or in the paper 'Nonlinear System Identification: A User-Oriented Road Map',
    https://arxiv.org/abs/1902.00683, page 31.
    The fit index is computed separately on each channel.

    Parameters
    ----------
    y_true : np.array
        Array of true values.  If must be at least 2D.
    y_pred : np.array
        Array of predicted values.  If must be compatible with y_true'
    time_axis : int
        Time axis. All other axes define separate channels.

    Returns
    -------
    fit_val : np.array
        Array of r_squared value.

    """

    err_norm = np.linalg.norm(y_true - y_pred, axis=time_axis, ord=2)  # || y - y_pred ||
    y_mean = np.mean(y_true, axis=time_axis)
    err_mean_norm = np.linalg.norm(y_true - y_mean, ord=2)  # || y - y_mean ||
    fit_val = 100 * (1 - err_norm / err_mean_norm)

    return fit_val


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


class MechanicalSystem_qu(nn.Module):  # koopman NN

    def __init__(self, dt, n_x=2, init_small=True):
        super(MechanicalSystem_qu, self).__init__()
        self.dt = dt  # sampling time
        self.hidden = 64

        self.phi = nn.Sequential(nn.Linear(n_x, self.hidden),
                                 # nn.LeakyReLU(negative_slope=0.4),
                                 nn.ReLU())
        self.k = nn.Linear(self.hidden, self.hidden, bias=False)  # nonlinear, to lift x
        self.phi_b = nn.Linear(1, self.hidden, bias=False)  #  linear, to lift u
        self.inv_phi = nn.Linear(self.hidden, 1, bias=False)  #

        if init_small:
            for i in self.phi.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
                    nn.init.constant_(i.bias, val=0)
            for i in self.k.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
            for i in self.phi_b.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)

            for i in self.inv_phi.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)


    def forward(self, x1, u1):
        list_dx: List[torch.Tensor]
        self.out_q = self.phi(x1)
        self.out_k = self.k(self.out_q)
        self.out_b = self.phi_b(u1)
        self.out_qu = self.out_k + self.out_b
        out_inv = self.inv_phi(self.out_qu)
        dv = out_inv / self.dt  # v, dv = net(x, v)

        list_dx = [x1[..., [1]], dv]  # [dot x=v, dot v = a]
        dx = torch.cat(list_dx, -1)
        return dx


class NeuralStateSpaceModel_qu(nn.Module):  # use this when variables are not pos and vel, no derivative relation
    n_x: Final[int]
    n_u: Final[int]
    n_feat: Final[int]

    def __init__(self, n_x=2, n_u=1, n_feat=64, scale_dx=1.0, init_small=True, activation='relu'):
        super(NeuralStateSpaceModel_qu, self).__init__()
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

        self.phi = nn.Sequential(nn.Linear(n_x, self.n_feat),
                                 activation)
        self.k = nn.Linear(self.n_feat, self.n_feat, bias=False)
        self.phi_b = nn.Linear(n_u, self.n_feat, bias=False)

        self.inv_phi = nn.Linear(self.n_feat, n_x, bias=False)

        if init_small:
            for m in self.phi.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)
            for m in self.k.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
            for m in self.phi_b.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    # nn.init.constant_(m.bias, val=0)
            for m in self.inv_phi.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    # nn.init.constant_(m.bias, val=0)

    def forward(self, in_x, in_u):

        self.out_q = self.phi(in_x)
        self.out_k = self.k(self.out_q)
        self.out_b = self.phi_b(in_u)

        self.out_qu = self.out_k + self.out_b

        dx = self.inv_phi(self.out_qu)

        dx = dx * self.scale_dx
        return dx



class ForwardEuler(nn.Module):

    def __init__(self, model, dt):
        super(ForwardEuler, self).__init__()
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


#
# ----------------------<<< original ----- >>>>>>>>>works---------------
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
        n=self.factor.n
        self.update = update  # choose case
        if train == 0:
            self.train = int(N)
        else:
            self.train = train  # stop update pem theta

        self.threshold1 = threshold1  # start update
        self.threshold2 = threshold2  # stop update
        self.sensitivity = sensitivity  # an sequence to monitor R2
        self.stop = []  # time and r2
        self.correction = []  # time and r2
        self.pem_out = np.zeros((N, 2))
        self.xhat_data = np.zeros((N, n))

    def forward(self, x0: torch.Tensor, u: torch.Tensor, y):
        x_step = x0
        self.y_pem = []
        self.y_pem0 = []
        self.r2 = np.zeros(self.N)
        self.alter = torch.zeros(1, 2)
        self.err = np.zeros(self.N)  # |y-yhat|
        self.A_data = np.zeros((self.N, 2))
        self.B_data = np.zeros((self.N, 2))
        # ---------------
        self.on = []
        self.check = []
        # ------------------
        q = 0
        ignition_true = 0

        while q < self.N:
            # works, in use---->>>>>>>
            if self.update == 12010:  # case 1201, in paper! u_in = 65, regulator G q and u
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

                    u_in = np.concatenate((self.model.out_k.clone().detach().numpy().T, u_step), axis=0)  #[q+, u]
                    # u_in =self.model.out_k.clone().detach().numpy().T

                    self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                    x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                    # self.pem_out.append(self.factor.Xhat[:, 0])
                    self.xhat_data[q, :] = x_out
                    x_step = torch.tensor(x_out, dtype=torch.float32)
                    q = q + 1

                u_in = np.concatenate((self.model.out_k.clone().detach().numpy().T, u_step), axis=0)
                # u_in = self.model.out_k.clone().detach().numpy().T
                self.factor.pem_one(0, u_in, on=False)  #(y[q-1] - y_nn)*


            # test functions, ignore :---------
            if self.update == 0:# not updating, no PEM
                # simple forward Euler
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()
                # self.match = R2(y[q - self.sensitivity:q], self.xhat_data[q - self.sensitivity:q, 0])
                # print(f'R2 is {self.match} at data {q}')
                q = q + 1


            if self.update == 1:  # with bar_x in x_step, # update non-stop:
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt  #+ torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)
                # self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect output of NN
                y_nn = x_step[:, 0].clone().detach().numpy()
                u_in = y_nn
                self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                self.xhat_data[q, :] = x_out
                x_step = torch.tensor(x_out, dtype=torch.float32)  # ! update input to NN !
                match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                # match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                q = q + 1

            # update with threshold,  adding resting PEM  # use this
            if self.update == 5:
                u_step = u[q]
                dx = self.model(x_step, u_step)
                # y_nn = x_step[:, 0].clone().detach().numpy()
                # self.factor.pem_one(y[q] * 0 - y_nn, y_nn, on=False)  # for pem n-step ahead
                x_step = x_step + dx * self.dt + torch.tensor(self.factor.Xhat[:, 0],
                                                              dtype=torch.float32)  # not updating pem added
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect
                self.err[q] = y[q] - x_step[0, 0].clone().detach().numpy()
                # # --------------------------------------------------------
                # if q < 1000:
                #     y_nn = x_step[:, 0].clone().detach().numpy()
                #     u_in = y_nn
                #     self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                #     x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                #     self.xhat_data[q, :] = x_out  # collect
                #     x_step = torch.tensor(x_out, dtype=torch.float32)  # ! update input to NN !
                #
                # # --------------------------------------------------------
                self.y_pem0.append([self.factor.Xhat[0, 0], q])
                self.y_pem.append([None, q])

                # self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])
                match = R2(y[q - self.sensitivity:q, 0],
                           self.xhat_data[q - self.sensitivity:q, 0])  # check the dimension before use
                # match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])

                # if q > self.sensitivity:
                #     match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                # if q <= self.sensitivity:
                #     match = R2(y[0:q, 0], self.xhat_data[0:q, 0])
                match = round(match, 3)
                self.r2[q] = match
                if match < self.threshold1:
                    print(f'update at {q}, with R2= {match}')
                    self.correction.append([match, q])
                    while q < self.N:
                        u_step = u[q]
                        dx = self.model(x_step, u_step)
                        x_step = x_step + dx * self.dt
                        y_nn = x_step[:, 0].clone().detach().numpy()
                        u_in = x_step.clone().detach().numpy().T
                        self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                        # self.factor.pem_one(y[q] - y_nn, y_nn, on=True)

                        self.y_pem.append([self.factor.Xhat[0, 0], q])
                        self.y_pem0.append([None, q])

                        self.err[q] = y[q] - x_step[0, 0].clone().detach().numpy()
                        x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                        self.xhat_data[q, :] = x_out
                        x_step = torch.tensor(x_out, dtype=torch.float32)  # don't delete this ! update input to NN !
                        match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                        # match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])

                        # if q > self.sensitivity:
                        #     match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                        # if q <= self.sensitivity:
                        #     match = R2(y[0:q, 0], self.xhat_data[0:q, 0])
                        match = round(match, 3)
                        self.r2[q] = match
                        if match > self.threshold2:
                            self.stop.append([match, q])
                            print(f'finish at  {q}, with R2= {match}')
                            break
                        q = q + 1

                y_nn = x_step[:, 0].clone().detach().numpy()
                u_in = x_step.clone().detach().numpy().T
                self.factor.pem_one(y[q] - y_nn, u_in, on=False)
                # print(q)
                # self.factor.pem_one(y[q]*0 - y_nn, y_nn, on=False)  # for pem n-step ahead
                q = q + 1

            # same as 5, only for tank, y = x1
            if self.update == 6:  # for tanks, with regulator added
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt  #+ torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect

                # --------------------------------------------------------
                # y_nn = x_step[:, 0].clone().detach().numpy()
                # # u_in = self.xhat_data[q - 1, [0]]  # ==y_out
                # u_in = y_nn
                # self.factor.pem_one(y[q] - y_nn, u_in, on=False)
                # y_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                # self.xhat_data[q, :] = y_out  # collect
                # x_step = torch.tensor(y_out, dtype=torch.float32)  # ! update input to NN !
                # --------------------------------------------------------
                # self.y_pem0[q, :] = np.copy(self.factor.Xhat[0, 0])  ## different color
                self.y_pem0.append([self.factor.Xhat[1, 0], q])
                self.y_pem.append([None, q])
                # self.y_pem0[q] = np.copy(self.factor.Xhat[0, 0])

                self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])

                match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 1])
                if match < self.threshold1:

                    print(f'update at {q}, with R2= {match}')
                    self.correction.append(q)
                    while q < self.N:
                        u_step = u[q]
                        dx = self.model(x_step, u_step)
                        x_step = x_step + dx * self.dt  #+ torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)

                        y_nn = x_step[:, 1].clone().detach().numpy()
                        # u_in = self.xhat_data[q - 1, [0]]  # ==y_out
                        u_in = y_nn
                        self.factor.pem_one(y[q] - y_nn, u_in, on=True)

                        # self.y_pem[q, :] = np.copy(self.factor.Xhat[0, 0])  # adding
                        self.y_pem.append([self.factor.Xhat[1, 0], q])
                        self.y_pem0.append([None, q])
                        # self.y_pem[q] = np.copy(self.factor.Xhat[0, 0])
                        self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])
                        y_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                        # print('shape,', y_out.shape)
                        self.xhat_data[q, :] = y_out
                        x_step = torch.tensor(y_out, dtype=torch.float32)  # ! update input to NN !

                        match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 1])
                        if match > self.threshold2:
                            self.stop.append(q)
                            print(f'finish at  {q}, with R2= {match}')
                            self.factor.pem_one(y[q] - y_nn, u_in, on=False)
                            break
                        q = q + 1
                q = q + 1

            if self.update == 12:  # case 1 but with u size 2, input x_nn directly, works!
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt  #+ torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)
                # self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect output of NN
                y_nn = x_step[:, 0].clone().detach().numpy()
                u_in = x_step.clone().detach().numpy().T
                self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                self.xhat_data[q, :] = x_out
                x_step = torch.tensor(x_out, dtype=torch.float32)  # ! update input to NN !
                match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                # match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                q = q + 1

            if self.update == 1200:  # case 121, stop pem update at given time, self.train, u_in = 64, best
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
                    # u_in = x_step.clone().detach().numpy().T
                    u_in = self.model.out_k.clone().detach().numpy().T
                    # print(f'{q} Bhat,', self.factor.Bhat)
                    self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                    x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                    # self.pem_out.append(self.factor.Xhat[:, 0])
                    self.xhat_data[q, :] = x_out
                    x_step = torch.tensor(x_out, dtype=torch.float32)
                    q = q + 1

                u_in = self.model.out_k.clone().detach().numpy().T


                self.factor.pem_one(0, u_in, on=False)  #(y[q-1] - y_nn)*

            if self.update == 1201:  # case 1200, stop pem update at given time, self.train, u_in = 64, regulator G only x not u
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
                    # u_in = x_step.clone().detach().numpy().T
                    u_in = np.concatenate((self.model.out_k.clone().detach().numpy().T, u_step), axis=0)
                    # u_in =self.model.out_k.clone().detach().numpy().T
                    # print(f'{q} Bhat,', self.factor.Bhat)
                    self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                    x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                    # self.pem_out.append(self.factor.Xhat[:, 0])
                    self.xhat_data[q, :] = x_out
                    x_step = torch.tensor(x_out, dtype=torch.float32)
                    q = q + 1

                u_in = np.concatenate((self.model.out_k.clone().detach().numpy().T, u_step), axis=0)
                # u_in = self.model.out_k.clone().detach().numpy().T

                # print(f'{q} Bhat,', self.factor.Bhat)
                self.factor.pem_one(0, u_in, on=False)  #(y[q-1] - y_nn)*

            if self.update == 121:  # case 12 but stop pem update at given time, self.train
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt + torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect output of NN

                while q < self.train:
                    u_step = u[q]
                    dx = self.model(x_step, u_step)
                    x_step = x_step + dx * self.dt
                    y_nn = x_step[:, 0].clone().detach().numpy()
                    u_in = x_step.clone().detach().numpy().T
                    # print(f'{q} Bhat,', self.factor.Bhat)
                    self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                    x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                    self.pem_out.append(self.factor.Xhat[:, 0])
                    self.xhat_data[q, :] = x_out
                    x_step = torch.tensor(x_out, dtype=torch.float32)

                    # if q>= self.train:
                    #     break
                    q = q + 1

                u_in = x_step.clone().detach().numpy().T
                y_nn = x_step[:, 0].clone().detach().numpy()
                # print(f'{q} Bhat,', self.factor.Bhat)
                self.factor.pem_one(0, u_in, on=False)  #(y[q-1] - y_nn)*
                self.pem_out.append(self.factor.Xhat[:, 0])
                q = q + 1
            # works, in use, do not change <<<<< ---
            # # --->>>>  work, but not solid---

            if self.update == 120:  # case 121, stop pem update at given time, self.train, u_in = 64, experiment of pem matrix
                u_step = u[q]
                dx = self.model(x_step, u_step)
                self.ignition = int(200)
                self.age = int(4000)

                x_step = x_step + dx * self.dt + torch.tensor(self.factor.Xhat[:, 0],
                                                              dtype=torch.float32) * ignition_true
                self.err[q] = y[q] - x_step[0, 0].clone().detach().numpy()
                # x_step = x_step
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect output of NN
                self.pem_out[q, :] = self.factor.Xhat[:, 0]
                self.A_data[q, :] = self.factor.Ahat[1, :]
                self.B_data[q, :] = self.factor.Bhat[0:2, 0]
                q = q + 1

                while q >= self.ignition and q < self.age:  # 200<=q<1500

                    ignition_true = 1  # pem working, no aging
                    u_step = u[q]
                    dx = self.model(x_step, u_step)
                    x_step = x_step + dx * self.dt
                    y_nn = x_step[:, 0].clone().detach().numpy()
                    # u_in = x_step.clone().detach().numpy().T
                    u_in = self.model.out_k.clone().detach().numpy().T
                    # print(f'{q} Bhat,', self.factor.Bhat)
                    self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                    x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                    # self.pem_out.append(self.factor.Xhat[:, 0])
                    self.xhat_data[q, :] = x_out
                    self.pem_out[q, :] = self.factor.Xhat[:, 0]
                    self.A_data[q, :] = self.factor.Ahat[1, :]
                    self.B_data[q, :] = self.factor.Bhat[0:2, 0]
                    self.err[q] = y[q] - x_step[0, 0].clone().detach().numpy()
                    x_step = torch.tensor(x_out, dtype=torch.float32)

                    q = q + 1

                while q < self.train and q > self.age:  # 1500--3000
                    self.model.q_bar_age_true = 1
                    u_step = u[q]
                    dx = self.model(x_step, u_step)
                    x_step = x_step + dx * self.dt
                    y_nn = x_step[:, 0].clone().detach().numpy()
                    # u_in = x_step.clone().detach().numpy().T
                    u_in = self.model.out_k.clone().detach().numpy().T
                    # print(f'{q} Bhat,', self.factor.Bhat)
                    self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                    x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                    # self.pem_out.append(self.factor.Xhat[:, 0])
                    self.xhat_data[q, :] = x_out
                    self.pem_out[q, :] = self.factor.Xhat[:, 0]
                    self.A_data[q, :] = self.factor.Ahat[1, :]
                    self.B_data[q, :] = self.factor.Bhat[0:2, 0]
                    self.err[q] = y[q] - x_step[0, 0].clone().detach().numpy()
                    x_step = torch.tensor(x_out, dtype=torch.float32)
                    q = q + 1

                u_in = self.model.out_k.clone().detach().numpy().T
                self.factor.pem_one(0, u_in, on=False)  # (y[q-1] - y_nn)*

                # q = q + 1

            # PEM running from beginning, use steps as switch
            if self.update == 2:
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt
                y_nn = x_step[:, 0].clone().detach().numpy()
                u_in = y_nn
                # if q <= self.train or q % self.step == 0:
                if q <= self.train or all(np.remainder(q, self.step)) == 0:
                    self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                    self.on.append(q)
                if q > self.train:
                    self.factor.pem_one(y[q] - y_nn, u_in, on=False)
                x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]  # must have [:, 0], from 2x1 to 1x2
                self.xhat_data[q, :] = x_out
                x_step = torch.tensor(x_out, dtype=torch.float32)  # ! update input to NN !
                match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                # match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                q = q + 1


        return self.xhat_data


