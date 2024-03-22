"""
original and test functions

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


#  #------------- torch model >>>>>> -----------
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


class ForwardEuler(nn.Module):  # to train

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

# ----------------------<<< basic --------------------

# ----  >>>>>>>>>work>>>>>>>>----------------------


class ForwardEulerPEM(nn.Module):  # use steps or R2 as switch

    def __init__(self, model, factor, dt, N, update, threshold1=0, threshold2=0,
                 sensitivity=600, step=1, train=2000, param=np.array):  # sensitivity=100

        super(ForwardEulerPEM, self).__init__()
        self.factor = factor
        self.model = model
        self.dt = dt
        self.N = N

        self.update = update  # choose case
        self.step = step
        self.train = train
        self.threshold1 = threshold1  # start update
        self.threshold2 = threshold2  # stop update
        self.sensitivity = sensitivity  # an sequence to monitor R2
        self.stop = []
        self.correction = []
        self.xhat_data = np.zeros((N, 2))
        self.param = [param]

    def forward(self, x0: torch.Tensor, u: torch.Tensor, y):
        x_step = x0
        self.Thehat = np.zeros((self.N, 6))

        self.y_pem = []
        self.y_pem0 = []
        self.r2 = np.zeros(self.N)
        self.alter = torch.zeros(1, 2)
        self.err = np.zeros(self.N)  # |y-yhat|
        # ---------------
        self.on = []
        self.check = []
        # ------------------
        q = 0
        while q < self.N:

            if self.update == 0: # not updating, no PEM, just basic inference in simple forward Euler
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()
                q = q + 1

            if self.update == 1:  # update non-stop:
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt
                y_nn = x_step[:, 0].clone().detach().numpy()
                self.factor.pem_one(y[q] - y_nn, y_nn, on=True)
                x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                self.xhat_data[q, :] = x_out
                x_step = torch.tensor(x_out, dtype=torch.float32)  # ! update input to NN !
                # match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                q = q + 1

            if self.update == 8: # xtep = x_step+pem, copied from 5, use self.train to stop PEM, not R2, not added when stop
                u_step = u[q]
                dx = self.model(x_step, u_step)

                x_step = x_step + dx * self.dt + torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)# non-updating pem added

                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect

                self.y_pem0.append([self.factor.Xhat[0, 0], q])
                self.y_pem.append([None, q])

                if q > self.sensitivity:
                    match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                if q <= self.sensitivity:
                    match = R2(y[0:q, 0], self.xhat_data[0:q, 0])
                self.r2[q] = match
                while q < self.train:
                        u_step = u[q]
                        dx = self.model(x_step, u_step)
                        x_step = x_step + dx * self.dt
                        y_nn = x_step[:, 0].clone().detach().numpy()
                        self.factor.pem_one(y[q] - y_nn, y_nn, on=True)
                        self.y_pem.append([self.factor.Xhat[0, 0], q])
                        self.y_pem0.append([None, q])
                        self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])
                        x_out = x_step.clone().detach().numpy()+ self.factor.Xhat[:, 0]
                        self.xhat_data[q, :] = x_out
                        x_step = torch.tensor(x_out, dtype=torch.float32)  # don't delete this ! update input to NN !
                        match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                        self.r2[q] = match
                        q = q + 1
                y_nn = x_step[:, 0].clone().detach().numpy()
                self.factor.pem_one(y[q] - y_nn, y_nn, on=False)  # for pem n-step ahead
                q = q + 1

            if self.update == 5:  # update with threshold,  adding resting PEM !! # use this!!
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt + torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)# non-updating pem added
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


                self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])
                match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0]) # check the dimension before use
                # match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                # if q > self.sensitivity:
                #     match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                # if q <= self.sensitivity:
                #     match = R2(y[0:q, 0], self.xhat_data[0:q, 0])
                self.r2[q] = match
                if match < self.threshold1:
                    print(f'update at {q}, with R2= {match}')
                    self.correction.append([match, q])
                    while q < self.N:
                        u_step = u[q]
                        dx = self.model(x_step, u_step)
                        x_step = x_step + dx * self.dt
                        y_nn = x_step[:, 0].clone().detach().numpy()
                        self.factor.pem_one(y[q] - y_nn, y_nn, on=True)
                        self.y_pem.append([self.factor.Xhat[0, 0], q])
                        self.y_pem0.append([None, q])
                        self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])
                        self.err[q] = y[q] - x_step[0, 0].clone().detach().numpy()
                        x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                        self.xhat_data[q, :] = x_out
                        x_step = torch.tensor(x_out, dtype=torch.float32)  # don't delete this ! update input to NN !
                        match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                        # if q > self.sensitivity:
                        #     match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                        # if q <= self.sensitivity:
                        #     match = R2(y[0:q, 0], self.xhat_data[0:q, 0])
                        self.r2[q] = match
                        if match > self.threshold2:
                            self.stop.append([match, q])
                            print(f'finish at  {q}, with R2= {match}')
                            break
                        q = q + 1
                y_nn = x_step[:, 0].clone().detach().numpy()
                self.factor.pem_one(y[q] - y_nn, y_nn, on=False)  # for pem n-step ahead
                q = q + 1
        return self.xhat_data


class ForwardEulerPEM_ahead(nn.Module):  # same as ForwardEulerPEM, but with n step ahead prediction

    def __init__(self, model, factor, dt, N, update, ahead_step):

        super(ForwardEulerPEM_ahead, self).__init__()
        self.factor = factor
        self.model = model
        self.dt = dt
        self.N = N
        self.update = update  # choose case
        self.step = ahead_step
        self.xhat_data = np.zeros((N, 2))
        self.E_data = []
        self.predict_data = []

    def forward(self, x0: torch.Tensor, u: torch.Tensor, y, pre_ahead=True):
        x_step = x0
        q = 0
        while q < self.N:

            if self.update == 8: #move pem to (dx+pem)*dt
                u_step = u[q]
                dx = self.model(x_step, u_step) + torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)
                x_step = x_step + dx * self.dt
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect
                y_nn = x_step[:, 0].clone().detach().numpy()
                u_in = y_nn
                self.factor.pem_one(y[q] - y_nn, u_in, on=True)

                # --------------------------------------------------------
                if pre_ahead:
                    predict = []
                    if q < self.N - self.step:  # 1000 <
                        xhat_step = x_step
                        for z in range(self.step):
                            u_step = u[q + z + 1]
                            dx = self.model(xhat_step, u_step) + torch.tensor(self.factor.Xhat[:, 0],
                                                                              dtype=torch.float32)
                            xhat_step_new = xhat_step + dx * self.dt
                            y_nn = xhat_step_new[:, 0].clone().detach().numpy()
                            u_in = y_nn
                            self.factor.pem_one(y[q] - y_nn, u_in, on=False)  #
                            predict.append(xhat_step_new[:, 0].clone().detach().numpy())
                            xhat_step = xhat_step_new

                            # self.E = mse(y[q:q + self.step], predict)
                        self.E = mse(y[q + self.step], predict[-1])
                        # print('q=', q)
                        # print('E=', self.E)
                        self.E_data.append(self.E)
                        self.predict_data.append(predict[self.step-1])

                y_out = x_step.clone().detach().numpy()

                self.xhat_data[q, :] = y_out
                x_step = torch.tensor(y_out, dtype=torch.float32)  # ! update input to NN !

            q = q + 1

        return self.xhat_data




