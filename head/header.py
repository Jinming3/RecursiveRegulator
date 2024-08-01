
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.jit import Final
from typing import List, Tuple, Any
from pem import PEM , PEM_mimo, mse


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


def get_batch(batch_num, batch_length, Y, U): 
    N = len(Y)
    batch_start = np.random.choice(np.arange(N - batch_length, dtype=np.int64), batch_num, replace=False)
    batch_index = batch_start[:, np.newaxis] + np.arange(batch_length)  # batch sample index
    batch_index = batch_index.T  # (batch_length, batch_num, n_x)
    batch_x0 = x_fit[batch_start, :]  # (batch_num, n_x), initials in each batch
    batch_x = x_fit[[batch_index]]
    batch_u = torch.tensor(U[batch_index, :])
    batch_y = torch.tensor(Y[batch_index])
    return batch_y#, batch_u #,batch_x0, batch_x,


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


class NeuralStateSpaceModel(nn.Module):  # when not pos and vel, no derivative relation
    n_x: Final[int]
    n_u: Final[int]
    n_feat: Final[int]

    def __init__(self, n_x=2, n_u=1, n_feat=64, scale_dx=1.0, init_small=True, activation='relu'):
        super(NeuralStateSpaceModel, self).__init__()
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

        self.net = nn.Sequential(
            nn.Linear(n_x + n_u, n_feat),  # 2 states, 1 input
            activation,
            nn.Linear(n_feat, n_x)
        )

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, in_x, in_u):
        in_xu = torch.cat((in_x, in_u), -1)  # concatenate x and u over the last dimension to create the [xu] input
        dx = self.net(in_xu)  # \dot x = f([xu])
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




class ForwardEulerPEM(nn.Module):  # use steps or R2 as switch

    def __init__(self, model,
                 factor,
                 dt, N, update, threshold1=0, threshold2=0,
                 sensitivity=600, train=2000):  #sensitivity=100

        super(ForwardEulerPEM, self).__init__()
        self.factor = factor
        self.model = model
        self.dt = dt
        self.N = N

        self.update = update  # choose case

        self.train = train
        self.threshold1 = threshold1  # start update
        self.threshold2 = threshold2  # stop update
        self.sensitivity = sensitivity  # an sequence to monitor R2
        self.stop = []
        self.correction = []
        self.xhat_data = np.zeros((N, 2))


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
        x_mul = torch.ones(1, 2)
        # ------------------
        q = 0
        while q < self.N:
            # not updating, no PEM
            if self.update == 0:
                # simple forward Euler
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()
                q = q + 1
            # update non-stop:
            if self.update == 1:  
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt 
                y_nn = x_step[:, 0].clone().detach().numpy()
                u_in = y_nn
                self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                self.xhat_data[q, :] = x_out
                x_step = torch.tensor(x_out, dtype=torch.float32)  # ! update input to NN !
                match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                q = q + 1
           
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
                x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]  
                self.xhat_data[q, :] = x_out
                x_step = torch.tensor(x_out, dtype=torch.float32)  # ! update input to NN !
                match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                # match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                q = q + 1
            

            if self.update == 8: 
                u_step = u[q]
                dx = self.model(x_step, u_step)                
                x_step = x_step + dx * self.dt + torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)
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
                        x_step = torch.tensor(x_out, dtype=torch.float32) 
                        # match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                        match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])

                        # if q > self.sensitivity:
                        #     match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                        # if q <= self.sensitivity:
                        #     match = R2(y[0:q, 0], self.xhat_data[0:q, 0])
                        self.r2[q] = match
                      
                        q = q + 1
                y_nn = x_step[:, 0].clone().detach().numpy()
                self.factor.pem_one(0, y_nn, on=False)  

                q = q + 1

            # update with threshold
            if self.update == 5:
                u_step = u[q]
                dx = self.model(x_step, u_step)
                
                x_step = x_step + dx * self.dt + torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)# not updating pem added
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect
                self.err[q] = y[q] - x_step[0, 0].clone().detach().numpy()
               
                self.y_pem0.append([self.factor.Xhat[0, 0], q])
                self.y_pem.append([None, q])


                self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])
                match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0]) # check the dimension before use
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
                        self.factor.pem_one(y[q] - y_nn, y_nn, on=True)
                        self.y_pem.append([self.factor.Xhat[0, 0], q])
                        self.y_pem0.append([None, q])
                        self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])
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
                # print(q)
                self.factor.pem_one(y[q]*0 - y_nn, y_nn, on=False)  # for pem n-step ahead
                q = q + 1

            if self.update == 9: 
                u_step = u[q]
                dx = self.model(x_step, u_step) + torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)
                
                x_step = x_step + dx * self.dt
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect
             
                self.y_pem0.append([self.factor.Xhat[0, 0], q])
                self.y_pem.append([None, q])

                self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])
                if q > self.sensitivity:
                    match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                if q <= self.sensitivity:
                    match = R2(y[0:q, 0], self.xhat_data[0:q, 0])

                self.r2[q] = match
                if match < self.threshold1:
                    print(f'update at {q}, with R2= {match}')
                    self.correction.append(q)
                    while q < self.N:
                        u_step = u[q]
                        dx = self.model(x_step, u_step) + torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)
                      
                        x_step = x_step + dx * self.dt
                        y_nn = x_step[:, 0].clone().detach().numpy()

                        self.factor.pem_one(y[q] - y_nn, y_nn, on=True)

                        self.y_pem.append([self.factor.Xhat[0, 0], q])
                        self.y_pem0.append([None, q])

                        self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])
                        y_out = x_step.clone().detach().numpy()
                        self.xhat_data[q, :] = y_out
                        x_step = torch.tensor(y_out, dtype=torch.float32)  # ! update input to NN !
                        # match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                        match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])

                        # if q > self.sensitivity:
                        #     match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                        # if q <= self.sensitivity:
                        #     match = R2(y[0:q, 0], self.xhat_data[0:q, 0])
                        self.r2[q] = match
                        if match > self.threshold2:
                            
                            self.stop.append(q)
                            print(f'finish at  {q}, with R2= {match}')
                            break

                        q = q + 1
                q = q + 1
                
        return self.xhat_data







