"""
contains test functions

"""

import copy

import numpy as np

import torch
import torch.nn as nn
from torch.jit import Final
from typing import List, Tuple, Any


def R2(Y_sys, Yhat):
    s1 = torch.sum((Y_sys - Yhat) ** 2)
    mean = torch.mean(Y_sys)
    s2 = torch.sum((Y_sys - mean) ** 2)
    return 1.0 - s1 / s2


def get_batch(batch_num, batch_length, u, y, total, dt):
    batch_start = np.random.choice(np.arange(total - batch_length, dtype=np.int64), batch_num, replace=False)
    batch_index = batch_start[:, np.newaxis] + np.arange(batch_length)  # batch sample index
    batch_index = batch_index.T  # (batch_length, batch_num, n_x)

    v_est = np.concatenate((np.array([0]), np.diff(y[:, 0, 0])))
    # v_est = v_est.reshape(-1, 1) / dt
    X = np.zeros((y.shape[0], 2), dtype=np.float32)
    X[:, 0] = np.copy(y[:, 0, 0])
    X[:, 1] = np.copy(v_est)
    x_fit = torch.tensor(X, dtype=torch.float32, requires_grad=True)

    batch_x0 = x_fit[batch_start, :]  # (batch_num, n_x), initials in each batch
    batch_x = x_fit[[batch_index]]
    batch_u = torch.tensor(u[batch_index, 0])
    batch_y = torch.tensor(y[batch_index, 0])
    return batch_x0, batch_x, batch_u, batch_y  #

# ------------- torch original -----------
class MechanicalSystem(nn.Module):  # original

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


class ForwardEuler(nn.Module):  # original

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


# ----- evolution ------


class Retrain(nn.Module):  # original structure but new weight

    def __init__(self, dt, n_x=2):  #  load: nn.Module,
        super(Retrain, self).__init__()
        self.dt = dt  # sampling time
        self.hidden = 64
        self.net = nn.Sequential(nn.Linear(n_x + 1, self.hidden),  # 3*1
                                 # nn.LeakyReLU(negative_slope=0.4),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden, 1))
        # self.linear0 = nn.Linear(n_x + 1, self.hidden)
        # self.linear2 = nn.Linear(self.hidden, 1)


    def forward(self, x1, u1):
        list_dx: List[torch.Tensor]
        in_xu = torch.cat((x1, u1), -1)
        dv = self.net(in_xu) / self.dt  # v, dv = net(x, v)
        list_dx = [x1[..., [1]], dv]  # [dot x=v, dot v = a]
        dx = torch.cat(list_dx, -1)
        return dx


class ForwardEulerEvolution(nn.Module):

    def __init__(self, model, dt, N, optimizer, update=True, epoch=10000, threshold1=0, threshold2=0, sensitivity=500, slot=2000, lr=0.0001):
        super(ForwardEulerEvolution, self).__init__()
        self.model = model
        self.dt = dt
        self.N = N
        self.optimizer = optimizer
        self.update = update
        self.epoch = epoch  # retraining epoches if evolute
        self.threshold1 = threshold1  # start retrain
        self.threshold2 = threshold2  # stop retrain
        self.sensitivity = sensitivity  # monitor R2
        self.slot = slot  # data size used for retraining

        self.lr = lr


        self.stop = []
        self.evolute = []
        self.xhat_data = torch.zeros((N, 2))

    def forward(self, x0: torch.Tensor, u: torch.Tensor, y: torch.tensor):
        x_step = x0

        q = 0
        while q < self.N:
            u_step = u[q]

            dx = self.model(x_step, u_step)
            x_step = x_step + dx * self.dt
            self.xhat_data[q, :] = x_step[0, :].clone().detach()
            self.match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
            # self.match = R2(y[0:q, 0, 0], self.xhat_data[0:q, 0])
            print(f'R2 is {self.match} at data {q}')
            # if self.update:
            if self.update and self.match < self.threshold1:  # update

                self.evolute.append(q)
                print('evolute at ', q)

                # self.model.train()

                # model2.load_state_dict(self.model.state_dict())
                # model2.train()
                # model2.requires_grad = True
                # optimizer = torch.optim.Adam([{'params': params_net, 'lr': self.lr}], lr=self.lr * 10)

                # x_train = self.xhat_data[q - self.slot:q]
                # x_train = torch.tensor(x_train[:, None, :], requires_grad=True)

                retrain = ForwardEuler(model=self.model, dt=self.dt)
                batch_num = 64
                batch_length = 32

                for epoch in range(self.epoch):
                    x_train = x_step.clone().detach()  #
                    u_train = u[q - self.slot:q]  # copy.deepcopy()  # a small batch (slot, 1)????? or one element training???
                    y_train = y[q - self.slot:q]  # copy.deepcopy()  #

                    # batch_x0, batch_x, batch_u, batch_y = get_batch(batch_num=batch_num, batch_length=batch_length, u=u_train, y=y_train, total=self.slot, dt=self.dt)
                    # batch_xhat = retrain(batch_x0, batch_u)
                    # batch_yhat = batch_xhat[:, :, [0]]
                    # error_out = batch_yhat - batch_y


                    xhat = retrain(x_train, u_train)
                    yhat = xhat[:, :, [0]]
                    error_out = yhat - y_train

                    # ---- same as ForwardEuler-------
                    # xhat_list = []
                    # for u_train_step in u_train.split(1):
                    #     u_train_step = u_train_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
                    #     # dx = self.model(x_train, u_train_step)
                    #     dx = self.model(x_train, u_train_step)
                    #     x_train = x_train + dx * self.dt
                    #     xhat_list += [x_train]
                    # xhat = torch.stack(xhat_list, 0)
                    # ---------------------

                    loss = torch.mean((error_out / 0.1) ** 2)  # error_scale = 0.1
                    # if (epoch + 1) % 100 == 0:
                    #     print(f'epoch {epoch + 1}/{self.epoch}: loss= {loss.item():.4f}, yhat= {yhat[:,0, 0]:.4f}')

                    loss.backward(retain_graph=True)  #
                    self.optimizer.step()
                    # print('retrain epoch is', epoch)
                    # print(R2(y[q - self.slot:q, 0, 0], self.xhat_data[q - self.slot:q, 0]))
                    self.optimizer.zero_grad()

                    if (epoch + 1) % 100 == 0:  # check for break
                        print('loss', loss)
                        # ??????????????????????
                        with torch.no_grad():
                            x_vali = retrain(x_train, u_train)
                            y_vali = x_vali[:, :, 0]

                            self.match = R2(y_train[:, 0, 0], y_vali[:, 0])  #[q - self.slot:q, 0, 0]

                            print(f'epoch {epoch + 1}/{self.epoch}: R2= {self.match}')
                            if self.match > self.threshold2:
                                self.stop.append(q)
                                print('evolute finish at ', q)
                                # self.model.eval()
                                break






                # with torch.no_grad():
                #     dx = self.model(x_step, u_step)
                #     x_step = x_step + dx * self.dt
                #     self.xhat_data[q, :] = x_step[0, :].clone().detach()  #
                #     self.match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                #     # self.match = R2(y[0:q, 0, 0], self.xhat_data[0:q, 0])
                #
                #     if self.match > self.threshold2:
                #
                #         self.stop.append(q)
                #         print('stop evolute at ', q)
                #         # self.model.eval()
                #         break

                # self.model.load_state_dict(copy.deepcopy(model2.state_dict()))
                #
            q = q + 1

            # u_step = u[q]
            #     # y_step = y[q]
            # dx = self.model(x_step, u_step)
            # x_step = x_step + dx * self.dt
            # self.xhat_data[q, :] = x_step[0, :].clone().detach()  #
            # self.match = R2(y[q - self.slot:q], self.xhat_data[q - self.slot:q])

                # q = q + 1



        return self.xhat_data


"""

# --------------- linear + nonlinear step update test --------------------------
class Innovation(nn.Module):  # innovation model with K

    def __init__(self, a1, b, dt, n_x=2, grad=True):
        super(Innovation, self).__init__()
        self.grad = grad
        # self.a1 = torch.randn(1, n_x, dtype=torch.float32) * 0.1
        # self.A1.weight = torch.nn.Parameter(self.a1, requires_grad=True)
        # self.b = torch.randn(n_x, 1, dtype=torch.float32) * 0.1
        # self.B.weight = torch.nn.Parameter(self.b, requires_grad=True)

        self.a0 = torch.tensor([[0], [1]], dtype=torch.float32)
        self.A1 = nn.Linear(n_x, 1, bias=False)
        self.A1.weight = torch.nn.Parameter(torch.tensor(a1.astype(np.float32)), requires_grad=self.grad)
        self.B = nn.Linear(1, n_x, bias=False)
        self.B.weight = torch.nn.Parameter(torch.tensor(b.astype(np.float32)), requires_grad=self.grad)

        # as nonlinear part
        self.hidden = 64
        self.dt = dt
        self.net = nn.Sequential(nn.Linear(n_x + 1, self.hidden),  # 3*1
                                 # nn.LeakyReLU(negative_slope=0.4),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden, 1))

        for i in self.net.modules():
            if isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, mean=0, std=1e-3)
                nn.init.constant_(i.bias, val=0)

        # self.k = torch.randn(n_x+1, dtype=torch.float32) * 0.1
        # self.K = nn.Parameter(self.k, requires_grad=True)

    def forward(self, x, u):  # not? specific for acc  # , a1, b
        m1 = torch.cat((torch.matmul(x, self.a0), self.A1(x)), -1)
        m2 = self.B(u)
        xhat1 = m1 + m2
        in_xu = torch.cat((x, u), -1)
        dv = self.net(in_xu) / self.dt
        list_dx = [x[..., [1]], dv]  # [dot x=v, dot v = a]
        xhat0 = torch.cat(list_dx, -1)

        return xhat1, xhat0


# class Nonlinear(nn.Module):  # linear adopted from PEM, nonlinear are nn
#
#     def __init__(self, a1, b, n_x=2):
#         super(Nonlinear, self).__init__()
#         # linear adopts from PEM
#         self.a0 = torch.tensor([[0], [1]], dtype=torch.float32)
#         self.A1 = nn.Linear(n_x, 1, bias=False)
#         self.A1.weight = torch.nn.Parameter(torch.tensor(a1.astype(np.float32)), requires_grad=False)
#         self.B = nn.Linear(1, n_x, bias=False)
#         self.B.weight = torch.nn.Parameter(torch.tensor(b.astype(np.float32)), requires_grad=False)
#
#         # nonlinear part
#         self.hidden = 64
#         self.net = nn.Sequential(nn.Linear(n_x + 1, self.hidden),  # 3*1
#                                  # nn.LeakyReLU(negative_slope=0.4),
#                                  nn.ReLU(),
#                                  nn.Linear(self.hidden, 1))
#
#         for i in self.net.modules():
#             if isinstance(i, nn.Linear):
#                 nn.init.normal_(i.weight, mean=0, std=1e-3)
#                 nn.init.constant_(i.bias, val=0)
#
#         # self.k = torch.randn(n_x+1, dtype=torch.float32) * 0.1
#         # self.K = nn.Parameter(self.k, requires_grad=True)
#
#     def forward(self, x, u):  # discrete
#         xhat1 = torch.cat((torch.matmul(x, self.a0), self.A1(x)), -1) + self.B(u)
#         in_xu = torch.cat((x, u), -1)
#         xhat0 = self.net(in_xu)
#
#         return xhat1, xhat0


# class linear(nn.Module):
#
#     def __init__(self, n_x=2):
#         super(linear, self).__init__()
#
#         A = np.random.rand(1, n_x) * 0.1
#         B = np.random.rand(n_x, 1) * 0.1
#         print('A = ', A)
#         print('B = ', B)
#         self.A = nn.Linear(n_x, 1, bias=False)
#         self.B = nn.Linear(1, n_x, bias=False)
#         self.A.weight = torch.nn.Parameter(torch.tensor(A.astype(np.float32)), requires_grad=True)
#         self.B.weight = torch.nn.Parameter(torch.tensor(B.astype(np.float32)), requires_grad=True)
#
#         self.c = torch.randn(n_x, dtype=torch.float32) * 0.1
#         self.bias = nn.Parameter(self.c, requires_grad=True)
#         print('bias = ', self.c)
#
#     def forward(self, x, u):
#         dx = torch.cat((x[:, [1]], self.A(x)), dim=-1) + self.B(u)+ self.bias[None, :]
#
#         return dx

# -------- simulator ------------------

class ForwardEulerCombi(nn.Module):

    def __init__(self, model, dt):
        super(ForwardEulerCombi, self).__init__()
        self.model = model
        self.dt = dt

    def forward(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        xhat_list = list()
        self.x_nl_list = []  # xhat nonlinear  : List[torch.Tensor]
        self.x_l_list = []
        x_step = x0
        for u_step in u.split(1):
            u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
            xhat1, xhat0 = self.model(x_step, u_step)
            x_step = x_step + (xhat1 + xhat0) * self.dt
            xhat_list += [x_step]
            self.x_nl_list += [xhat0]
            self.x_l_list += [xhat1]

        xhat = torch.stack(xhat_list, 0)
        self.x_nl_list = torch.stack(self.x_nl_list, 0)
        self.x_l_list = torch.stack(self.x_l_list, 0)

        return xhat


class ModelUpdate(nn.Module):  # innovation linear adopt from PEM

    def __init__(self, n_x=2, dt=1):
        super(ModelUpdate, self).__init__()

        # self.a1 = torch.randn(1, n_x, dtype=torch.float32) * 0.1
        # self.A1.weight = torch.nn.Parameter(self.a1, requires_grad=True)
        # self.b = torch.randn(n_x, 1, dtype=torch.float32) * 0.1
        # self.B.weight = torch.nn.Parameter(self.b, requires_grad=True)

        self.a0 = torch.tensor([[0], [1]], dtype=torch.float32)
        self.A1 = nn.Linear(n_x, 1, bias=False)
        self.B = nn.Linear(1, n_x, bias=False)

        # as nonlinear part
        self.hidden = 64
        self.dt = dt
        self.net = nn.Sequential(nn.Linear(n_x + 1, self.hidden),  # 3*1
                                 # nn.LeakyReLU(negative_slope=0.4),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden, 1))  # 2

        for i in self.net.modules():
            if isinstance(i, nn.Linear):
                nn.init.constant_(i.weight, val=0)
                nn.init.constant_(i.bias, val=0)

    def forward(self, x, u, a1, b):  # discrete
        self.A1.weight = torch.nn.Parameter(a1, requires_grad=False)
        self.B.weight = torch.nn.Parameter(b, requires_grad=False)

        # m1 = torch.cat((torch.matmul(x, self.a0), self.A1(x)), -1)
        # m2 = self.B(u)
        # xhat1 = m1 + m2
        xhat1 = torch.cat((torch.matmul(x, self.a0), self.A1(x)), -1) + self.B(u)  # linear
        in_xu = torch.cat((x, u), -1)
        dv = self.net(in_xu) / self.dt  # v, dv = net(x, v)
        list_dx = [x[..., [1]], dv]  # [dot x=v, dot v = a]
        xhat0 = torch.cat(list_dx, -1)
        # xhat0 = self.net(in_xu)  # nonlinear

        return xhat1, xhat0


class ForwardEulerUpdate(nn.Module):

    def __init__(self, model, dt, update=True):
        super(ForwardEulerUpdate, self).__init__()
        self.model = model
        self.dt = dt
        self.update = update

    #

    def forward(self, x0: torch.Tensor, u: torch.Tensor, a1, b):  # -> tuple[Any, Any, Any]:  # single element
        a1 = torch.tensor(a1.astype(np.float32))
        b = torch.tensor(b.astype(np.float32))
        if not self.update:
            self.x_nl_list = []  #: List[torch.Tensor]  #  # xhat nonlinear  : List[torch.Tensor]
            self.x_l_list = []  #
            self.xhat_list: List[torch.Tensor] = []
            x_step = x0  # initial
            for u_step in u.split(1):
                u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
                xhat1, xhat0 = self.model(x_step, u_step, a1, b)
                x_step = x_step + (xhat1 + xhat0) * self.dt

                self.x_nl_list += [xhat0]
                self.x_l_list += [xhat1]
                self.xhat_list += [x_step]

            self.xhat_list = torch.stack(self.xhat_list, 0)
            self.x_nl_list = torch.stack(self.x_nl_list, 0)
            self.x_l_list = torch.stack(self.x_l_list, 0)

            return self.xhat_list

        if self.update:
            u_step = u
            x_step = x0
            xhat1, xhat0 = self.model(x_step, u_step, a1, b)  #

            x_step = x_step + (xhat1 + xhat0) * self.dt

            # self.x_nl_list += [xhat1]
            # self.x_l_list += [xhat0]

            # self.x_nl_list = torch.stack(self.x_nl_list, 0)
            # self.x_l_list = torch.stack(self.x_l_list, 0)

            return x_step, xhat0, xhat1  #


# ---- other maybe wrong test from head -----------
# --------- model -------------
# -------- do not  ----
class StateSpaceNet(nn.Module):  # 3*2, u diminished, wrong?

    def __init__(self, n_x=2, hidden_size=100, dt=1.0, init_small=True):
        super(StateSpaceNet, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt  # sampling time

        self.net = nn.Sequential(nn.Linear(n_x + 1, self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size, n_x))

        if init_small:
            for i in self.net.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-4)
                    nn.init.constant_(i.bias, val=0)

    def forward(self, x1, u1):
        in_xu = torch.cat((x1, u1), -1)
        dx = self.net(in_xu)
        dx = dx * self.dt
        return dx


# ----------not close-------------
class StateSpaceLinear(nn.Module):  # nonlinear NN, linear A B
    def __init__(self, A, B, n_x=2):
        super(StateSpaceLinear, self).__init__()
        self.A = nn.Linear(n_x, n_x, bias=False)
        self.B = nn.Linear(1, n_x, bias=False)
        self.A.weight = torch.nn.Parameter(torch.tensor(A.astype(np.float32)), requires_grad=True)
        self.B.weight = torch.nn.Parameter(torch.tensor(B.astype(np.float32)), requires_grad=True)

    def forward(self, x, u):
        dx = self.A(x) + self.B(u)
        return dx


# -----test --- model -----------
class LinearSpace(nn.Module):  # train well, cannot recognize different refs

    def __init__(self, width, init_small=True):
        super(LinearSpace, self).__init__()
        # self.hidden = 70
        # self.net = nn.Sequential(nn.Linear(width + 1, self.hidden),
        #                          # nn.ReLU(),
        #                          nn.Linear(self.hidden, width))
        self.net = nn.Linear(width + 1, width)
        if init_small:
            for i in self.net.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
                    nn.init.constant_(i.bias, val=0)

    def forward(self, x1, u1):

        in_xu = torch.cat((x1, u1), -1)
        out = self.net(in_xu)
        return out


class MechanicalSystem4(nn.Module):

    def __init__(self, dt, n_x=2, init_small=True):
        super(MechanicalSystem4, self).__init__()
        self.dt = dt  # sampling time
        self.hidden = 64
        self.net = nn.Sequential(
            nn.Linear(n_x + n_x + 1, self.hidden),
            # nn.Linear(n_x + 1, self.hidden),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=0.4),
            nn.Linear(self.hidden, 1))

        if init_small:
            for i in self.net.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
                    nn.init.constant_(i.bias, val=0)
                    # nn.init.normal_(i.bias, mean=0, std=1e-3)

    def forward(self, x1, u1, x1_l):
        list_dx: List[torch.Tensor]
        # in_xu = torch.cat((x1, u1), -1)  # when x_step
        in_xu = torch.cat((x1, u1, x1_l), -1)  # when x_nl
        dv = self.net(in_xu) / self.dt  # v, dv = net(x, v)
        list_dx = [x1[..., [1]], dv]  # [dot x=v, dot v = a]
        dx = torch.cat(list_dx, -1)
        return dx


class L3(nn.Module):
    def __init__(self, L, dt=1.0, n_x=2):  #
        super(L3, self).__init__()
        self.L3 = nn.Linear(n_x + 1, n_x, bias=False)
        # self.L3 = nn.Linear(n_x, n_x, bias=False)
        # self.L3 = nn.Linear(n_x + 1, 1, bias=False)  #
        self.L3.weight = torch.nn.Parameter(torch.tensor(L.astype(np.float32)), requires_grad=True)
        self.dt = dt

        # for i in self.L3.modules():
        #     if isinstance(i, nn.Linear):
        #         nn.init.normal_(i.weight, mean=0, std=0.01)
        #         # nn.init.constant_(i.bias, val=0)

    # def forward(self, x1, u1):
    #     list_dx: List[torch.Tensor]
    #     in_xu = torch.cat((x1, u1), -1)
    #     dv = self.L3(in_xu)# / self.dt  # v, dv = net(x, v)
    #     list_dx = [x1[..., [1]], dv]  # [dot x=v, dot v = a]
    #     dx = torch.cat(list_dx, -1)
    #     return dx

    def forward(self, x1, u1):
        in_xu = torch.cat((x1, u1), -1)
        dx = self.L3(in_xu)
        # dx = self.L3(x1)
        return dx


class Direction(nn.Module):

    def __init__(self, dt, n_x=2, init_small=True):
        super(Direction, self).__init__()
        self.dt = dt  # sampling time
        self.hidden = 40
        self.net = nn.Sequential(
            # nn.Linear(n_x + n_x + 1, self.hidden),
            nn.Linear(4, self.hidden),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=0.4),
            nn.Linear(self.hidden, n_x)  # 1
        )

        if init_small:
            for i in self.net.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
                    nn.init.constant_(i.bias, val=0)
                    # nn.init.normal_(i.bias, mean=0, std=1e-3)

    # def forward(self, x1, u1, x1_l):
    #     list_dx: List[torch.Tensor]
    #     # in_xu = torch.cat((x1, u1), -1)  # when x_step
    #     in_xu = torch.cat((x1, u1, x1_l), -1)  # when x_nl
    #     dv = self.net(in_xu) / self.dt  # v, dv = net(x, v)
    #     list_dx = [x1[..., [1]], dv]  # [dot x=v, dot v = a]
    #     dx = torch.cat(list_dx, -1)
    #     return dx
    def forward(self, xd, u1, xl):
        list_dx: List[torch.Tensor]
        in_xu = torch.cat((xd, u1, xl), -1)  # when x_step
        # in_xu = torch.cat((x3, u1, x1_l), -1)  # when x_nl
        # dv = self.net(in_xu) / self.dt  # v, dv = net(x, v)
        # list_dx = [x1[..., [1]], dv]  # [dot x=v, dot v = a]
        # dx = torch.cat(list_dx, -1)
        dx = self.net(in_xu)
        return dx


class Residue(nn.Module):

    def __init__(self, dt, n_x=2, init_small=True):
        super(Residue, self).__init__()
        self.dt = dt  # sampling time
        self.hidden = 64
        self.net = nn.Sequential(
            # nn.Linear(n_x + n_x + 1, self.hidden),
            nn.Linear(n_x + 1, self.hidden),
            nn.ReLU(),
            # nn.LeakyReLU(negative_slope=0.4),
            nn.Linear(self.hidden, 1))

        if init_small:
            for i in self.net.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
                    nn.init.constant_(i.bias, val=0)
                    # nn.init.normal_(i.bias, mean=0, std=1e-3)

    def forward(self, x1, u1):  # , x1_l
        list_dx: List[torch.Tensor]
        in_xu = torch.cat((x1, u1), -1)  # when x_step
        # in_xu = torch.cat((x1, u1, x1_l), -1)  # when x_nl
        dv = self.net(in_xu) / self.dt  # v, dv = net(x, v)
        list_dx = [x1[..., [1]], dv]  # [dot x=v, dot v = a]
        dx = torch.cat(list_dx, -1)
        # dx = self.net(in_xu)
        return dx


# -----test ------ simulator --------
class Solution(nn.Module):
    def __init__(self, model, linear, dt, update=False):
        super(Solution, self).__init__()
        self.model = model
        self.dt = dt
        self.linear = linear
        self.update = update

    def forward(self, x0, u, xd, y=np.array([...])):
        xhat_list: List[torch.Tensor] = []  # xhat all
        self.x_nl_list = []  # xhat nonlinear  : List[torch.Tensor]
        self.x_l_list = []  # xhat linear  : List[torch.Tensor]
        x_l = x0

        for u_step in u.split(1):
            u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
            x_nl = self.model(xd, u_step, x_l)  # (2, 2)

            if not self.update:
                x_l = self.linear(x0, u_step)

            self.x_nl_list += [x_nl]
            self.x_l_list += [x_l]
            x_step = x_nl + x_l
            xhat_list += [x_step]

        xhat = torch.stack(xhat_list, 0)
        self.x_nl_list = torch.stack(self.x_nl_list, 0)  #
        self.x_l_list = torch.stack(self.x_l_list, 0)  #
        return xhat


class ForwardEulerwithPem(nn.Module):

    def __init__(self, model, linear, dt, update=False):
        super(ForwardEulerwithPem, self).__init__()
        self.model = model
        self.dt = dt
        self.linear = linear
        self.update = update

    def forward(self, x0, u, y=np.array([...])):  # x0_l,
        xhat_list: List[torch.Tensor] = []  # xhat all
        self.x_nl_list = []  # xhat nonlinear  : List[torch.Tensor]
        self.x_l_list = []  # xhat linear  : List[torch.Tensor]

        # x_nl = x0
        # x_l = x0
        x_step = x0

        if not self.update:  # for pem in first training
            self.linear = self.linear
            self.x_l = np.zeros((y.shape[0], y.shape[1], 2))
            for i in range(y.shape[1]):
                out = self.linear(u[:, i, :].detach().numpy(), y[:, i, :].detach().numpy())
                self.x_l[:, i, :] = out
            self.x_l = torch.tensor(self.x_l, dtype=torch.float32)  # batch xhat
            j = 0
            for u_step in u.split(1):
                # x_nl = self.model(self.x_l[[j], :, :], u_step)
                u_step = u_step.squeeze(0)
                x_nl = self.model(x_step, u_step)
                # x_step = x_step + x_nl * self.dt + self.x_l[[j], :, :] * self.dt
                x_step = x_step + x_nl * self.dt + self.x_l[j, :, :] * self.dt

                self.x_nl_list += [x_nl]
                xhat_list += [x_step]
                j = j + 1

        if self.update:
            self.x_l = self.linear(u.detach().numpy(), y.detach().numpy())
            j = 0
            for u_step in u.split(1):
                u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
                xj = torch.tensor(self.x_l[j], dtype=torch.float32)
                xj = xj[None, :]
                # x_nl = self.model(x_step, u_step)
                x_nl = self.model(xj, u_step)
                x_step = x_step + (xj + x_nl) * self.dt
                # x0 = x_l[j, :, :]
                # # x_nl = self.model(x_l, u_step)
                # x_nl = self.model(x0, u_step)
                # x_step = x_step +x_nl * self.dt + x_l[j, :, :]* self.dt
                j = j + 1
                # self.x_l_list += [xj]
                self.x_nl_list += [x_nl]
                xhat_list += [x_step]

        xhat = torch.stack(xhat_list, 0)
        self.x_nl_list = torch.stack(self.x_nl_list, 0)  #
        # if self.update:
        #     self.x_l_list = torch.stack(self.x_l_list, 0)  #
        return xhat


class ForwardEulerwithLinear(nn.Module):

    def __init__(self, model, linear, dt, update=False):
        super(ForwardEulerwithLinear, self).__init__()
        self.model = model
        self.dt = dt
        self.linear = linear
        self.update = update

    def forward(self, x0, u, y=np.array([...])):  # x0_l,
        xhat_list: List[torch.Tensor] = []  # xhat all
        self.x_nl_list = []  # xhat nonlinear  : List[torch.Tensor]
        self.x_l_list = []  # xhat linear  : List[torch.Tensor]

        # x_nl = x0
        # x_l = x0_l
        # x_l = x0
        x_step = x0

        # if not self.update:  # for pem in first training
        #     self.x_l = np.zeros((y.shape[0], y.shape[1], 2))
        #     for i in range(y.shape[1]):
        #         out = self.linear(u[:, i, :].detach().numpy(), y[:, i, :].detach().numpy())
        #         self.x_l[:, i, :] = out
        #     self.x_l = torch.tensor(self.x_l, dtype=torch.float32)  # batch xhat
        #     j = 0
        #     for u_step in u.split(1):
        #         x_nl = self.model(self.x_l[[j], :, :], u_step)
        #         x_step = x_step + x_nl * self.dt + self.x_l[[j], :, :] * self.dt
        #         self.x_nl_list += [x_nl]
        #         xhat_list += [x_step]
        #         j = j+1

        if not self.update:  # first training is L3
            for u_step in u.split(1):
                u_step = u_step.squeeze(0)
                x_l = self.linear(x_step, u_step)
                x_nl = self.model(x_l, u_step)
                # x_l = x_l + x_l * self.dt  #
                x_step = x_step + (x_l + x_nl) * self.dt

                self.x_nl_list += [x_nl]
                self.x_l_list += [x_l]
                xhat_list += [x_step]

        if self.update:
            self.x_l = self.linear(u.detach().numpy(), y.detach().numpy())
            j = 0
            for u_step in u.split(1):
                u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
                xj = torch.tensor(self.x_l[j], dtype=torch.float32)
                xj = xj[None, :]
                # x_nl = self.model(x_step, u_step)
                x_nl = self.model(xj, u_step)
                x_step = x_step + (xj + x_nl) * self.dt
                # x0 = x_l[j, :, :]
                # # x_nl = self.model(x_l, u_step)
                # x_nl = self.model(x0, u_step)
                # x_step = x_step +x_nl * self.dt + x_l[j, :, :]* self.dt
                j = j + 1
                # self.x_l_list += [xj]
                self.x_nl_list += [x_nl]
                xhat_list += [x_step]

        xhat = torch.stack(xhat_list, 0)
        self.x_nl_list = torch.stack(self.x_nl_list, 0)  #
        # self.x_l_list = torch.stack(self.x_l_list, 0)  #
        return xhat


class ForwardEulerL3(nn.Module):

    def __init__(self, model, linear, dt, update=False):
        super(ForwardEulerL3, self).__init__()
        self.model = model
        self.dt = dt
        self.linear = linear
        self.update = update

    # def forward(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    def forward(self, x0, u, y=np.array([...])):
        xhat_list: List[torch.Tensor] = []  # xhat all
        self.x_nl_list = []  # xhat nonlinear  : List[torch.Tensor]
        self.x_l_list = []  # xhat linear  : List[torch.Tensor]
        x_nl = x0
        x_l = x0
        x_step = x0
        i = 0

        for u_step in u.split(1):
            u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
            dx0 = self.model(x_nl, u_step)

            x_nl = x_nl + self.dt * dx0  #

            if not self.update:
                x_l = self.linear(x_step, u_step)  # * self.dt

            # if self.update:
            #     y_step = y[i, None]
            #     y0 = dx0[0, [0]].detach().numpy()
            #     u0 = u_step.detach().numpy()
            #     # part_linear = []  ## : List[np.array]
            #     # part_start = self.linear(u0, y_step)
            #     part_linear = self.linear(u0, y_step)  #-y0
            #     # part_linear += [part_start]
            #     # part_linear = np.stack(part_linear, 0)
            #     # part_linear = part_linear.squeeze(2)
            #     dx1 = torch.tensor(part_linear, dtype=torch.float32)
            #     dx = dx0 + dx1

            # if self.update:
            #     y_step = y[i]
            #     part_linear= [] ## : List[np.array]
            #     for j in range(len(u_step)):
            #         u0 = u_step[j].detach().numpy()
            #         y0 = y_step[j].detach().numpy()
            #
            #         part_start = self.linear(u0, y0)
            #         # part_linear = part_linear.ravel()
            #         # part_linear = torch.tensor(part_linear[np.newaxis, :], dtype=torch.float32)
            #         part_linear += [part_start]
            #     part_linear = np.stack(part_linear, 0)
            #     part_linear = part_linear.squeeze(1)
            #     part_linear = torch.tensor(part_linear, dtype=torch.float32)
            #
            #     dx = dx0 + part_linear
            self.x_nl_list += [x_nl]
            self.x_l_list += [x_l]

            x_step = x_nl + x_l
            xhat_list += [x_step]
            i = i + 1

        xhat = torch.stack(xhat_list, 0)

        self.x_nl_list = torch.stack(self.x_nl_list, 0)  #
        self.x_l_list = torch.stack(self.x_l_list, 0)  #
        return xhat
    # def forward(self, x1, u1, y1: torch.Tensor):  # x0(1, 2), u(1 1), y(1, 1)
    #
    #     dx0 = self.model(x1, u1)
    #     if self.update:
    #         u0 = u1.clone().detach().numpy()
    #         y0 = y1.clone().detach().numpy()
    #         part_linear = self.linear(u0, y0)
    #
    #         part_linear = torch.tensor(part_linear, dtype=torch.float32, requires_grad=False)
    #
    #         dx = dx0 + part_linear
    #         x1 = x1 + self.dt * dx
    #
    #     return x1


class RK4Linear(nn.Module):  # ??
    def __init__(self, model, linear, dt, update=False):
        super(RK4Linear, self).__init__()
        self.model = model
        self.dt = dt
        self.linear = linear
        self.update = update
        # self.linear_list = []

    def forward(self, x0, u, y=np.array([...])):  #
        xhat_list: List[torch.Tensor] = []

        x_step = x0
        i = 0
        for u_step in u.split(1):
            u_step = u_step.squeeze(0)
            k1 = self.model(x_step, u_step)
            k2 = self.model(x_step + self.dt * k1 / 2.0, u_step)
            k3 = self.model(x_step + self.dt * k2 / 2.0, u_step)
            k4 = self.model(x_step + self.dt * k3, u_step)
            part_nonli = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            if not self.update:
                part_linear = self.linear(x_step, u_step)
                x_step = x_step + self.dt * part_nonli + self.dt * part_linear
            if self.update:
                # y_step = y[i]
                y_step = y[i, None]
                part_linear = self.linear(u_step.detach().numpy(), y_step)  # x_step,.detach().numpy()
                # self.linear_list += [part_linear]
                part_linear = part_linear.squeeze(2)

                part_linear = torch.tensor(part_linear, dtype=torch.float32)
                x_step = x_step + self.dt * part_nonli + self.dt * part_linear

            xhat_list += [x_step]
            i = i + 1
        xhat = torch.stack(xhat_list, 0)

        return xhat


class ForwardEulerLinear(nn.Module):

    def __init__(self, model, linear, dt, update=False):
        super(ForwardEulerLinear, self).__init__()
        self.model = model
        self.dt = dt
        self.linear = linear
        self.update = update

    # def forward(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    def forward(self, x0, u, y=np.array([...])):
        xhat_list: List[torch.Tensor] = []  # xhat all
        self.x_nl_list = []  # xhat nonlinear  : List[torch.Tensor]
        self.x_l_list = []  # xhat linear  : List[torch.Tensor]
        x_step = x0
        i = 0
        for u_step in u.split(1):
            u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
            dx0 = self.model(x_step, u_step)

            if not self.update:
                dx1 = self.linear(x_step, u_step)

            # if self.update:
            #     y_step = y[i, None]
            #     y0 = dx0[0, [0]].detach().numpy()
            #     u0 = u_step.detach().numpy()
            #     # part_linear = []  ## : List[np.array]
            #     # part_start = self.linear(u0, y_step)
            #     part_linear = self.linear(u0, y_step)  #-y0
            #     # part_linear += [part_start]
            #     # part_linear = np.stack(part_linear, 0)
            #     # part_linear = part_linear.squeeze(2)
            #     dx1 = torch.tensor(part_linear, dtype=torch.float32)
            #     dx = dx0 + dx1

            # if self.update:
            #     y_step = y[i]
            #     part_linear= [] ## : List[np.array]
            #     for j in range(len(u_step)):
            #         u0 = u_step[j].detach().numpy()
            #         y0 = y_step[j].detach().numpy()
            #
            #         part_start = self.linear(u0, y0)
            #         # part_linear = part_linear.ravel()
            #         # part_linear = torch.tensor(part_linear[np.newaxis, :], dtype=torch.float32)
            #         part_linear += [part_start]
            #     part_linear = np.stack(part_linear, 0)
            #     part_linear = part_linear.squeeze(1)
            #     part_linear = torch.tensor(part_linear, dtype=torch.float32)
            #
            #     dx = dx0 + part_linear
            dx = dx0 + dx1
            # x_step = dx0 + dx1
            self.x_nl_list += [dx0]
            self.x_l_list += [dx1]

            x_step = x_step + self.dt * dx
            xhat_list += [x_step]
            i = i + 1

        xhat = torch.stack(xhat_list, 0)

        self.x_nl_list = torch.stack(self.x_nl_list, 0)  #
        self.x_l_list = torch.stack(self.x_l_list, 0)  #
        return xhat
    # def forward(self, x1, u1, y1: torch.Tensor):  # x0(1, 2), u(1 1), y(1, 1)
    #
    #     dx0 = self.model(x1, u1)
    #     if self.update:
    #         u0 = u1.clone().detach().numpy()
    #         y0 = y1.clone().detach().numpy()
    #         part_linear = self.linear(u0, y0)
    #
    #         part_linear = torch.tensor(part_linear, dtype=torch.float32, requires_grad=False)
    #
    #         dx = dx0 + part_linear
    #         x1 = x1 + self.dt * dx
    #
    #     return x1

#  ------------- test --- end ----------------------
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, Model
#  ----------------- tf model not finished -------------------
# class Canonical(Model):  # discrete time
#
#     def __init__(self, init_small=True):
#         super(Canonical, self).__init__()
#
#         if init_small:
#             self.initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1e-3, seed=None)
#
#     def build(self, n_x=2):
#         self.a0 = np.array([[1, 0]])
#         self.a = self.add_weight(shape=(1, n_x),
#                                  initializer=self.initializer,
#                                  trainable=True
#                                  )
#         self.b = self.add_weight(shape=(n_x, 1),
#                                  initializer=self.initializer,
#                                  trainable=True)
#
#     def call(self, x, u):
#         xhat = tf.concat((tf.matmul(self.a0, x), tf.matmul(self.a, x)), axis=0) + self.b * u
#         return xhat
#
#
# # -------- simulator ------------------
# class ForwardEulerDiscrete(Model):
#
#     def __init__(self, model):
#         super(ForwardEulerDiscrete, self).__init__()
#         self.model = model
#
#     def call(self, x_u):  # x0, u
#         xhat_list = list()
#         x_step = x_u[0]
#         u = x_u[1]
#         for u_step in u.split(1):
#             # u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
#             x_step = self.model(x_step, u_step)
#             xhat_list += [x_step]
#
#         xhat = np.stack(xhat_list, 0)
#         return xhat

"""
