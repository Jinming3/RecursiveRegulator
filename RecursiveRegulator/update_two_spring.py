
import matplotlib

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('F:/Project/head/tight.mplstyle')
import os
import sys
import math
import time
sys.path.append(os.path.join("F:/Project/head/"))
import header
from pem import PEM  # _step as PEM
from pem import normalize, R2
from header import MechanicalSystem, ForwardEulerPEM, ForwardEuler
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


#   ---- motion----
def sinwave(dt, i, w, A): #=0.1,=1.0
    # out = []

    # for k in range(int(time / dt)):
    x = A * np.cos(w * i * math.pi * dt)
    # x = A * np.sin(w*k * math.pi* dt)
    # out.append(x)
    return x


# -- tri wave ---
def triangle(dt, i, A=2):  # , time_all
    out = []
    p = 8
    # for k in range(int(time_all / dt)):
    #     x = 2 * np.abs(k * dt / p - math.floor(k * dt / p + 0.5))  # 2 * -1
    x = A* np.abs(i * dt / p - math.floor(i * dt / p + 0.5))
    # out.append(x)
    return x


class Motion:
    def __init__(self, dt, pos1, pos2, vel1, vel2, acc1, acc2):
        self.pos1 = pos1
        self.pos2 = pos2
        self.vel1 = vel1
        self.vel2 = vel2
        self.acc1 = acc1
        self.acc2 = acc2
        self.dt = dt

    def measure(self,ref, noise_process=0.0, noise_measure=0.0):  # ref is a scalar value, not a function
        self.u = ref#ref(self.dt, i)
        self.acc1 = -(k1 + k2 / m1) * self.pos1 - (d1 + d2) / m1 * self.vel1 + k2 / m1 * self.pos2 + d2 / m1 * self.vel2 - Fc / m1 * np.sign(self.vel1)
        self.acc2 = k2 / m2 * self.pos1 + d2 / m2 * self.vel1 - k2 / m2 * self.pos2 - d2 / m2 * self.vel2 + 1 / m2 * self.u - Fc / m2 * np.sign(
            self.vel2) + np.random.randn() * noise_process
        self.vel1 = self.vel1 + self.acc1 * self.dt
        self.vel2 = self.vel2 + self.acc2 * self.dt
        self.pos1 = self.pos1 + self.vel1 * self.dt
        self.pos2 = self.pos2 + self.vel2 * self.dt
        output = self.pos2 + np.random.randn() * noise_measure
        return output


m1 = 20
m2 = 20
k1 = 1000
k2 = 2000
d1 = 1
d2 = 5
Fc = 0.05
Y_sys = []
U = []

m1_all = []
m2_all = []
k1_all = []
k2_all = []
d1_all = []
d2_all = []

time_all = 150
dt = 0.05
N = int(time_all / dt)
time_exp = np.arange(N) * dt
# changing = np.array([20, 50]) / dt
changing = np.array([20, 40, 55, 70, 100]) / dt
# changing = np.array([90]) / dt # time_all
changing = changing.astype(int)



# ##--- changing -------------------
# # # # ---- start original system with no change---
# sampling = Motion(dt, pos1=0, pos2=0, vel1=0, vel2=0, acc1=0, acc2=0)
#
# for i in range(N):
#     y = sampling.measure(ref=sinwave)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     # m1_all.append(m1)
#     # m2_all.append(m2)
#     # k1_all.append(k1)
#     # k2_all.append(k2)
#     # d1_all.append(d1)
#     # d2_all.append(d2)

# -----------------------------------------
# # #  --- system changing ---
# # # --- original ------
sampling = Motion(dt, pos1=0, pos2=0, vel1=0, vel2=0, acc1=0, acc2=0)
scale = 10e-3
for i in range(changing[0]): # original condition with noise
    y = sampling.measure(ref=sinwave(dt=dt, i=i, w=0.5, A=2), noise_process=scale*0.1, noise_measure=scale*0.01)
    Y_sys.append(y)
    U.append(sampling.u)
    m1_all.append(m1)
    m2_all.append(m2)
    k1_all.append(k1)
    k2_all.append(k2)
    d1_all.append(d1)
    d2_all.append(d2)

m1 = m1 * 0.98
m2 = m2 * 0.98
k1 = k1 * 0.96
k2 = k2 * 0.98
d1 = d1 * 0.96
d2 = d2 * 0.98
for i in range(changing[0], changing[1]):
    y = sampling.measure(ref=sinwave(dt=dt, i=i, w=0.5, A=1), noise_process=scale*0.1, noise_measure=scale*0.01)
    Y_sys.append(y)
    U.append(sampling.u)
    m1_all.append(m1)
    m2_all.append(m2)
    k1_all.append(k1)
    k2_all.append(k2)
    d1_all.append(d1)
    d2_all.append(d2)

for i in range(changing[1], changing[2]):
    y = sampling.measure(ref=1, noise_process=scale*0.1, noise_measure=scale*0.001)
    Y_sys.append(y)
    U.append(sampling.u)
    m1_all.append(m1)
    m2_all.append(m2)
    k1_all.append(k1)
    k2_all.append(k2)
    d1_all.append(d1)
    d2_all.append(d2)
m1 = m1 * 0.95
m2 = m2 * 0.98
k1 = k1 * 0.96
k2 = k2 * 0.96
d1 = d1 * 0.99
d2 = d2 * 0.98
for i in range(changing[2], changing[3]):
    y = sampling.measure(ref=-1, noise_process=scale*0.1, noise_measure=scale*0.001)
    Y_sys.append(y)
    U.append(sampling.u)
    m1_all.append(m1)
    m2_all.append(m2)
    k1_all.append(k1)
    k2_all.append(k2)
    d1_all.append(d1)
    d2_all.append(d2)

# m1 = m1 * 0.9
# m2 = m2 * 0.8
# k1 = k1 * 0.9
# k2 = k2 * 0.9
# d1 = d1 * 0.9
# d2 = d2 * 0.9


for i in range(changing[3], changing[4]):
    y = sampling.measure(ref=sinwave(dt, i, 1/2, 0.6)+sinwave(dt, i, 1/3, 0.4), noise_process=scale*0.1, noise_measure=scale*0.001)
    Y_sys.append(y)
    U.append(sampling.u)
    m1_all.append(m1)
    m2_all.append(m2)
    k1_all.append(k1)
    k2_all.append(k2)
    d1_all.append(d1)
    d2_all.append(d2)

m1 = m1 * 0.98
m2 = m2 * 0.97
k1 = k1 * 0.98
k2 = k2 * 0.96
d1 = d1 * 1.99
d2 = d2 * 1.98
for i in range(changing[4], N):
    y = sampling.measure(ref=sinwave(dt, i, 0.1, 0.6), noise_process=scale*0.1, noise_measure=scale*0.001)
    Y_sys.append(y)
    U.append(sampling.u)
    m1_all.append(m1)
    m2_all.append(m2)
    k1_all.append(k1)
    k2_all.append(k2)
    d1_all.append(d1)
    d2_all.append(d2)
# -----------------------------------
# # # --------------- only one sudden change -------------------

# m1 = m1 * 0.9
# m2 = m2 * 0.9
# k1 = k1 * 0.9
# k2 = k2 * 0.9
# d1 = d1 * 0.9
# d2 = d2 * 0.9
# for i in range(changing[0], N):
#     y = sampling.measure(ref=sinwave, noise_process=10 ** -3, noise_measure=10 ** -4)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     m1_all.append(m1)
#     m2_all.append(m2)
#     k1_all.append(k1)
#     k2_all.append(k2)
#     d1_all.append(d1)
#     d2_all.append(d2)

# # # # # --------------- sudden change in params and in ref, change ref is no good ?-------------------
# sampling = Motion(dt, pos1=0, pos2=0, vel1=0, vel2=0, acc1=0, acc2=0)
# for i in range(changing[0]):
#     y = sampling.measure(ref=sinwave)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     m1_all.append(m1)
#     m2_all.append(m2)
#     k1_all.append(k1)
#     k2_all.append(k2)
#     d1_all.append(d1)
#     d2_all.append(d2)
#
#
# m1 = m1 * 0.9
# m2 = m2 * 0.9
# k1 = k1 * 0.9
# k2 = k2 * 0.9
# d1 = d1 * 0.9
# d2 = d2 * 0.9
# for i in range(changing[0], changing[1]):
#     y = sampling.measure(ref=sinwave, noise_process=10 ** -5, noise_measure=10 ** -5)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     m1_all.append(m1)
#     m2_all.append(m2)
#     k1_all.append(k1)
#     k2_all.append(k2)
#     d1_all.append(d1)
#     d2_all.append(d2)
#
# for i in range(changing[1], N):
#     y = sampling.measure(ref=triangle, noise_process=0, noise_measure=0)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     m1_all.append(m1)
#     m2_all.append(m2)
#     k1_all.append(k1)
#     k2_all.append(k2)
#     d1_all.append(d1)
#     d2_all.append(d2)

# --------------------------------------
# # # --- continuously degenerating ----
# #
# sampling = Motion(dt, pos1=0, pos2=0, vel1=0, vel2=0, acc1=0, acc2=0)
# for i in range(changing[0]):
#     y = sampling.measure(ref=sinwave)
#     Y_sys.append(y)
#     U.append(sampling.u)
#
#
# for i in range(changing[0], N):
#     if i % 50 == 0:
#         m1 = m1 - 0.01 * i * dt
#         m2 = m2 - 0.01 * i * dt
#         d1 = d1 - 0.0001 * i * dt
#         d2 = d2 - 0.0001 * i * dt
#         k1 = k1 - 0.0001 * i * dt
#         k2 = k2 - 0.0001 * i * dt
#     y = sampling.measure(ref=sinwave, noise_process=0, noise_measure=0)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     m1_all.append(m1)
#     m2_all.append(m2)
#     k1_all.append(k1)
#     k2_all.append(k2)
#     d1_all.append(d1)
#     d2_all.append(d2)

# ----------------------------
# ------------------ prepare data type ---------
Y_sys = normalize(Y_sys, 1)
U = normalize(U, 1)
Y_sys = np.asarray(Y_sys, dtype=np.float32)
U = np.asarray(U, dtype=np.float32)
U = U[:, np.newaxis]
dt = torch.tensor(dt, dtype=torch.float32)

lr = 0.0001  # not used in PEM updateing
system = 'two_spring_motion5'
model_filename = f"{system}"
initial_filename = f"{system}_initial"
model = MechanicalSystem(dt=dt)  #
x_fit = torch.load(os.path.join("models", initial_filename))
checkpoint = torch.load(os.path.join("models", model_filename))
# model.eval()

optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': lr},
    {'params': [x_fit], 'lr': lr}
], lr=lr * 10)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # , strict=False




threshold1 = 1#0.97  # start retrain, R2
threshold2 = 1#0.98  # stop retrain
factor = PEM(2, 6, N)
factor.P_old2 *= 0.9
factor.Psi_old2 *= 0.9
np.random.seed(3)
factor.Thehat_old = np.random.rand(6, 1) * 0.1
# print('seed=', np.random.get_state()[1][0])#
factor.Xhat_old = np.array([[2], [0]])

# simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N, optimizer=optimizer, update=0, threshold1=threshold1, threshold2=threshold2)
simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N,  update=1,threshold1=threshold1, threshold2=threshold2) #optimizer=optimizer,
# simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N, optimizer=optimizer, update=5, threshold1=threshold1, threshold2=threshold2)


# x_fit = np.zeros((1, n_x), dtype=np.float32)
# x_fit[0, 0] = np.copy(Y_sys[0, 0])
# x_fit[0, 1] = 0
# x_step = x0
# x0 = torch.tensor(x_fit[[0], :], dtype=torch.float32)

x0 = x_fit[[0], :].detach()

u = torch.tensor(U[:, None, :])  # [:, None, :]
y = Y_sys[:, np.newaxis]


simulator0 = ForwardEuler(model=model, dt=dt)
start_time = time.time()
with torch.no_grad():
    xhat0 = simulator0(x0, u)
    xhat0 = xhat0.detach().numpy()
    xhat0 = xhat0.squeeze(1)
    yhat0 = xhat0[:, 0]
    # yhat0=yhat0[:, None]
print(f"\n NN  time: {time.time() - start_time:.2f}")



start_time = time.time()
xhat_data = simulator(x0, u, y)
print(f"\nTrain time: {time.time() - start_time:.2f}")
# ----- optimization inside NN loop, stepwise --------
yhat = xhat_data[:, 0]
Thehat = simulator.Thehat
stop = simulator.stop
correction = simulator.correction
print(f'update at {correction}')
print(f'stop at {stop}')

# # ->>>---- update == False, optimization outside NN loop, soo faster than stepwise --
# yhat_stable = xhat_data[:, 0]
# factor.forward(y, yhat_stable)
# yhat = factor.Yhat_data
# Thehat = factor.Thehat_data
# ------ <<<--------------------------------------

# np.savetxt('yhat0_two_spring.txt', yhat)

# yhat0 = np.loadtxt('yhat0_two_spring.txt')
print("nn R^2 = ", R2(Y_sys, yhat0))
print("inference evolution R^2 = ", R2(Y_sys, yhat))
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(time_exp, Y_sys, 'g', label='y')
ax[0].plot(time_exp, yhat0, 'r--', label='$\hat{y}_{NN}$')
ax[0].plot(time_exp[changing], Y_sys[changing], 'kx')
ax[0].set_ylabel("(a)")
ax[0].legend(loc=4)
ax[1].plot(time_exp, Y_sys, 'g', label='y')
ax[1].plot(time_exp, yhat, 'r--', label='$\hat{y}$')
# ax[0].plot(time_exp, simulator.y_pem, 'y--')
ax[1].plot(time_exp[changing], Y_sys[changing], 'kx')
ax[1].set_ylabel("(b)")
ax[1].legend(loc=4)
ax[2].plot(time_exp, U, 'k', label='u')
# ax[2].plot(time_exp[changing], U[changing], 'kx', label='changing')
ax[2].set_ylabel("(c)")
ax[2].set_xlabel('time(s)')
ax[2].legend(loc=4)

#
# # #
# fig, ax = plt.subplots(6, 1, sharex=True)
# ax[0].plot(time_exp, Thehat[:, 0], 'g', label='a0')
# ax[0].plot(time_exp[changing], Thehat[changing, 0], 'kx')
# # ax[0].plot(time_exp[correction], Thehat[correction, 0], 'yx')
# # ax[0].plot(time_exp[stop], Thehat[stop, 0], 'mx')
# ax[0].legend()
# ax[1].plot(time_exp, Thehat[:, 1], 'g', label='a1')
# # ax[1].plot(time_exp[changing], Thehat[changing, 1], 'kx')
# ax[1].legend()
# ax[2].plot(time_exp, Thehat[:, 2], 'b', label='b0')
# # ax[2].plot(time_exp[changing], Thehat[changing, 2], 'kx')
# ax[2].legend()
# ax[3].plot(time_exp, Thehat[:, 3], 'b', label='b1')
# # ax[3].plot(time_exp[changing], Thehat[changing, 3], 'kx')
# ax[3].legend()
# ax[4].plot(time_exp, Thehat[:, 4], 'k', label='k0')
# # ax[4].plot(time_exp[changing], Thehat[changing, 4], 'kx')
# ax[4].legend()
# ax[5].plot(time_exp, Thehat[:, 5], 'k', label='k1')
# # ax[5].plot(time_exp[changing], Thehat[changing, 5], 'kx')
#
# ax[5].set_xlabel('Time(s)')
# ax[5].legend()
# # ----------- degenerating physical parameters --------
# fig, ax = plt.subplots(6, 1, sharex=True)
# ax[0].plot(m1_all, 'k', label='m1')
# ax[0].legend()
# ax[1].plot(m2_all, 'k', label='m2')
# ax[1].legend()
# ax[2].plot(k1_all, 'k', label='k1')
# ax[2].legend()
# ax[3].plot(k2_all, 'k', label='k2')
# ax[3].legend()
# ax[4].plot(d1_all, 'k', label='d1')
# ax[4].legend()
# ax[5].plot(d2_all, 'k', label='d2')
# ax[5].legend()
# plt.figure()
# plt.plot(time_exp, simulator.y_pem, label='y_{pem}')
# # plt.plot(time_exp[correction], simulator.y_pem[correction], 'yx')
# # plt.plot(time_exp[stop], simulator.y_pem[stop], 'mx')
# plt.xlabel('Time')

# simulator.y_pem = np.array(simulator.y_pem)
# simulator.y_pem0 = np.array(simulator.y_pem0)
# ts = 0.05
# plt.figure()
# plt.plot(time_exp, simulator.y_pem, 'r', label='$\hat{y}_{pem}$')
# plt.plot(time_exp, simulator.y_pem0, 'g', label='$\hat{y}_{pem0}$')
# plt.plot(simulator.y_pem[:, 1]*ts, simulator.y_pem[:, 0], 'r', label='$\hat{y}_{pem}$')  #
# plt.plot(simulator.y_pem0[:, 1]*ts, simulator.y_pem0[:, 0], 'g', label='PEM resting')
# plt.plot(simulator.y_pem[:, 0], 'r-', label='$\hat{y}_{pem}$')
# plt.plot(simulator.y_pem0[:, 0], 'g-', label='$\hat{y}_{pem}0$')
# plt.plot(time_exp[correction], simulator.y_pem[correction], 'yx')
# plt.plot(time_exp[stop], simulator.y_pem[stop], 'mx')
# plt.xlabel('Time(s)')
# plt.legend()
# plt.show()
