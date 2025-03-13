""""

"""
import matplotlib
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('F:/Project/head/tight.mplstyle')

import matplotlib.pylab as pylab
pylab.rcParams['font.family'] = "Times New Roman"#'sans-serif'


import sys, os
import math
import time
import header
from pem import PEM
from pem import normalize, R2
from header import MechanicalSystem_u, ForwardEulerPEM, ForwardEuler


params = {
    # 'figure.figsize': (4.8, 3.7),
          'legend.fontsize': 11,
          'axes.labelsize': 15,
          'axes.labelpad': 0.5,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'legend.labelspacing': 0.3

          }
pylab.rcParams.update(params)

# params = {
#     # 'figure.figsize': (4.8, 3.7),
#     'axes.labelsize': 11,
#     'axes.labelpad': 0.5,
#     'xtick.labelsize': 11,
#     'ytick.labelsize': 11,
#     'legend.fontsize': 11,
#     'legend.labelspacing': 0.1,
#     'legend.borderpad': 0.2,
#     'legend.borderaxespad': 0.25,
#     'legend.handletextpad': 0.3,
#     'legend.handlelength': 1,
#     'legend.loc': 'lower right',
#
# }
# pylab.rcParams.update(params)


# -- sin/cos wave ---
def sinwave(dt, time_all, w=0.5):
    out = []
    A = 1
    for k in range(int(time_all / dt)):
        x = A * np.cos(w * k * math.pi * dt)
        # x = A * np.sin(w*k * math.pi* dt)
        out.append(x)
    return out


# -- tri wave ---
def triangle(dt, time_all):
    out = []
    p = 2
    for k in range(int(time_all / dt)):
        x = 2 * np.abs(k * dt / p - math.floor(k * dt / p + 0.5))  # 2 * -1

        out.append(x)
    return out


class EMPS(object):
    def __init__(self, dt, pos, vel, acc, u):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.dt = dt
        self.u = u  # control signal voltage

    def measure(self, pos_ref, noise_process=0.0, noise_measure=0.0):
        self.u = kp * kv * (pos_ref - self.pos) - kv * self.vel
        if self.u > satu:  # Maximum voltage (saturation)
            self.u = satu
        if self.u < -satu:
            self.u = -satu
        force = gt * self.u
        self.acc = force / M - Fv / M * self.vel - Fc / M * np.sign(
            self.vel) - offset / M + np.random.randn() * noise_process
        self.vel = self.vel + self.acc * self.dt
        self.pos = self.pos + self.vel * self.dt
        # position limit
        # if self.pos > 0.5:
        #     self.pos = 0.5
        # if self.pos < -0.5:
        #     self.pos = -0.5
        output = self.pos + np.random.randn() * noise_measure
        return output


gt, kp, kv = 35.15, 160.18, 243.45
M, Fv = 95.1089, 203.5034
Fc, offset = 20.3935, -3.1648
# Fc, offset = 0, 0  # remove nonlinear part
satu = 10  # saturation
system = 'update_ku'
dt = 0.005
time_all = np.array([70])
changing = np.array([20, 50]) / dt
changing = changing.astype(int)
# dt = 0.05
# time_all = np.array([100])  #
# change1 = int(time_all/10*3)
# change2 = int(time_all/10*6)
# changing = np.array([change1, change2])/dt


N = int(time_all[-1] / dt)
time_exp = np.arange(N) * dt
Y_sys = []
U = []
M_all = []
Fc_all = []
Fv_all = []
ref_signal = []
p_ref = sinwave(dt, time_all)  # ref signal not change
sig = 'sinwave'
p_tri = triangle(dt, time_all)


sampling = EMPS(dt, pos=0, vel=0, acc=0, u=0)
for i in range(changing[0]):
    y = sampling.measure(p_ref[i], noise_process=10 ** -4, noise_measure=10 ** -5)
    Y_sys.append(y)
    U.append(sampling.u)
    M_all.append(M)
    Fv_all.append(Fv)
    Fc_all.append(Fc)
    ref_signal.append(p_ref[i])

aging = 0.80
offset = offset * 0.99
M = M * aging
Fv = Fv * aging
Fc = Fc * aging
for i in range(changing[0], changing[1]):
    y = sampling.measure(p_ref[i], noise_process=10 ** -3, noise_measure=10 ** -4)
    Y_sys.append(y)
    U.append(sampling.u)
    M_all.append(M)
    Fv_all.append(Fv)
    Fc_all.append(Fc)
    ref_signal.append(p_ref[i])

for i in range(changing[1], N):
    y = sampling.measure(p_tri[i], noise_process=10 ** -3, noise_measure=10 ** -4)
    Y_sys.append(y)
    U.append(sampling.u)
    M_all.append(M)
    Fv_all.append(Fv)
    Fc_all.append(Fc)
    ref_signal.append(p_tri[i])




# U = np.loadtxt("data_U_change.txt",dtype=np.float32)
# # Y_sys= np.loadtxt("data_Y_change.txt",dtype=np.float32)

##------
Y_sys = normalize(Y_sys, 1)
U = normalize(U, 1)
np.savetxt("data_U_change_emps.txt", U)
np.savetxt("data_Y_change_emps.txt", Y_sys)

Y_sys = np.asarray(Y_sys, dtype=np.float32)
U = np.asarray(U, dtype=np.float32)
U = U[:, np.newaxis]
dt = torch.tensor(dt, dtype=torch.float32)
u = torch.tensor(U[:, None, :])  # [:, None, :]
y = Y_sys[:, np.newaxis]

lr = 0.0001  # not used in PEM updateing

model_filename = f"{system}"
initial_filename = f"{system}_initial"
model = MechanicalSystem_u(dt=dt)  #

x_fit = torch.load(os.path.join("models", initial_filename))
checkpoint = torch.load(os.path.join("models", model_filename))
# model.eval()
x0 = x_fit[[0], :].detach()

optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': lr},
    {'params': [x_fit], 'lr': lr}
], lr=lr * 10)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # , strict=False
start_time = time.time()


simulator0 = ForwardEuler(model=model, dt=dt)
with torch.no_grad():
    xhat0 = simulator0(x0, u)
    xhat0=xhat0.detach().numpy()
    xhat0 = xhat0.squeeze(1)
    yhat0 = xhat0[:, 0]
print(f"\n NN Train time: {time.time() - start_time:.2f}")

threshold1 = 1 #0.90  # start retrain, R2
# threshold2 = 0.98  # stop retrain
threshold2 = 1  # stop retrain
# threshold2 = 0.97  # stop retrain
n=2
ur = 64
t = n +n*ur + n
factor = PEM(n, t, N, ur=ur)
# factor.P_old2 *= 0.09
factor.P_old2 *= 9e-2 #4 #0.0009

factor.Psi_old2 *= 0.9
np.random.seed(3)

# factor.Thehat_old = np.random.rand(t, 1) * 0.01
factor.Thehat_old = np.random.rand(t, 1) * 1e-2#4#10#7#10 #4  8

# print('seed=', np.random.get_state()[1][0])#
# factor.Xhat_old = np.array([[2], [0]])
factor.Xhat_old = np.zeros((n, 1))
# update = 5 # original update
# update = 9 # add pem fix
update = 1201 # q size == hidden, inside koopman space
off = 0#int(35/dt)
simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N, update=update,
                            threshold1=threshold1, threshold2=threshold2, train=off)  # optimizer=optimizer,

# x_fit = np.zeros((1, n_x), dtype=np.float32)
# x_fit[0, 0] = np.copy(Y_sys[0, 0])
# x_fit[0, 1] = 0
# x_step = x0
# x0 = torch.tensor(x_fit[[0], :], dtype=torch.float32)


start_time = time.time()
xhat_data = simulator(x0, u, y)
print(f"\nTrain time: {time.time() - start_time:.2f}")
# ----- optimization inside NN loop, stepwise --------
yhat = xhat_data[:, 0]
# Thehat = simulator.Thehat
# stop = simulator.stop
# correction = simulator.correction
# print(f'update at {correction}')
# print(f'stop at {stop}')

# # ->>>---- update == False, optimization outside NN loop, soo faster than stepwise --
# yhat_stable = xhat_data[:, 0]
# factor.forward(y, yhat_stable)
# yhat = factor.Yhat_data
# Thehat = factor.Thehat_data
# ------ <<<--------------------------------------
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
    fit_val = 100*(1 - err_norm/err_mean_norm)

    return fit_val
print('fit = ', fit_index(Y_sys,yhat))

print("inference evolution R^2 = ", R2(Y_sys, yhat))
print('nn r2=', R2(Y_sys, yhat0))

fig, ax = plt.subplots(2, 1, sharex=True,  tight_layout=True, figsize=(9, 6))  #
ax[0].plot(time_exp, Y_sys, 'g', label='$y$')
ax[0].plot(time_exp, yhat0, 'r--', label='$\hat{y}_{N}$')
ax[0].plot(time_exp[changing], Y_sys[changing], 'kx')
ax[0].set_ylabel("(a)")
ax[0].legend(bbox_to_anchor=(0.9, 0.5))
ax[1].plot(time_exp, Y_sys, 'g', label='$y$')
ax[1].plot(time_exp, yhat, 'r--', label='$\hat{y}$')
ax[1].plot(time_exp[changing], Y_sys[changing], 'kx')
if off!=0:
    ax[1].plot(time_exp[off], Y_sys[off], 'bx')

ax[1].set_ylabel("(b)")
ax[1].legend(bbox_to_anchor=(0.9, 0.6))

# ax[2].plot(time_exp, U, 'k', label='$u$')
# # ax[1].plot(time_exp[changing], U[changing], 'kx', label='changing')
# ax[2].set_ylabel("(c)")
# ax[2].legend()
# ax[3].plot(time_exp, ref_signal, 'k', label='$ref$')  #
# ax[3].legend()
# ax[3].set_ylabel("(d)")
ax[1].set_xlabel('Time($s$)')
# #
# fig, ax = plt.subplots(6, 1, sharex=True)
# ax[0].plot(time_exp, Thehat[:, 0], 'g', label='a0')
# ax[0].plot(time_exp[changing], Thehat[changing, 0], 'kx')
# # ax[0].plot(time_exp[correction], Thehat[correction, 0], 'yx')
# # ax[0].plot(time_exp[stop], Thehat[stop, 0], 'mx')
# ax[0].legend()
# ax[1].plot(time_exp, Thehat[:, 1], 'g', label='a1')
# ax[1].plot(time_exp[changing], Thehat[changing, 1], 'kx')
# ax[1].legend()
# ax[2].plot(time_exp, Thehat[:, 2], 'b', label='b0')
# ax[2].plot(time_exp[changing], Thehat[changing, 2], 'kx')
# ax[2].legend()
# ax[3].plot(time_exp, Thehat[:, 3], 'b', label='b1')
# ax[3].plot(time_exp[changing], Thehat[changing, 3], 'kx')
# ax[3].legend()
# ax[4].plot(time_exp, Thehat[:, 4], 'k', label='k0')
# ax[4].plot(time_exp[changing], Thehat[changing, 4], 'kx')
# ax[4].legend()
# ax[5].plot(time_exp, Thehat[:, 5], 'k', label='k1')
# ax[5].plot(time_exp[changing], Thehat[changing, 5], 'kx')
# ax[5].legend()
# ax[5].set_xlabel('time(s)')
# ----------- degenerating physical parameters --------
# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(M_all, 'g', label='M')
# ax[0].legend()
# ax[1].plot(Fc_all, 'k', label='Fc')
# ax[1].legend()
# ax[2].plot(Fv_all, 'k', label='Fv')
# ax[2].legend()

# simulator.y_pem = np.array(simulator.y_pem)
# simulator.y_pem0 = np.array(simulator.y_pem0)
# ts = 0.005
# plt.figure()
# # plt.plot(time_exp, simulator.y_pem, 'r', label='$\hat{y}_{pem}$')
# # plt.plot(time_exp, simulator.y_pem0, 'g', label='$\hat{y}_{pem0}$')
# plt.plot(simulator.y_pem[:, 1]*ts, simulator.y_pem[:, 0], 'r', label=r'$\bar{y}_{pem}$')
# plt.plot(simulator.y_pem0[:, 1]*ts, simulator.y_pem0[:, 0], 'g', label='PEM disabled')
# # plt.plot(simulator.y_pem[:, 0], 'r-', label='$\hat{y}_{pem}$')
# # plt.plot(simulator.y_pem0[:, 0], 'g-', label='$\hat{y}_{pem}0$')
# # plt.plot(time_exp[correction], simulator.y_pem[correction], 'yx')
# # plt.plot(time_exp[stop], simulator.y_pem[stop], 'mx')
# plt.xlabel('time(s)')
# plt.legend()

# simulator.correction = np.array(simulator.correction)
# simulator.stop = np.array(simulator.stop)
# plt.figure()
# plt.plot(simulator.correction[:, 1], simulator.correction[:, 0], 'r', label='update')  # time_exp,
# plt.plot(simulator.stop[:, 1], simulator.stop[:, 0], 'b', label='stop')
# plt.xlabel('time(s)')
# plt.legend()

# plt.figure()
# plt.plot(simulator.r2, 'r', label='$R^2$')  # time_exp,
# plt.xlabel('time(s)')
# plt.legend()
#
# plt.figure()
# plt.plot(np.abs(simulator.err), 'r', label='$error$')  # time_exp,
# plt.xlabel('time(s)')
# plt.legend()
# plt.show()

# yhat_dynonet12 = np.loadtxt('yhat_dynonet12.txt', dtype=np.float32)
# yhat_dynonet2 = np.loadtxt('yhat_dynonet2.txt', dtype=np.float32)
# yhat_edmdc = np.loadtxt('yhat_edmdc.txt', dtype=np.float32)
# yhat_edmdc_online = np.loadtxt('yhat_edmdc_online.txt', dtype=np.float32)
# # print('R2 dynonet= ', R2(Y_sys, yhat_dynonet12))
#
# fig, ax = plt.subplots(1, 1, sharex=True,  tight_layout=True, figsize=(9, 6))  #
# ax.plot(time_exp, Y_sys, 'k', label='$y$')
# ax.plot(time_exp, yhat, 'r--', label='$\hat{y}$')
# ax.plot(time_exp, yhat_dynonet12, 'g--', label='$\hat{y}_{dynonet}$')
# ax.plot(time_exp, yhat_edmdc_online, 'b--', label='$\hat{y}_{edmdc}$')
# ax.plot(time_exp[changing], Y_sys[changing], 'kx')
# ax.set_ylabel("Prediction")
# ax.legend()
# ax.set_xlabel('Time($s$)')

# fig, ax = plt.subplots(1, 1, sharex=True,  tight_layout=True, figsize=(9, 6))  #
# ax.plot(time_exp, Y_sys-yhat_dynonet12, 'g--', label='$dynonet$')
# ax.plot(time_exp, Y_sys-yhat_edmdc_online, 'b--', label='$edmdc$')
# ax.plot(time_exp, Y_sys-yhat, 'r--', label='$RR$')
# ax.plot(time_exp[changing], (Y_sys-yhat)[changing], 'kx')
# ax.set_ylabel("Error")
# ax.legend()
# ax.set_xlabel('Time($s$)')