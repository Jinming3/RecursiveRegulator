
import matplotlib
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

import os
import sys
import math
import time

sys.path.append(os.path.join("../head/"))
import header
from pem import PEM
from pem import normalize, R2
from header import MechanicalSystem_i, ForwardEulerPEM, ForwardEuler_i


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
system = 'update_i'
dt = 0.005
time_all = np.array([70])
changing = np.array([20, 50]) / dt

N = int(time_all[-1] / dt)
time_exp = np.arange(N) * dt
Y_sys = []
U = []
M_all = []
Fc_all = []
Fv_all = []
ref_signal = []

changing = changing.astype(int)

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

aging = 0.85
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

Y_sys = normalize(Y_sys, 1)
U = normalize(U, 1)
Y_sys = np.asarray(Y_sys, dtype=np.float32)
U = np.asarray(U, dtype=np.float32)
U = U[:, np.newaxis]
dt = torch.tensor(dt, dtype=torch.float32)
u = torch.tensor(U[:, None, :])
y = Y_sys[:, np.newaxis]

lr = 0.0001

model_filename = f"{system}"
initial_filename = f"{system}_initial"
model = MechanicalSystem_i(dt=dt)  

x_fit = torch.load(os.path.join("models", initial_filename))
checkpoint = torch.load(os.path.join("models", model_filename))
model.eval()
x0 = x_fit[[0], :].detach()

optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': lr},
    {'params': [x_fit], 'lr': lr}
], lr=lr * 10)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # , strict=False
start_time = time.time()

simulator0 = ForwardEuler_i(model=model, dt=dt)
with torch.no_grad():
    xhat0 = simulator0(x0, u)
    xhat0 = xhat0.detach().numpy()
    xhat0 = xhat0.squeeze(1)
    yhat0 = xhat0[:, 0]
print(f"\n NN Train time: {time.time() - start_time:.2f}")

threshold1 = 1  #0.90  # start retrain, R2
threshold2 = 1  # 0.97 stop retrain

n = 2
ur = 64
t = n + n * ur + n
factor = PEM(n, t, N, ur=ur)
factor.P_old2 *= 9e-2

factor.Psi_old2 *= 0.9
np.random.seed(3)

factor.Thehat_old = np.random.rand(t, 1) * 1e-2

factor.Xhat_old = np.zeros((n, 1))
update = 1200  # q size == hidden, inside koopman space
off = 0  #int(35/dt)
simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N, update=update,
                            threshold1=threshold1, threshold2=threshold2, train=off)


start_time = time.time()
xhat_data = simulator(x0, u, y)
print(f"\nTrain time: {time.time() - start_time:.2f}")
yhat = xhat_data[:, 0]


# ------ <<<--------------------------------------
print("inference evolution R^2 = ", R2(Y_sys, yhat))
print('nn r2=', R2(Y_sys, yhat0))
fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(time_exp, Y_sys, 'g', label='y')
ax[0].plot(time_exp, yhat0, 'r--', label='$\hat{y}_{NN}$')
ax[0].plot(time_exp[changing], Y_sys[changing], 'kx')
ax[0].set_ylabel("(a)")
ax[0].legend(bbox_to_anchor=(0.9, 0.5))
ax[1].plot(time_exp, Y_sys, 'g', label='y')
ax[1].plot(time_exp, yhat, 'r--', label='$\hat{y}$')
ax[1].plot(time_exp[changing], Y_sys[changing], 'kx')
if off != 0:
    ax[1].plot(time_exp[off], Y_sys[off], 'bx')

ax[1].set_ylabel("(b)")
ax[1].legend(bbox_to_anchor=(0.9, 0.6))
ax[2].plot(time_exp, U, 'k', label='u')
ax[2].set_ylabel("(c)")
ax[2].legend()
ax[3].plot(time_exp, ref_signal, 'k', label='ref')  #
ax[3].legend()
ax[3].set_ylabel("(d)")
ax[3].set_xlabel('time(s)')
