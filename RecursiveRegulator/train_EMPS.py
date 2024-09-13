"""
train EMPS and save, static condition
"""
import pandas as pd
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import os
import sys
import math
sys.path.append(os.path.join("../head/"))
from header import R2, normalize, MechanicalSystem_i, ForwardEuler

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# -- sin/cos wave ---
def sinwave(dt, time, w=0.5):
    out = []
    A = 1
    for k in range(int(time / dt)):
        x = A * np.cos(w * k * math.pi * dt)
        # x = A * np.sin(w*k * math.pi* dt)
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
satu = 10  # saturation
# Fc, offset = 0, 0  # remove nonlinear part
# satu = 100  # saturation
# # -------------------------------------------------
# # ------- time-invariant system -----
system = 'update_i'

dt = 0.005
time_all = np.array([10])# 20seconds


sampling = EMPS(dt, pos=0, vel=0, acc=0, u=0)

Y_sys = []
U = []

p_ref = sinwave(dt, time_all)
sig = 'sinwave'

simu = 'train'
# simu = 'noise'
noise = 0
if simu == 'train':
    noise = 0
if simu == 'noise':
    noise = 0.01
#
for i in p_ref:
    p_control = i
    y = sampling.measure(p_control, noise * 10, noise)
    Y_sys.append(y)
    U.append(sampling.u)

Y_sys = np.array(Y_sys, dtype=np.float32)
U = np.array(U, dtype=np.float32)
Y_sys = Y_sys[:, np.newaxis]
U = U[:, np.newaxis]


Y_sys = normalize(Y_sys, 1)
U = normalize(U, 1)


def vel(pos):
    v_est = np.concatenate((np.array([0]), np.diff(pos[:, 0])))
    v_est = v_est.reshape(-1, 1) / dt
    return v_est


v_est = vel(Y_sys)
dt = torch.tensor(dt , dtype=torch.float32)  #
# -----------------------------------------------------------------------
num_epoch = 10000
batch_num = 64
batch_length = 32
weight = 1.0
lr = 0.0001

n_x = 2
N = len(Y_sys)
np.random.seed(3)
torch.manual_seed(3407)

X = np.zeros((N, n_x), dtype=np.float32)
X[:, 0] = np.copy(Y_sys[:, 0])
X[:, 1] = np.copy(v_est[:, 0])
x_fit = torch.tensor(X, dtype=torch.float32, requires_grad=True)

model = MechanicalSystem_i(dt=dt)
simulator = ForwardEuler(model=model, dt=dt)
params_net = list(simulator.model.parameters())
params_initial = [x_fit]
optimizer = torch.optim.Adam([
    {'params': params_net, 'lr': lr},
    {'params': params_initial, 'lr': lr}
], lr=lr*10)


def get_batch(batch_num=batch_num, batch_length=batch_length):
    batch_start = np.random.choice(np.arange(N - batch_length, dtype=np.int64), batch_num, replace=False)
    batch_index = batch_start[:, np.newaxis] + np.arange(batch_length)  # batch sample index
    batch_index = batch_index.T  # (batch_length, batch_num, n_x)
    batch_x0 = x_fit[batch_start, :]  # (batch_num, n_x), initials in each batch
    batch_x = x_fit[[batch_index]]
    batch_u = torch.tensor(U[batch_index, :])
    batch_y = torch.tensor(Y_sys[batch_index])
    return batch_x0, batch_x, batch_u, batch_y

# compute initial error as scale.
with torch.no_grad():
    batch_x0, batch_x, batch_u, batch_y = get_batch()
    batch_xhat = simulator(batch_x0, batch_u)
    traced_simulator = torch.jit.trace(simulator, (batch_x0, batch_u))
    batch_yhat = batch_xhat[:, :, [0]]
    error_init = batch_yhat - batch_y
    error_scale = torch.sqrt(torch.mean(error_init**2, dim=(0, 1)))  # root MSE

LOSS = []

start_time = time.time()
for epoch in range(num_epoch):
    batch_x0, batch_x, batch_u, batch_y = get_batch()
    batch_xhat = traced_simulator(batch_x0, batch_u)
    # output loss
    batch_yhat = batch_xhat[:, :, [0]]
    error_out = batch_yhat - batch_y
    loss_out = torch.mean((error_out/error_scale[0])**2)  # divided by scale
    # state estimate loss
    error_state = (batch_xhat - batch_x)/error_scale
    loss_state = torch.mean(error_state**2)  # MSE

    loss = loss_out + weight * loss_state
    LOSS.append(loss.item())
  

    if (epoch+1) % 100 == 0:  # unpack before print
        print(f'epoch {epoch+1}/{num_epoch}: loss= {loss.item():.5f}, yhat= {batch_yhat[-1, -1, 0]:.4f}')

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


print(f"\nTrain time: {time.time() - start_time:.2f}")
# Save model
if not os.path.exists("models"):
    os.makedirs("models")
model_filename = f"{system}"
initial_filename = f"{system}_initial"

torch.save({'model_state_dict': simulator.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
           os.path.join("models", model_filename))


torch.save(x_fit, os.path.join("models", initial_filename))

fig, ax = plt.subplots(1, 1)
ax.plot(LOSS, label='loss_total')
ax.grid(True)
ax.set_xlabel("Iteration")
plt.legend()


# initial state estimate
x0_vali = x_fit[0, :].detach().numpy()
x0_vali[1] = 0.0
x0_vali = torch.tensor(x0_vali)
u_vali = torch.tensor(U)
with torch.no_grad():
    xhat_vali = simulator(x0_vali[None, :], u_vali[:, None])
    xhat_vali=xhat_vali.detach().numpy()
    xhat_vali = xhat_vali.squeeze(1)
    yhat_vali = xhat_vali[:, 0]


print("R^2 = ", R2(Y_sys[:, 0], yhat_vali))  # .detach().numpy()
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(Y_sys, 'g', label='y')
ax[0].plot(yhat_vali, 'r--', label='$\hat{y}$')
ax[0].legend()

ax[1].plot(U, 'k', label='u')
ax[1].set_xlabel('Time')
ax[1].legend()


