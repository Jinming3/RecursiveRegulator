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
from numpy.linalg import inv

sys.path.append(os.path.join("F:/Project/head/"))
from header import R2, normalize, MechanicalSystem, ForwardEuler

# -------------
# define the system parameters
m1 = 20
m2 = 20
k1 = 1000
k2 = 2000
d1 = 1
d2 = 5
Fc = 0.05 # 20.3935

def sinwave(dt, i, w=0.5): 
    A = 2
    x = A * np.cos(w * i * math.pi * dt)
    
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

    def get_y(self,  noise_process=0.0, noise_measure=0.0):
        self.u = sinwave(self.dt, i)
       
        self.acc1 = -(k1 + k2 / m1) * self.pos1 - (d1 + d2) / m1 * self.vel1 + k2 / m1 * self.pos2 + d2 / m1 * self.vel2 - Fc / m1 * np.sign(
            self.vel1)
        self.acc2 = k2 / m2 * self.pos1 + d2 / m2 * self.vel1 - k2 / m2 * self.pos2 - d2 / m2 * self.vel2 + 1 / m2 * self.u - Fc / m2 * np.sign(
            self.vel2) + 0 * np.random.randn() * noise_process
        self.vel1 = self.vel1 + self.acc1 * self.dt
        self.vel2 = self.vel2 + self.acc2 * self.dt
        self.pos1 = self.pos1 + self.vel1 * self.dt
        self.pos2 = self.pos2 + self.vel2 * self.dt
        output = self.pos2 + np.random.randn() * noise_measure
        return output


time_all = 100
dt = 0.05
N = int(time_all / dt)
time_exp = np.arange(N) * dt


real = Motion(dt, pos1=0, pos2=0, vel1=0, vel2=0, acc1=0, acc2=0)

Y_sys = []
U = []
for i in range(N):
    y = real.get_y()
    Y_sys.append(y)
    U.append(real.u)

#  -- prepare --
Y_sys = normalize(Y_sys, 1)
U = normalize(U, 1)  # unless U is constant
Y_sys = np.array(Y_sys, dtype=np.float32)
U = np.array(U, dtype=np.float32)
Y_sys = Y_sys[:, np.newaxis]
U = U[:, np.newaxis]


def vel(pos):
    v_est = np.concatenate((np.array([0]), np.diff(pos[:, 0])))
    v_est = v_est.reshape(-1, 1) / dt
    return v_est

v_est = vel(Y_sys)
dt = torch.tensor(dt, dtype=torch.float32)  #
# # -----------------------------------------------------------------------
system = 'two_spring_motion5'
num_epoch = 10000
batch_num = 64
batch_length = 32
weight = 1.0  # initial state weight in loss function
lr = 0.0001
# state space
n_x = 2
np.random.seed(3)
torch.manual_seed(3407)

X = np.zeros((N, n_x), dtype=np.float32)
X[:, 0] = np.copy(Y_sys[:, 0])
X[:, 1] = np.copy(v_est[:, 0])
x_fit = torch.tensor(X, dtype=torch.float32, requires_grad=True)

model = MechanicalSystem(dt=dt)

simulator = ForwardEuler(model=model, dt=dt)
params_net = list(simulator.model.parameters())
params_initial = [x_fit]
optimizer = torch.optim.Adam([
    {'params': params_net, 'lr': lr},
    {'params': params_initial, 'lr': lr}
], lr=lr * 10)


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
    error_scale = torch.sqrt(torch.mean(error_init ** 2, dim=(0, 1)))  # root MSE

LOSS = []

start_time = time.time()
for epoch in range(num_epoch):
    batch_x0, batch_x, batch_u, batch_y = get_batch()
    batch_xhat = traced_simulator(batch_x0, batch_u)
    # output loss
    batch_yhat = batch_xhat[:, :, [0]]
    error_out = batch_yhat - batch_y
    loss_out = torch.mean((error_out / error_scale[0]) ** 2)  # divided by scale
    # state estimate loss
    error_state = (batch_xhat - batch_x) / error_scale
    loss_state = torch.mean(error_state ** 2)  # MSE

    loss = loss_out + weight * loss_state
    LOSS.append(loss.item())

    if (epoch + 1) % 100 == 0:  # unpack before print
        print(f'epoch {epoch + 1}/{num_epoch}: loss= {loss.item():.5f}, yhat= {batch_yhat[-1, -1, 0]:.4f}')

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

# fig, ax = plt.subplots(1, 1)
# ax.plot(LOSS, label='loss_total')
# ax.grid(True)
# ax.set_xlabel("Iteration")
# plt.legend()

# initial state estimate
x0_vali = x_fit[0, :].detach().numpy()
x0_vali[1] = 0.0
x0_vali = torch.tensor(x0_vali)
u_vali = torch.tensor(U)
with torch.no_grad():
    xhat_vali = simulator(x0_vali[None, :], u_vali[:, None])
    xhat_vali = xhat_vali.detach().numpy()
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
# plt.show()
