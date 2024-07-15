""""

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
from scipy import signal
sys.path.append(os.path.join("F:/Project/head/"))
from header import R2, normalize, ForwardEuler, NeuralStateSpaceModel  # MechanicalSystem,
import matplotlib.pylab as pylab

params = {
    # 'figure.figsize': (4.8, 3.7),
    'legend.fontsize': 11,
    'axes.labelsize': 11,
    'axes.labelpad': 0.5,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.labelspacing': 0.3
}
pylab.rcParams.update(params)

# df_data = pd.read_csv("F:/Project/DATA/RLC/RLC_data_id.csv")
# Y_sys = np.array(df_data[['V_C']]).astype(np.float32)
# time_exp = np.array(df_data['time']).astype(np.float32)
# dt = time_exp[1] - time_exp[0]
# U = np.array(df_data[['V_IN']]).astype(np.float32)
# X = np.array(df_data[['V_C', 'I_L']]).astype(np.float32)

# np.random.seed(3)
# torch.manual_seed(3407)
np.random.seed(0)
torch.manual_seed(0)
def inductance(il, L0):
    out = L0 * (0.9 * (1 / math.pi * np.arctan(-5 * (np.abs(il) - 5)) + 0.5) + 0.1)
    return out

def white(bandwidth, time_all, std_devi, dt): # Sample rate in Hz # Duration of the white noise in seconds
    fs_noise = 2*bandwidth # Noise generation sampling frequency, should be at least twice the bandwidth
    t_noise = np.arange(0, time_all, 1/fs_noise)
    noise = std_devi*np.random.randn(len(t_noise))
    num_samples = int(time_all/dt)
    sampled_noise = signal.resample(noise, num_samples)
    return sampled_noise


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

# ----
class rlc:
    def __init__(self, vc, il, dvc, dil,  dt):
        self.vc = vc  # capacitor voltage (V)
        self.il = il  # inductor current (A)
        self.dvc = dvc  # derivative
        self.dil = dil  # derivative
        self.dt = dt


    def get_y(self, u, noise_process=0, noise_measure=0):
        self.u = u
        self.L = inductance(self.il, L0)
        self.dvc = self.il /C
        self.dil= -1/self.L * self.vc -R/self.L*self.il + 1/self.L * u + np.random.randn() * noise_process
        self.vc = self.vc + self.dvc * self.dt + np.random.normal(0, 10) * noise_measure
        self.il = self.il + self.dil * self.dt + np.random.normal(0, 1) * noise_measure

        output = self.vc
        return output

time_all = 2 *10**(-3)  # 2ms
dt=0.5*10**(-6)

L0=50*10**(-6)

C = 270 * 10 ** (-9)  #capacitor
R=3 #resistor 5 #
bandwidth= 300e2 #150e2 #150e3
std_devi = 80

# -------
system = 'RLC_aging'
circuit = rlc(vc=0, il=0, dvc=0, dil=0, dt=dt)

v_in = white(bandwidth, time_all, std_devi, dt)
Y_sys = []
U = []
X = []

for i in range(int(time_all/dt)):
    Y = circuit.get_y(v_in[i], noise_measure=0, noise_process=0) #       1  ,sinwave(dt=dt, i=i, w=bandwidth, A=1)
    Y_sys.append(Y)
    U.append(circuit.u)
    X.append([circuit.vc, circuit.il])




dt = torch.tensor(dt, dtype=torch.float32)  #
Y_sys = np.reshape(Y_sys, (-1, 1)).astype(np.float32)
U = np.reshape(U, (-1, 1)).astype(np.float32)
X = np.array(X).astype(np.float32)
N = len(Y_sys)


Y_sys = normalize(Y_sys, 1)
U = normalize(U, 1)
X = normalize(X, 1)

# -----------------------------------------------------------------------
error_scale = 0.01#10 #

num_epoch = 10000
batch_num = 64
batch_length = 64
# batch_length = 128
# batch_length = 128
weight = 1.0  # initial state weight in loss function
lr = 0.001
# lr = 0.001
n_x = 2


x_fit = torch.tensor(X, dtype=torch.float32, requires_grad=True)
model = NeuralStateSpaceModel()
# model = MechanicalSystem(dt=dt)
# simulator = header.RK4(model=model, dt=dt)
simulator = ForwardEuler(model=model, dt=1.0)  #, dt=dt # not acceleration, no dt
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
    # traced_simulator = torch.jit.trace(simulator, (batch_x0, batch_u))
    batch_yhat = batch_xhat[:, :, [0]]
    error_init = batch_yhat - batch_y
    # error_scale = torch.sqrt(torch.mean(error_init ** 2, dim=(0, 1)))  # root MSE

LOSS = []

start_time = time.time()
for epoch in range(num_epoch):
    batch_x0, batch_x, batch_u, batch_y = get_batch()
    # batch_xhat = traced_simulator(batch_x0, batch_u)
    batch_xhat = simulator(batch_x0, batch_u)
    # output loss
    batch_yhat = batch_xhat[:, :, [0]]
    error_out = batch_yhat - batch_y
    loss_out = torch.mean((error_out / error_scale) ** 2)  # divided by scale
    # state estimate loss
    # error_state = (batch_xhat - batch_x) / error_scale
    # loss_state = torch.mean(error_state ** 2)  # MSE
    # loss = loss_out + weight * loss_state
    loss = loss_out
    LOSS.append(loss.item())

    if (epoch + 1) % 100 == 0:  # unpack before print
        print(
            f'epoch {epoch + 1}/{num_epoch}: loss= {loss.item():.5f}, y= {batch_y[-1, -1, 0]:.4f}, yhat= {batch_yhat[-1, -1, 0]:.4f}')

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(f"\nTrain time: {time.time() - start_time:.2f}")
# Save model
if not os.path.exists("models"):
    os.makedirs("models")
model_filename = f"{system}"
initial_filename = f"{system}_initial"

# torch.save({'model_state_dict': simulator.model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict()},
#            os.path.join("models", model_filename))
#
# torch.save(x_fit, os.path.join("models", initial_filename))

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
    # xhat_vali = simulator(x0_vali, u_vali)

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
