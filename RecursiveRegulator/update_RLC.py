import matplotlib

matplotlib.use('TKAgg')
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import math
import time
from scipy import signal

sys.path.append(os.path.join("../head/"))
import header
from pem import PEM
from pem import normalize, R2
from header import NeuralStateSpaceModel_i, ForwardEulerPEM, ForwardEuler_i  

import matplotlib.pylab as pylab

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

system = 'RLC_aging_i'
np.random.seed(7)
torch.manual_seed(0)


def inductance(il, L0):
    out = L0 * (0.9 * (1 / math.pi * np.arctan(-5 * (np.abs(il) - 5)) + 0.5) + 0.1)
    return out


def white(bandwidth, time_all, std_devi, dt):  # Sample rate in Hz # Duration of the white noise in seconds
    fs_noise = 2 * bandwidth  # Noise generation sampling frequency, should be at least twice the bandwidth
    t_noise = np.arange(0, time_all, 1 / fs_noise)
    noise = std_devi * np.random.randn(len(t_noise))
    num_samples = int(time_all / dt)
    sampled_noise = signal.resample(noise, num_samples)
    return sampled_noise


class rlc:
    def __init__(self, vc, il, dvc, dil, dt):
        self.vc = vc  # capacitor voltage (V)
        self.il = il  # inductor current (A)
        self.dvc = dvc  # derivative
        self.dil = dil  # derivative
        self.dt = dt

    def get_y(self, u, noise_process=0.0, noise_measure=0.0):
        self.u = u
        self.L = inductance(self.il, L0)
        self.dvc = self.il / C
        self.dil = -1 / self.L * self.vc - R / self.L * self.il + 1 / self.L * u + np.random.randn() * noise_process
        self.vc = self.vc + self.dvc * self.dt + np.random.normal(0, 10) * noise_measure
        self.il = self.il + self.dil * self.dt + np.random.normal(0, 1) * noise_measure

        output = self.vc
        return output


# # ------
# induct = []
# iA = np.arange(0, 20, 0.1)
# for j in iA:
#     induct.append(inductance(j, L0))
# induct= np.asarray(induct, dtype=np.float32)
# induct = induct*10**6
# plt.figure()
# plt.rc('axes', titlesize=18)     # fontsize of the axes title
# plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
# plt.plot(iA, induct, 'k')
# plt.xlabel('$i_L (A)$')
# plt.ylabel('$L (\mu H)$')

# ------------------------

ts = 0.5 * 10 ** (-6)
time_all = 3 * 10 ** (-3)  # ms
N = int(time_all / ts)

dt = torch.tensor(ts, dtype=torch.float32)  #
Y_sys = []
U = []
circuit = rlc(vc=0, il=0, dvc=0, dil=0, dt=dt)

#------ sampling aging ---
changing = np.array([0.5, 1, 2.5]) * 10 ** (-3) / ts  # 3ms, 3000us
changing = changing.astype(int)

# ----------------------------------
L0 = 50 * 10 ** (-6)
C = 270 * 10 ** (-9)  #capacitor
R = 3
bandwidth = 300e2
std_devi = 80
v_in = white(bandwidth, time_all, std_devi, dt)

for i in range(changing[0]):  # original condition with noise
    Y = circuit.get_y(v_in[i], noise_measure=1e-8, noise_process=1e-7)  # 1e-3    1
    Y_sys.append(Y)
    U.append(circuit.u)
# -------------------------------------------------------------------------------
L0 = 40 * 10 ** (-6)
C = 170 * 10 ** (-9)  #capacitor
R = 7  #3 #resistor
bandwidth = 350e2
std_devi = 60
v_in = white(bandwidth, time_all, std_devi, dt)
for i in range(changing[0], changing[1]):
    Y = circuit.get_y(v_in[i], noise_measure=1e-3, noise_process=1e-1) 
    Y_sys.append(Y)
    U.append(circuit.u)

L0 = 30 * 10 ** (-6)
C = 100 * 10 ** (-9)  #capacitor
R = 14
bandwidth = 100e2
std_devi = 70
v_in = white(bandwidth, time_all, std_devi, dt)
for i in range(changing[1], changing[2]):
    Y = circuit.get_y(v_in[i], noise_measure=1e-4, noise_process=1e-2)  #
    Y_sys.append(Y)
    U.append(circuit.u)

L0 = 20 * 10 ** (-6)
C = 70 * 10 ** (-9)  #capacitor
R = 17
bandwidth = 200e2
std_devi = 30
v_in = white(bandwidth, time_all, std_devi, dt)
for i in range(changing[2], N):
    Y = circuit.get_y(v_in[i], noise_measure=1e-3, noise_process=1)  # 1e-3    
    Y_sys.append(Y)
    U.append(circuit.u)

Y_sys = np.reshape(Y_sys, (-1, 1)).astype(np.float32)
U = np.reshape(U, (-1, 1)).astype(np.float32)

U = U[:, np.newaxis]

Y_sys = normalize(Y_sys, 1)
U = normalize(U, 1)

lr = 0.001
model_filename = f"{system}"
initial_filename = f"{system}_initial"
model = NeuralStateSpaceModel_i()  #
x_fit = torch.load(os.path.join("models", initial_filename))
checkpoint = torch.load(os.path.join("models", model_filename))
model.eval()

optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': lr},
    {'params': [x_fit], 'lr': lr}
], lr=lr * 10)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # , strict=False

threshold1 = 1  #0.96  # start retrain, R2
threshold2 = 1  #0.98  # stop retrain

update = 1200
ur = 64

off = 0  #int(2* 10**(-3)/ ts ) #total 3ms = 3000us
n = 2
t = n + n * ur + n
factor = PEM(n, t, N, ur=ur)

factor.P_old2 *= 9e-2
factor.Psi_old2 *= 0.9
np.random.seed(3)
factor.Thehat_old = np.random.rand(t, 1) * 1e-2
factor.Xhat_old = np.zeros((n, 1))

simulator = ForwardEulerPEM(model=model, factor=factor, dt=1, N=N, update=update, threshold1=threshold1,
                            threshold2=threshold2, train=off)


x0 = x_fit[[0], :].detach()

u = torch.tensor(U)
y = Y_sys[:, np.newaxis]

simulator0 = ForwardEuler_i(model=model, dt=1.0)
start_time = time.time()
with torch.no_grad():
    xhat0 = simulator0(x0, u)
    xhat0 = xhat0.detach().numpy()
    xhat0 = xhat0.squeeze(1)
    yhat0 = xhat0[:, 0]
print(f"\n NN  time: {time.time() - start_time:.2f}")

start_time = time.time()
xhat_data = simulator(x0, u, y)
print(f"\nregulator time: {time.time() - start_time:.2f}")
yhat = xhat_data[:, 0]

print("nn R^2 = ", R2(Y_sys[:, 0], yhat0))

print("inference evolution R^2 = ", R2(Y_sys[:, 0], yhat))

time_exp = np.arange(N) * ts * 10 ** (6)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(time_exp, Y_sys, 'g', label='y')
ax[0].plot(time_exp, yhat0, 'r--', label='$\hat{y}_{NN}$')
ax[0].plot(time_exp[changing], Y_sys[changing], 'kx')
ax[0].set_ylabel("(a)")
ax[0].legend()
ax[0].legend(bbox_to_anchor=(0.9, 0.55))
ax[1].plot(time_exp, Y_sys, 'g', label='y')
ax[1].plot(time_exp, yhat, 'r--', label='$\hat{y}$')
ax[1].plot(time_exp[changing], Y_sys[changing], 'kx')
if off != 0:
    ax[1].plot(time_exp[off], Y_sys[off], 'bx')
ax[1].set_ylabel("(b)")
ax[1].legend()
ax[1].legend(bbox_to_anchor=(0.9, 0.6))
ax[1].set_xlabel('time($\mu s$)')

