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
from header import MechanicalSystem_i, ForwardEulerPEM, ForwardEuler_i

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


#   ---- motion----
def sinwave(dt, i, w, A):
    x = A * np.sin(w * i * math.pi * dt)

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

    def measure(self, ref, noise_process=0.0, noise_measure=0.0):  # ref is a scalar value, not a function
        self.u = ref
        self.acc1 = -(k1 + k2 / m1) * self.pos1 - (
                    d1 + d2) / m1 * self.vel1 + k2 / m1 * self.pos2 + d2 / m1 * self.vel2 - Fc / m1 * np.sign(self.vel1)
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
changing = np.array([20, 40, 55, 70, 100, 125]) / dt
changing = changing.astype(int)

sampling = Motion(dt, pos1=0, pos2=0, vel1=0, vel2=0, acc1=0, acc2=0)
scale = 1e-3
for i in range(changing[0]):  # original condition with noise
    y = sampling.measure(ref=sinwave(dt=dt, i=i, w=0.5, A=3), noise_process=scale * 1e-5,
                         noise_measure=scale * 1e-8)  #2 1e-5  1e-8
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
    y = sampling.measure(ref=sinwave(dt=dt, i=i, w=0.5, A=1.5), noise_process=scale * 1e-3, noise_measure=scale * 1e-3)
    Y_sys.append(y)
    U.append(sampling.u)
    m1_all.append(m1)
    m2_all.append(m2)
    k1_all.append(k1)
    k2_all.append(k2)
    d1_all.append(d1)
    d2_all.append(d2)

for i in range(changing[1], changing[2]):
    y = sampling.measure(ref=1, noise_process=scale * 0.01, noise_measure=scale * 0.001)
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
    y = sampling.measure(ref=-1, noise_process=scale * 0.01, noise_measure=scale * 0.001)
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
    y = sampling.measure(ref=sinwave(dt, i, 1 / 2, 0.6) + sinwave(dt, i, 1 / 3, 0.4), noise_process=scale * 0.1,
                         noise_measure=scale * 0.001)
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
for i in range(changing[4], changing[5]):
    y = sampling.measure(ref=sinwave(dt, i, 0.1, 0.6), noise_process=scale * 0.1, noise_measure=scale * 0.001)
    Y_sys.append(y)
    U.append(sampling.u)
    m1_all.append(m1)
    m2_all.append(m2)
    k1_all.append(k1)
    k2_all.append(k2)
    d1_all.append(d1)
    d2_all.append(d2)

m1 = m1 / math.sqrt(m1)
m2 = m2 * math.sqrt(m2)
k1 = k1 - math.sqrt(k1)
k2 = k2 - k2 ** (-2)
d1 = d1 * 1.99
d2 = d2 * 1.98

for i in range(changing[5], N):
    y = sampling.measure(ref=sinwave(dt=dt, i=i, w=0.5, A=2), noise_process=scale * 0.1, noise_measure=scale * 0.001)
    Y_sys.append(y)
    U.append(sampling.u)
    m1_all.append(m1)
    m2_all.append(m2)
    k1_all.append(k1)
    k2_all.append(k2)
    d1_all.append(d1)
    d2_all.append(d2)

Y_sys = normalize(Y_sys, 1)
U = normalize(U, 1)
Y_sys = np.asarray(Y_sys, dtype=np.float32)
U = np.asarray(U, dtype=np.float32)
U = U[:, np.newaxis]
dt = torch.tensor(dt, dtype=torch.float32)
lr = 0.0001
system = 'two_spring_motion5_8'
model_filename = f"{system}"
initial_filename = f"{system}_initial"
model = MechanicalSystem_i(dt=dt)  #
x_fit = torch.load(os.path.join("models", initial_filename))
checkpoint = torch.load(os.path.join("models", model_filename))
# model.eval()

optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': lr},
    {'params': [x_fit], 'lr': lr}
], lr=lr * 10)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # , strict=False

threshold1 = 1  #0.97  # start retrain, R2
threshold2 = 1  #0.98  # stop retrain
n = 2
ur = 64
t = n + n * ur + n
factor = PEM(n, t, N, ur=ur)
factor.P_old2 *= 0.9
factor.Psi_old2 *= 0.9
np.random.seed(3)
factor.Thehat_old = np.random.rand(t, 1) * 0.01
# print('seed=', np.random.get_state()[1][0])#
factor.Xhat_old = np.zeros((n, 1))
update = 1200
off = 0  #int(80/dt)
simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N, update=update, threshold1=threshold1,
                            threshold2=threshold2, train=off)


x0 = x_fit[[0], :].detach()

u = torch.tensor(U[:, None, :])
y = Y_sys[:, np.newaxis]

simulator0 = ForwardEuler_i(model=model, dt=dt)
start_time = time.time()
with torch.no_grad():
    xhat0 = simulator0(x0, u)
    xhat0 = xhat0.detach().numpy()
    xhat0 = xhat0.squeeze(1)
    yhat0 = xhat0[:, 0]

print(f"\n NN  time: {time.time() - start_time:.2f}")

start_time = time.time()
xhat_data = simulator(x0, u, y)
print(f"\nTrain time: {time.time() - start_time:.2f}")

yhat = xhat_data[:, 0]

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
ax[1].plot(time_exp[changing], Y_sys[changing], 'kx')
if off != 0:
    ax[1].plot(time_exp[off], Y_sys[off], 'bx')
ax[1].set_ylabel("(b)")
ax[1].legend(loc=4)
ax[2].plot(time_exp, U, 'k', label='u')
ax[2].set_ylabel("(c)")
ax[2].set_xlabel('time(s)')
ax[2].legend(loc=4)


