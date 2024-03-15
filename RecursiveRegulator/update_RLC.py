""""

"""
import matplotlib
matplotlib.use('TKAgg')
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import math
sys.path.append(os.path.join("F:/Project/head/"))
import header
from pem import PEM  # _step as PEM
from pem import normalize, R2
from header import NeuralStateSpaceModel, ForwardEulerPEM  # MechanicalSystem

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

df_data = pd.read_csv("F:/Project/DATA/RLC/RLC_data_test.csv")
Y_sys = np.array(df_data[['V_C']]).astype(np.float32)
time_exp = np.array(df_data['time']).astype(np.float32)
dt = time_exp[1] - time_exp[0]
U = np.array(df_data[['V_IN']]).astype(np.float32)
X = np.array(df_data[['V_C', 'I_L']]).astype(np.float32)
system = 'RLC'


N = len(Y_sys)
# Y_sys = normalize(Y_sys, 1)
# U = normalize(U, 1)

dt = torch.tensor(dt, dtype=torch.float32)  #
# ----------------------------
# ------------------ prepare data type ---------
# Y_sys = normalize(Y_sys, 1)
# U = normalize(U, 1)
Y_sys = np.asarray(Y_sys, dtype=np.float32)
U = np.asarray(U, dtype=np.float32)
U = U[:, np.newaxis]


lr = 0.001  # not used in PEM updateing

model_filename = f"{system}"
initial_filename = f"{system}_initial"
model = NeuralStateSpaceModel()  #
x_fit = torch.load(os.path.join("models", initial_filename))
checkpoint = torch.load(os.path.join("models", model_filename))
# model.eval()

optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': lr},
    {'params': [x_fit], 'lr': lr}
], lr=lr * 10)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # , strict=False


threshold1 = 0.96  # start retrain, R2
threshold2 = 0.98  # stop retrain
factor = PEM(2, 6, N)
factor.P_old2 *= 0.0009
factor.Psi_old2 *= 0.9
np.random.seed(3)
factor.Thehat_old = np.random.rand(6, 1) * 0.1
# print('seed=', np.random.get_state()[1][0])#
factor.Xhat_old = np.array([[2], [0]])

# simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N, optimizer=optimizer, update=1,threshold1=threshold1, threshold2=threshold2)
# simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N, optimizer=optimizer, update=0, threshold1=threshold1, threshold2=threshold2)
simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N, optimizer=optimizer, update=2, threshold1=threshold1, threshold2=threshold2)
# simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N, optimizer=optimizer, update=3, threshold1=threshold1, threshold2=threshold2)
# simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N, optimizer=optimizer, update=4, threshold1=threshold1, threshold2=threshold2)
# simulator = ForwardEulerPEM(model=model, factor=factor, dt=1.0, N=N, optimizer=optimizer, update=5, threshold1=threshold1, threshold2=threshold2)


# x_fit = np.zeros((1, n_x), dtype=np.float32)
# x_fit[0, 0] = np.copy(Y_sys[0, 0])
# x_fit[0, 1] = 0
# x_step = x0
# x0 = torch.tensor(x_fit[[0], :], dtype=torch.float32)

x0 = x_fit[[0], :].detach()

u = torch.tensor(U)  # [:, None, :]  , :
y = Y_sys[:, np.newaxis]
xhat_data = simulator(x0, u, y)

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
print("inference evolution R^2 = ", R2(Y_sys[:, 0], yhat))
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(time_exp, Y_sys, 'g', label='y')
ax[0].plot(time_exp, yhat, 'r--', label='$\hat{y}$')
# ax[0].plot(time_exp, simulator.y_pem, 'y--')
ax[0].legend()
ax[1].plot(time_exp, U[:, 0, 0], 'k', label='u')
ax[1].set_xlabel('Time(s)')
ax[1].legend()


#
fig, ax = plt.subplots(6, 1, sharex=True)
ax[0].plot(time_exp, Thehat[:, 0], 'g', label='a0')

ax[0].legend()
ax[1].plot(time_exp, Thehat[:, 1], 'g', label='a1')
ax[1].legend()
ax[2].plot(time_exp, Thehat[:, 2], 'b', label='b0')
ax[2].legend()
ax[3].plot(time_exp, Thehat[:, 3], 'b', label='b1')
ax[3].legend()
ax[4].plot(time_exp, Thehat[:, 4], 'k', label='k0')
ax[4].legend()
ax[5].plot(time_exp, Thehat[:, 5], 'k', label='k1')
ax[5].legend()
ax[5].set_xlabel('Time(s)')



plt.figure()
plt.plot(time_exp, simulator.y_pem, label='y_{pem}')
# plt.plot(time_exp[correction], simulator.y_pem[correction], 'yx')
# plt.plot(time_exp[stop], simulator.y_pem[stop], 'mx')
plt.xlabel('Time(s)')


plt.show()


