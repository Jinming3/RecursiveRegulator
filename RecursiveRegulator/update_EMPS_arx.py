""""

"""
import matplotlib
matplotlib.use("TkAgg")
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
from pem import PEM, PEM_mimo  # _step as PEM
from pem import normalize, R2
from header import MechanicalSystem, ForwardEulerPEM, ForwardEuler
import pandas as pd
import statsmodels.api as sm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# import matplotlib.pylab as pylab
#
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

# def simple(params, Y, U, dt=1, train_time=1, pre_ahead=True):
#
#     N = len(Y)
#     yhat_data = []
#     yhat_step = []
#     if train_time == 1:
#         for i in range(0, N):
#             yhat = Y[i - 1]
#             yhat_data.append(yhat)
#     if train_time > 1:
#             for i in range(train_time):
#                 yhat = Y[i - 1]
#                 yhat_data.append(yhat)
#             for i in range(train_time-1, N-1):
#                 yhat_new = yhat
#                 yhat_data.append(yhat_new)
#
#
#
#     return yhat_data



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


system = 'update'
dt = 0.005
time_all = np.array([70])
changing = np.array([20, 50]) / dt

# system = 'update3'
# dt = 0.3
# time_all = np.array([100])  #
# change1 = int(time_all/10*3)
# change2 = int(time_all/10*6)
# changing = np.array([change1, change2])/dt
# train_time = int(40/ dt)

N = int(time_all[-1] / dt)
time_exp = np.arange(N) * dt

train_time = int(25/ dt)

# train_time=N
# update = 5  # original update,  yhat_pem add to x_step, use this

update = 8 # stop at self.train, pem in s_step


changing = changing.astype(int)

p_ref = sinwave(dt, time_all)  # ref signal not change
sig = 'sinwave'
p_tri = triangle(dt, time_all)

# --- changing -------------------
# # # # ---- start original system with no change---
# sampling = EMPS(dt, pos=0, vel=0, acc=0, u=0)
# for i in range(changing[0]):
#     y = sampling.measure(p_ref[i])
#     Y_sys.append(y)
#     U.append(sampling.u)
#     M_all.append(M)
#     Fv_all.append(Fv)
#     Fc_all.append(Fc)

# -----------------------------------------
# # #  --- system changing ---
# M = M * 0.7
# Fv = Fv * 0.7
# Fc = Fc * 0.8
# for i in range(changing[0], changing[1]):
#     y = sampling.measure(p_ref[i], noise_process=10 ** -3, noise_measure=10 ** -4)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     M_all.append(M)
#     Fv_all.append(Fv)
#     Fc_all.append(Fc)
#
# offset = offset * 0.7
# M = M * 0.5
# Fv = Fv * 0.6
# Fc = Fc * 0.75
# for i in range(changing[1], changing[2]):
#     y = sampling.measure(p_ref[i], noise_process=10 ** -2, noise_measure=10 ** -4)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     M_all.append(M)
#     Fv_all.append(Fv)
#     Fc_all.append(Fc)
#
# M = M * 0.97
# Fv = Fv * 0.9
# Fc = Fc * 0.9
# for i in range(changing[2], N):
#     y = sampling.measure(p_ref[i], noise_process=10 ** -3, noise_measure=10 ** -4)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     M_all.append(M)
#     Fv_all.append(Fv)
#     Fc_all.append(Fc)
# -----------------------------------
# # # --------------- only one sudden change -------------------
# sampling = EMPS(dt, pos=0, vel=0, acc=0, u=0)
# for i in range(changing[0]):
#     y = sampling.measure(p_ref[i], noise_process=10 ** -4, noise_measure=10 ** -4)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     M_all.append(M)
#     Fv_all.append(Fv)
#     Fc_all.append(Fc)
# offset = offset * 0.99
# M = M * 0.9
# Fv = Fv * 0.9
# Fc = Fc * 0.9
# for i in range(changing[0], N):
#     y = sampling.measure(p_ref[i], noise_process=10 ** -4, noise_measure=10 ** -4)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     M_all.append(M)
#     Fv_all.append(Fv)
#     Fc_all.append(Fc)

# # # # # --------------- sudden change in params and in ref -------use this in the paper------------
Y_sys = []
U = []
M_all = []
Fc_all = []
Fv_all = []
ref_signal = []
aging_factor=0.85
scale = 10**-3
sampling = EMPS(dt, pos=0, vel=0, acc=0, u=0)
for i in range(changing[0]):
    y = sampling.measure(p_ref[i], noise_process=scale, noise_measure=scale*0.1)
    Y_sys.append(y)
    U.append(sampling.u)
    M_all.append(M)
    Fv_all.append(Fv)
    Fc_all.append(Fc)
    ref_signal.append(p_ref[i])
offset = offset * 0.99
M = M * aging_factor
Fv = Fv * aging_factor
Fc = Fc * aging_factor
for i in range(changing[0], changing[1]):
    y = sampling.measure(p_ref[i], noise_process=scale, noise_measure=scale*0.1)
    Y_sys.append(y)
    U.append(sampling.u)
    M_all.append(M)
    Fv_all.append(Fv)
    Fc_all.append(Fc)
    ref_signal.append(p_ref[i])
# offset = offset * 0.99
# M = M * aging_factor
# Fv = Fv * aging_factor
# Fc = Fc * aging_factor
for i in range(changing[1], N):
    y = sampling.measure(p_tri[i], noise_process=scale, noise_measure=scale*0.1)
    # y = sampling.measure(p_ref[i], noise_process=10 ** -3, noise_measure=10 ** -4)
    Y_sys.append(y)
    U.append(sampling.u)
    M_all.append(M)
    Fv_all.append(Fv)
    Fc_all.append(Fc)
    # ref_signal.append(p_ref[i])
    ref_signal.append(p_tri[i])

# sampling = EMPS(dt, pos=0, vel=0, acc=0, u=0)
# def sample_aging(aging_factor):
#
#     aging_factor = aging_factor
#     for i in range(changing[0]):
#         y = sampling.measure(p_ref[i], noise_process=10 ** -4, noise_measure=10 ** -5)
#         Y_sys.append(y)
#         U.append(sampling.u)
#         M_all.append(M)
#         Fv_all.append(Fv)
#         Fc_all.append(Fc)
#         ref_signal.append(p_ref[i])
#
#     offset = offset * 0.99
#     M = M * aging_factor
#     Fv = Fv * aging_factor
#     Fc = Fc * aging_factor
#     for i in range(changing[0], changing[1]):
#         y = sampling.measure(p_ref[i], noise_process=10 ** -3, noise_measure=10 ** -4)
#         Y_sys.append(y)
#         U.append(sampling.u)
#         M_all.append(M)
#         Fv_all.append(Fv)
#         Fc_all.append(Fc)
#         ref_signal.append(p_ref[i])
#     offset = offset * 0.99
#     M = M * aging_factor
#     Fv = Fv * aging_factor
#     Fc = Fc * aging_factor
#     for i in range(changing[1], N):
#         y = sampling.measure(p_tri[i], noise_process=10 ** -3, noise_measure=10 ** -4)
#         # y = sampling.measure(p_ref[i], noise_process=10 ** -3, noise_measure=10 ** -4)
#         Y_sys.append(y)
#         U.append(sampling.u)
#         M_all.append(M)
#         Fv_all.append(Fv)
#         Fc_all.append(Fc)
#         # ref_signal.append(p_ref[i])
#         ref_signal.append(p_tri[i])
#
#     return Y_sys, U
# Y_sys, U = sample_aging(0.9)
# # --------------------------------------
# # # --- continuously degenerating ----
#
# sampling = EMPS(dt, pos=0, vel=0, acc=0, u=0)
# for i in range(changing[0]):
#     y = sampling.measure(p_ref[i])
#     Y_sys.append(y)
#     U.append(sampling.u)
#
# offset = offset * 0.97
# for i in range(changing[0], N):
#     if i % 50 == 0:
#         M = M - 0.01 * i * dt
#         Fv = Fv - 0.0001 * i * dt
#         Fc = Fc - 0.0001 * i * dt
#     y = sampling.measure(p_ref[i], noise_process=10 ** -2, noise_measure=10 ** -4)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     M_all.append(M)
#     Fv_all.append(Fv)
#     Fc_all.append(Fc)

# ----------------------------
# ------------------ prepare data type ---------
Y_sys = normalize(Y_sys, 1)
U = normalize(U, 1)
Y_sys = np.asarray(Y_sys, dtype=np.float32)
U = np.asarray(U, dtype=np.float32)
U = U[:, np.newaxis]
dt = torch.tensor(dt, dtype=torch.float32)
u = torch.tensor(U[:, None, :])  # [:, None, :]
Y_sys = Y_sys[:, np.newaxis]

lr = 0.0001  # not used in PEM updateing

model_filename = f"{system}"
initial_filename = f"{system}_initial"
model = MechanicalSystem(dt=dt)  #
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


simulator0 = ForwardEuler(model=model, dt=dt)
start_time = time.time()
with torch.no_grad():
    xhat0 = simulator0(x0, u)
    xhat0 = xhat0.detach().numpy()
    xhat0 = xhat0.squeeze(1)
    yhat0 = xhat0[:, 0]
    # yhat0=yhat0[:, None]
print(f"\n NN  time: {time.time() - start_time:.2f}")
# # -------
#
# data = pd.DataFrame({'y': Y_sys[0:train_time, 0], 'x':yhat0[0:train_time]})
# p = 2
# q = 1
# for i in range(1, p + 1):
#     data[f'y_lag_{i}'] = data['y'].shift(i)
# for j in range(1, q + 1):
#     data[f'x_lag_{j}'] = data['x'].shift(j)
# data = data.dropna()
# endog = data['y']
# exog = data[[ 'y_lag_2','y_lag_1','x_lag_1']]  #
# # Fit the ARX model
# linearParams = sm.OLS(endog, sm.add_constant(exog))  # OLS stands for Ordinary Least Squares
# results = linearParams.fit()
# yhat1_1=results.predict(sm.add_constant(exog))
#
# data2 = pd.DataFrame({'y2': Y_sys[train_time-p:N, 0], 'x2':yhat0[train_time-p:N]})
# for i in range(1, p + 1):
#     data2[f'y_lag_{i}'] = data2['y2'].shift(i)
#
# for j in range(1, q + 1):
#     data2[f'x_lag_{j}'] = data2['x2'].shift(j)
# data2 = data2.dropna()
# exog2 = data2[[ 'y_lag_2','y_lag_1','x_lag_1']]
# yhat1_2=results.predict(sm.add_constant(exog2))
# yhat1=np.concatenate((yhat1_1, yhat1_2), dtype=np.float32)
# # -------

# def arx_train(y, X, p, q):
#     """
#     Train an ARX model.
#
#     Parameters:
#     - y: 1D array, endogenous variable (dependent variable)
#     - X: 2D array, exogenous variables (independent variables)
#     - p: int, autoregressive order
#     - q: int, exogenous order
#
#     Returns:
#     - Coefficients: Tuple containing autoregressive and exogenous coefficients
#     """
#     n_obs = len(y)
#
#     # Create lagged variables for the autoregressive terms
#     lagged_y = np.zeros((n_obs, p))
#     for i in range(p):
#         lagged_y[i + 1:, i] = y[:-i - 1]
#
#     # Create lagged variables for the exogenous terms
#     lagged_X = np.zeros((n_obs, q))
#     for i in range(q):
#         lagged_X[i + 1:, i] = X[:-i - 1]
#
#     # Combine lagged variables for both autoregressive and exogenous terms
#     lagged_variables = np.hstack((lagged_y, lagged_X))
#
#     # Add a constant term for the intercept
#     lagged_variables = np.column_stack((np.ones(n_obs), lagged_variables))
#
#     # Perform Ordinary Least Squares (OLS) to estimate coefficients
#     coefficients = np.linalg.lstsq(lagged_variables, y, rcond=None)[0]
#
#     return coefficients

p = 2
yhat1 = []
y_lag_1 = 0
y_lag_2 = 0
for i in range(2, N):
    endog = Y_sys[i]
    # x = U[i]
    x = yhat0[i]
    exog = np.column_stack((y_lag_1, y_lag_2, x))
    if i < train_time:
        LinearParams = sm.OLS(endog, sm.add_constant(exog))
        results = LinearParams.fit()
    pred = results.predict(sm.add_constant(exog))

    yhat1.append(pred)
    y_lag_1 = pred
    y_lag_2 = y_lag_1
# yhat1=np.array(yhat1)


#
#
threshold1 = 0.91  # start retrain, R2
threshold2 = 0.95  # stop retrain
# threshold1 = 0.96  # original
# threshold2 = 0.98 # stop retrain
factor = PEM(2, 6, N)
factor.P_old2 *= 0.09  # 0.009#0.09
factor.Psi_old2 *= 0.9
np.random.seed(3)
factor.Thehat_old = np.random.rand(6, 1) * 0.01
# print('seed=', np.random.get_state()[1][0])#
factor.Xhat_old = np.array([[2], [0]])

simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N, update=update,
                            threshold1=threshold1, threshold2=threshold2, train=train_time)  # optimizer=optimizer,

# x_fit = np.zeros((1, n_x), dtype=np.float32)
# x_fit[0, 0] = np.copy(Y_sys[0, 0])
# x_fit[0, 1] = 0
# x_step = x0
# x0 = torch.tensor(x_fit[[0], :], dtype=torch.float32)


start_time = time.time()
xhat_data = simulator(x0, u, Y_sys)
yhat = xhat_data[:, 0]
print(f"\nTrain time: {time.time() - start_time:.2f}")
# Thehat = simulator.Thehat

# # --------- plot aging_rate -----
#
# rate_evl = [0.99,0.97,0.95,0.93, 0.9,0.87,0.85, 0.83,0.8,0.75, 0.7, 0.65,0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# r2_evl = []
# for r in rate_evl:
#     aging_factor = r
#     gt, kp, kv = 35.15, 160.18, 243.45
#     M, Fv = 95.1089, 203.5034
#     Fc, offset = 20.3935, -3.1648
#     Y_sys = []
#     U = []
#     sampling = EMPS(dt, pos=0, vel=0, acc=0, u=0)
#     for i in range(changing[0]):
#         y = sampling.measure(p_ref[i], noise_process=10 ** -4, noise_measure=10 ** -5)
#         Y_sys.append(y)
#         U.append(sampling.u)
#     offset = offset * 0.99
#     M = M * aging_factor
#     Fv = Fv * aging_factor
#     Fc = Fc * aging_factor
#     sampling = EMPS(dt, pos=0, vel=0, acc=0, u=0)
#
#     for i in range(changing[0], changing[1]):
#         y = sampling.measure(p_ref[i], noise_process=10 ** -3, noise_measure=10 ** -4)
#         Y_sys.append(y)
#         U.append(sampling.u)
#     offset = offset * 0.99
#     M = M * aging_factor
#     Fv = Fv * aging_factor
#     Fc = Fc * aging_factor
#     sampling = EMPS(dt, pos=0, vel=0, acc=0, u=0)
#
#     for i in range(changing[1], N):
#         y = sampling.measure(p_ref[i], noise_process=10 ** -3, noise_measure=10 ** -4)
#         Y_sys.append(y)
#         U.append(sampling.u)
#
#
#     Y_sys = normalize(Y_sys, 1)
#     U = normalize(U, 1)
#     Y_sys = np.asarray(Y_sys, dtype=np.float32)
#     U = np.asarray(U, dtype=np.float32)
#     U = U[:, np.newaxis]
#     dt = torch.tensor(dt, dtype=torch.float32)
#     u = torch.tensor(U[:, None, :])  # [:, None, :]
#     Y_sys = Y_sys[:, np.newaxis]
#     xhat_data = simulator(x0, u, Y_sys)
#     yhat = xhat_data[:, 0]
#     r2_evl.append(R2(Y_sys[:, 0], yhat))
#     print([r, R2(Y_sys[:, 0], yhat)])
#
# r2_evl=np.array(r2_evl)
# np.savetxt(f"aging_rate_{'%.4f' %dt}.txt", [rate_evl, r2_evl])
# rate_evl_plot = [1-p for p in rate_evl]
# fig, ax = plt.subplots(1, 1, sharex=True)
# ax.plot(rate_evl_plot,r2_evl, 'k', label='$R^2$')
# ax.legend()
# ax.set_xlabel('aging rate')
# ax.grid()


# ---------------------------


ts=dt.detach().numpy()
if update==5:
    stop_t = np.array(simulator.stop)[:, 1]*ts
    correction_t =np.array(simulator.correction)[:, 1]*ts
    stop = simulator.stop
    correction =simulator.correction
    print(f'update at {correction}_{correction_t}')
    print(f'stop at {stop}_{stop_t}')
    work = np.sum(np.array(simulator.stop)[:, 1]-np.array(simulator.correction)[:, 1])/N
#
# # # ->>>---- update == False, optimization outside NN loop, soo faster than stepwise --
# # yhat_stable = xhat_data[:, 0]
# # factor.forward(y, yhat_stable)
# # yhat = factor.Yhat_data
# # Thehat = factor.Thehat_data
# # ------ <<<--------------------------------------
# yhat_simple= simple(params=0, Y=Y_sys, U=U, dt=ts, train_time=train_time)

print("inference test R^2 = ", R2(Y_sys[:, 0], yhat0))  #

print(f"inference PEM_update{update}_R^2 = ", R2(Y_sys[:, 0], yhat))  # [:, 0]


print("inference ARX R^2 = ", R2(Y_sys[p:N], yhat1))  #



simulator.y_pem = np.array(simulator.y_pem)
simulator.y_pem0 = np.array(simulator.y_pem0)

# plt.plot(time_exp, simulator.y_pem, 'r', label='$\hat{y}_{pem}$')
# plt.plot(time_exp, simulator.y_pem0, 'g', label='$\hat{y}_{pem0}$')

# plt.plot(simulator.y_pem[:, 0], 'r-', label='$\hat{y}_{pem}$')
# plt.plot(simulator.y_pem0[:, 0], 'g-', label='$\hat{y}_{pem}0$')
# plt.plot(time_exp[correction], simulator.y_pem[correction], 'yx')
# plt.plot(time_exp[stop], simulator.y_pem[stop], 'mx')
plt.xlabel('time(s)')
plt.legend()


# print("inference simple R^2 = ", R2(Y_sys[:, 0], yhat_s[:, 0]))  #
# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(time_exp, Y_sys, 'g', label='y')
# ax[0].plot(time_exp, yhat0, 'r--', label='$\hat{y}_{NN}$')
# ax[0].plot(time_exp[changing], Y_sys[changing], 'kx')
# ax[0].set_ylabel("(a)")
# ax[0].legend(bbox_to_anchor=(0.9, 0.7))  #
# ax[1].plot(time_exp, Y_sys, 'g', label='y')
# ax[1].plot(time_exp, yhat, 'r--', label='$\hat{y}$')
# ax[1].plot(time_exp[changing], Y_sys[changing], 'kx')
# ax[1].set_ylabel("(b)")
# ax[1].legend(bbox_to_anchor=(0.9, 0.6))  #
# ax[2].plot(simulator.y_pem[:, 1]* ts , simulator.y_pem[:, 0], 'r', label=r"$ss_{update}$")  #
# ax[2].plot(simulator.y_pem0[:, 1]* ts , simulator.y_pem0[:, 0], 'g', label=r"$ss$")
# # ax[2].plot(time_exp, U, 'k', label='u')
# plt.ticklabel_format(axis='y', style='sci', scilimits=(1, 3))
# ax[2].set_ylabel("(c)")
# ax[2].legend(loc='upper left')  # bbox_to_anchor=(1.11, 0.8), fontsize=13
# ax[2].set_xlabel('time(s)')
# ax[3].plot(time_exp, ref_signal, 'k', label='ref')
# ax[3].set_ylabel("(d)")
# ax[3].legend()  # bbox_to_anchor=(1.11, 0.8)
# ax[3].set_xlabel('time(s)')
# yhat0 = np.loadtxt('yhat0_emps.txt')
# yhat0 = np.loadtxt('yhat05.txt')

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(time_exp, Y_sys, 'g', label='y')
# ax[0].plot(time_exp, yhat0, 'r--', label='$\hat{y}_{NN}$')
ax[0].plot(time_exp[p:N], yhat1, 'r--', label='$\hat{y}_{Ham}$')
ax[0].plot(time_exp[changing], Y_sys[changing], 'kx')
ax[0].plot(time_exp[train_time-1], Y_sys[train_time-1], 'bx')
ax[0].set_ylabel("(a)")
ax[0].legend()  # bbox_to_anchor=(1.141, 0.7)

ax[1].plot(time_exp[p:N], Y_sys[p:N], 'g', label='y')
# ax[1].plot(time_exp[p:N], yhat1, 'r--', label='$\hat{y}_{Ham}$')
ax[1].plot(time_exp, yhat, 'r--', label='$\hat{y}$')
ax[1].plot(time_exp[changing], Y_sys[changing], 'kx')
ax[1].plot(time_exp[train_time-1], Y_sys[train_time-1], 'bx')
ax[1].set_ylabel("(b)")
ax[1].legend()  #bbox_to_anchor=(1.141, 0.7)
ax[1].set_xlabel('time(s)')
# ax[2].plot(time_exp, Y_sys, 'g', label='y')
# ax[2].plot(time_exp, yhat, 'r--', label='$\hat{y}$')
# ax[2].plot(time_exp[changing], Y_sys[changing], 'kx')
# ax[2].plot(time_exp[train_time-1], Y_sys[train_time-1], 'bx')
# ax[2].set_ylabel("(c)")
# ax[2].legend()  # bbox_to_anchor=(1.11, 0.8)
# ax[2].set_xlabel('time(s)')

#   # compare to y=yk-1
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(time_exp, Y_sys, 'g', label='y')
# ax[0].plot(time_exp, yhat, 'r--', label='$yhat\_regulator$')
# ax[0].plot(time_exp[changing], Y_sys[changing], 'kx')
# ax[0].plot(time_exp[train_time-1], Y_sys[train_time-1], 'bx')
# # ax[0].set_ylabel("(a)")
# ax[0].legend(loc='lower left')  # bbox_to_anchor=(1.141, 0.7)
#
# ax[1].plot(time_exp, Y_sys, 'g', label='y')
#
# ax[1].plot(time_exp, yhat_simple, 'r--', label='$yhat\_compare$')
#
# ax[1].plot(time_exp[changing], Y_sys[changing], 'kx')
# ax[1].plot(time_exp[train_time-1], Y_sys[train_time-1], 'bx')
#
# # ax[1].set_ylabel("(b)")
# ax[1].legend(loc='lower left')  # bbox_to_anchor=(1.11, 0.8)
# ax[0].set_xlabel('Time')


# ax[2].plot(time_exp, Y_sys, 'g', label='y')
# ax[2].plot(time_exp, yhat_s, 'r--', label='s')
# # ax[2].plot(time_exp, U, 'k', label='u')
# # ax[1].plot(time_exp[changing], U[changing], 'kx', label='changing')
# ax[2].plot(time_exp[train_time-1], Y_sys[train_time-1], 'bx')
#
# ax[2].set_ylabel("(c)")
# ax[2].legend()
# ax[3].plot(time_exp, ref_signal, 'k', label='ref')  #
# ax[3].legend()
# ax[3].set_ylabel("(d)")
# ax[3].set_xlabel('time(s)')
#
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
# # ----------- degenerating physical parameters --------
# # fig, ax = plt.subplots(3, 1, sharex=True)
# # ax[0].plot(M_all, 'g', label='M')
# # ax[0].legend()
# # ax[1].plot(Fc_all, 'k', label='Fc')
# # ax[1].legend()
# # ax[2].plot(Fv_all, 'k', label='Fv')
# # ax[2].legend()
#
# simulator.y_pem = np.array(simulator.y_pem)
# simulator.y_pem0 = np.array(simulator.y_pem0)
# plt.figure()
# # plt.plot(time_exp, simulator.y_pem, 'r', label='$\hat{y}_{pem}$')
# # plt.plot(time_exp, simulator.y_pem0, 'g', label='$\hat{y}_{pem0}$')
# plt.plot(simulator.y_pem[:, 1]* ts , simulator.y_pem[:, 0], 'r', label=r"$\bar{y}_{pem}(update)$")  #
# plt.plot(simulator.y_pem0[:, 1]* ts , simulator.y_pem0[:, 0], 'g', label=r"$\bar{y}_{pem}(disable)$")
# # plt.plot(simulator.y_pem[:, 0], 'r-', label='$\hat{y}_{pem}$')
# # plt.plot(simulator.y_pem0[:, 0], 'g-', label='$\hat{y}_{pem}0$')
# # plt.plot(time_exp[correction], simulator.y_pem[correction], 'yx')
# # plt.plot(time_exp[stop], simulator.y_pem[stop], 'mx')
# plt.xlabel('time(s)')
# plt.legend()
#
# # simulator.correction = np.array(simulator.correction)
# # simulator.stop = np.array(simulator.stop)
# # plt.figure()
# # plt.plot(simulator.correction[:, 1], simulator.correction[:, 0], 'r', label='update')  # time_exp,
# # plt.plot(simulator.stop[:, 1], simulator.stop[:, 0], 'b', label='stop')
# # plt.xlabel('time(s)')
# # plt.legend()
#
# plt.figure()
# plt.plot(simulator.r2, 'r', label='$R^2$')  # time_exp,
# plt.xlabel('time(s)')
# plt.legend()
# #
# # plt.figure()
# # plt.plot(np.abs(simulator.err), 'r', label='$error$')  # time_exp,
# # plt.xlabel('time(s)')
# # plt.legend()
# # plt.show()
