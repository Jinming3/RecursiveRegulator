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
from pem import PEM
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


# -- sin/cos wave ---
def sinwave(dt, time_all, w=0.5):
    out = []
    A = 1
    for k in range(int(time_all / dt)):
        x = A * np.cos(w * k * math.pi * dt)
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
model.eval()
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
                            threshold1=threshold1, threshold2=threshold2, train=train_time)  


start_time = time.time()
xhat_data = simulator(x0, u, Y_sys)
yhat = xhat_data[:, 0]
print(f"\nTrain time: {time.time() - start_time:.2f}")


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

print("inference test R^2 = ", R2(Y_sys[:, 0], yhat0))  #

print(f"inference PEM_update{update}_R^2 = ", R2(Y_sys[:, 0], yhat))  # [:, 0]


print("inference ARX R^2 = ", R2(Y_sys[p:N], yhat1))  #




fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(time_exp, Y_sys, 'g', label='y')
ax[0].plot(time_exp[p:N], yhat1, 'r--', label='$\hat{y}_{Ham}$')
ax[0].plot(time_exp[changing], Y_sys[changing], 'kx')
ax[0].plot(time_exp[train_time-1], Y_sys[train_time-1], 'bx')
ax[0].set_ylabel("(a)")
ax[0].legend()  # bbox_to_anchor=(1.141, 0.7)

ax[1].plot(time_exp[p:N], Y_sys[p:N], 'g', label='y')
ax[1].plot(time_exp, yhat, 'r--', label='$\hat{y}$')
ax[1].plot(time_exp[changing], Y_sys[changing], 'kx')
ax[1].plot(time_exp[train_time-1], Y_sys[train_time-1], 'bx')
ax[1].set_ylabel("(b)")
ax[1].legend()  #bbox_to_anchor=(1.141, 0.7)
ax[1].set_xlabel('time(s)')




