
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import os
import torch
import matplotlib.pylab as pylab
import sys
sys.path.append(os.path.join(".."))
import NN_header
import pem
from pem import PEM

params = {'figure.figsize': (4.5, 3.7),
          'legend.fontsize': 11,
          'legend.labelspacing': 0.05,
          'legend.loc': 'upper right',
          'axes.labelsize': 11,
          'axes.labelpad': 0.5,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11
          }
pylab.rcParams.update(params)

# # # -------load CED --------
# df = pd.read_csv(os.path.join("..", "data", "CED", "DATAPRBS.csv"))
# Y_sys = np.array(df[['z2']]).astype(np.float32)
# U = np.array(df[['u2']]).astype(np.float32)
# ts = 0.02
# time_exp = np.arange(Y_sys.size).astype(np.float32)*ts
# system = 'ced'
# # ---- load  WH
# df = pd.read_csv(os.path.join("..", "data", "WH", "WienerHammerBenchmark.csv"))
# U = np.array(df[['uBenchMark']][120000:127000]).astype(np.float32)
# Y_sys = np.array(df[['yBenchMark']][120000:127000]).astype(np.float32)
# fs = 51200
# time_exp = np.arange(Y_sys.size).astype(np.float32)*(1/fs)
# system = 'WH'
# ---- load data f16
# df = pd.read_csv(os.path.join("..", "data", "F16", "F16Data_SineSw_Level1.csv"))
# Y_sys = np.array(df[['Acceleration2']][60000:80000]).astype(np.float32)
# U = np.array(df[['Force']][60000:80000]).astype(np.float32)
# fs = 400
# time_exp = np.arange(Y_sys.size).astype(np.float32)*(1/fs)
# system = 'F16'
# # ---- load data RLC
# df = pd.read_csv(os.path.join("..", "data", "RLC", "RLC_data_test.csv"))
# U = np.array(df[['V_IN']]).astype(np.float32)
# Y_sys = np.array(df[['V_C']]).astype(np.float32)
# system = 'RLC'
# Y_sys = NN_header.normalize(Y_sys, 1)
# U = NN_header.normalize(U, 1)
# # ----cascaded tanks------
# df = pd.read_csv(os.path.join("..", "data", "CTS", "tankdata.csv"))
# U = np.array(df[['uVal']]).astype(np.float32)
# Y_sys = np.array(df[['yVal']]).astype(np.float32)
# system = 'CTS'
# # -------EMPS system--------
df_data = pd.read_csv(os.path.join("..", "data", "EMPS", "DATA_EMPS_PULSES_SC.csv"))
time_exp = np.array(df_data['time_exp']).astype(np.float32)
Y_sys = np.array(df_data[['q_meas']]).astype(np.float32)
U = np.array(df_data[['u_in']]).astype(np.float32)
system = 'EMPS'
# # -------EMPS ageing--------
# df_data = pd.read_csv(os.path.join("..", "data", "EMPS","data_generate_ageing.csv"))
# time_exp = np.array(df_data['time']).astype(np.float32)
# Y_sys = np.array(df_data[['Y_sys']]).astype(np.float32)
# U = np.array(df_data[['u']]).astype(np.float32)
# system = 'ageing'


N = len(Y_sys)

start_time = time.time()
# ------ set up PEM --------
n = 2  # dim of X
t = n + n + n  # canonical form, dim_y = 1
online = PEM(n, t, N)
# parameters initialization
online.P_old2 *= 0.9
online.Psi_old2 *= 0.9
np.random.seed(3)
online.Thehat_old = np.random.rand(t, 1)
online.Xhat_old = np.random.rand(n, 1)  # np.zeros((n, 1))*Y_sys[0]  #

# --------- set up NN --------
U_u = torch.from_numpy(U.astype(np.float32))
U_y = torch.from_numpy(Y_sys.reshape((N, 1)).astype(np.float32))
input_size = 2
output_size = 1
hidden = 60
batch = 25 # 10, 20, rlc CTS 30
model = NN_header.NNFilter(input_size, hidden, output_size)
num_epoch = 30  # 20
model_filename = f"{system}_batch{batch}_epo{num_epoch}"  # in training
model.load_state_dict(torch.load(os.path.join("models", model_filename)))
# ------ inference online --------
# out = model.forward1(U_u)  #, U_y
out = model.forward(U_u, U_y)


U_pem = out[:, 0].detach().numpy()

online.pemt(Y_sys, U_pem)

# -------- analysis----------
infer_time = time.time() - start_time
print(f"\ninference time: {infer_time}")
error_rectify = Y_sys[:, 0] - online.Yhat_data
r2 = pem.R2(Y_sys[:, 0], online.Yhat_data)
print("R^2 = ", r2)
print(f"error_rectify range [{np.amin(error_rectify)}, {np.amax(error_rectify)}] ")
mean_error_rectify = np.mean(error_rectify)
print(f"mean_error_rectify = {mean_error_rectify}")

# # # -------------------------------------------------------

# df1 = pd.DataFrame([batch], columns=['batch'])
# df1 = pd.concat([df1, pd.DataFrame([num_epoch], columns=['epoch'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame([r2], columns=['r2'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame([infer_time], columns=['infer_time'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame([mean_error_rectify], columns=['mean_error'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame(Y_sys, columns=['y'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame(online.Yhat_data, columns=['yfilter'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame(U, columns=['u'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame(U_pem, columns=['uf'])], axis=1)
#
# df1.to_csv(f'infer_{system}.csv', index=False)

# # -----------CED -----------
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(time_exp, U, 'k', label='input')
# ax[0].set_ylabel("Voltage (V)")
# ax[0].legend()
# ax[0].grid(True)
# ax[1].plot(time_exp, Y_sys, 'g', label='measurement')
# ax[1].plot(time_exp, online.Yhat_data, 'r--', label='NN-filter')
# plt.ylabel("Velocity (m/s)")
# ax[1].legend()
# ax[1].grid(True)
# ax[1].set_xlabel("Time (s)")
# #---------- WH ---------
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(time_exp, U, 'k', label='input')
# ax[0].set_ylabel("Voltage (V)")
# ax[0].legend()
# ax[0].grid(True)
# ax[1].plot(time_exp, Y_sys, 'g', label='measurement')
# ax[1].plot(time_exp, online.Yhat_data, 'r--', label='NN-filter')
# plt.ylabel("Voltage (V)")
# ax[1].legend()
# ax[1].grid(True)
# ax[1].set_xlabel("Time (s)")
# -------- F16 ---------
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(time_exp, U, 'k', label='input')
# ax[0].set_ylabel("Force (N)")
# ax[0].legend()
# ax[0].grid(True)
# ax[1].plot(time_exp, Y_sys, 'g', label='measurement')
# ax[1].plot(time_exp, online.Yhat_data, 'r--', label='NN-filter')
# plt.ylabel("Acceleration ($m/s^2$)")
# ax[1].legend()
# ax[1].grid(True)
# ax[1].set_xlabel("Time (s)")
# # -----------EMPS -----------
# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(time_exp, Y_sys, 'g', label='measurement')
# ax[0].plot(time_exp, online.Yhat_data, 'r--', label='inference')
# # plt.ylabel("Position (m)")
# ax[0].legend()
# ax[0].grid(True)
# ax[1].plot(time_exp, U, 'k', label='u')
# # ax[0].set_ylabel("Voltage (V)")
# ax[1].legend()
# ax[1].grid(True)
# ax[2].plot(time_exp, U_pem, 'k', label="u'")
# ax[2].legend()
# ax[2].grid(True)
# ax[2].set_xlabel("Time (s)")
#
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(time_exp, U, 'k', label='u')
# ax[0].grid(True)
# ax[0].legend()
# ax[1].plot(time_exp, U_pem, 'k', label="$u'$")
# ax[1].set_xlabel('Time (s)')
# ax[1].grid(True)
# ax[1].legend()

# # --- plot any system -------
# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(Y_sys, 'g', label='measurement')
# ax[0].plot(online.Yhat_data, 'r--', label='inference')
# ax[0].legend()
# ax[0].grid(True)
# ax[1].plot(U, 'k', label='u')
# ax[1].legend()
# ax[1].grid(True)
# ax[2].plot(U_pem, 'k', label="u'")
# ax[2].legend()
# ax[2].grid(True)
# ax[2].set_xlabel("Time")
#
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(U, 'k', label='u')
# ax[0].grid(True)
# ax[0].legend()
# ax[1].plot(U_pem, 'k', label="$u'$")
# ax[1].set_xlabel('Time')
# ax[1].grid(True)
# ax[1].legend()
plt.show()
