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

# -------read EMPS signals --------
df = pd.read_csv(os.path.join("..", "data", "EMPS", "data_generate_ageing.csv"))
time_exp = np.array(df['time']).astype(np.float32)
Y_sys = np.array(df['Y_sys']).astype(np.float32)
U = np.array(df[['u']]).astype(np.float32)

N = len(Y_sys)
start_time = time.time()
# ------ set up PEM --------
n = 2  # dim of X
t = n + n + n  # canonical form, dim_y = 1
online = PEM(n, t, N)
# parameters initialization
online.P_old2[:, :] = 0.9
online.Psi_old2[:, :] = 0.9
np.random.seed(3)
online.Thehat_old = np.random.rand(t, 1)
online.Xhat_old = np.random.rand(n, 1)  # np.zeros((n, 1))*Y_sys[0]  #

# --------- set up NN --------
U_u = torch.from_numpy(U.astype(np.float32))
U_y = torch.from_numpy(Y_sys.reshape((N, 1)).astype(np.float32))
input_size = 2
output_size = 1
hidden = 60
model = NN_header.NNFilter(input_size, hidden, output_size)
model_filename = f"ageing_batch200"
# model_filename = f"nn_filter_whole"
model.load_state_dict(torch.load(os.path.join("models", model_filename)))
# ------ inference online --------
out = model.forward(U_u, U_y)  #
U_pem = out[:, 0].detach().numpy()

online.pemt(Y_sys, U_pem)
infer_time = time.time() - start_time
print(f"\ninference time: {infer_time}")
error_rectify = Y_sys - online.Yhat_data
print("R^2 = ", pem.R2(Y_sys, online.Yhat_data))
print(f"error_rectify range [{np.amin(error_rectify)}, {np.amax(error_rectify)}] ")
print(f"mean_error_rectify = {np.mean(error_rectify)}")
# # -------------------------------------------------------
# df1 = pd.concat([df, pd.DataFrame(online.Yhat_data, columns=['Yhat_pem'])], axis=1)
# # df1 = pd.concat([df1, pd.DataFrame(error_rectify, columns=['error_out'])], axis=1)
# df1.to_csv('ageing_batch_infer.csv', index=False)


fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(time_exp, Y_sys, 'g', label='measurement')
ax[0].plot(time_exp, online.Yhat_data, 'r--', label='NN-filter')
ax[0].set_ylabel('Position (m)')
ax[0].legend()
ax[0].grid(True)
ax[1].plot(time_exp, U, 'k', label='input')
ax[1].set_ylabel('Voltage (V)')
ax[1].set_xlabel("Time (s)")
ax[1].legend()
ax[1].grid(True)

# plt.figure(2)
# plt.plot(time_exp, error_rectify, 'k', label='prediction error')
# plt.xlabel('Time (s)')
# plt.legend()

plt.show()
