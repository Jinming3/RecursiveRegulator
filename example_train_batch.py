"""
use batches, DataLoader
in u, y, out u'
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch
import torch.nn as nn
import os
import time
import matplotlib.pylab as pylab
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append(os.path.join(".."))
import pem
from pem import PEM
import NN_header
params = {'figure.figsize': (4.5, 3.7),
          'legend.fontsize': 11,
          'legend.labelspacing': 0.05,
          'axes.labelsize': 11,
          'axes.labelpad': 0.5,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11
          }
pylab.rcParams.update(params)

# # # -------read CED signals --------
# df = pd.read_csv(os.path.join("..", "data", "CED", "DATAPRBS.csv"))
# Y_sys = np.array(df[['z1']]).astype(np.float32)
# U = np.array(df[['u1']]).astype(np.float32)
# system = 'ced'

# # ---- load data WH
# df = pd.read_csv(os.path.join("..", "data", "WH", "WienerHammerBenchmark.csv"))
# U = np.array(df[['uBenchMark']][6000:10000]).astype(np.float32)
# Y_sys = np.array(df[['yBenchMark']][6000:10000]).astype(np.float32)
# system = 'WH'
# ---- load data F16
df = pd.read_csv(os.path.join("..", "data", "F16", "F16Data_SineSw_Level1.csv"))
Y_sys = np.array(df[['Acceleration2']][40000:60000]).astype(np.float32)
U = np.array(df[['Force']][40000:60000]).astype(np.float32)
system = 'F16'
#
# # ----------RLC ---------------
# df = pd.read_csv(os.path.join("..", "data", "RLC", "RLC_data_id.csv"))
# U = np.array(df[['V_IN']]).astype(np.float32)
# Y_sys = np.array(df[['V_C']]).astype(np.float32)
# system = 'RLC'
# Y_sys = NN_header.normalize(Y_sys, 1)
# U = NN_header.normalize(U, 1)
# # ----cascaded tanks------
# df = pd.read_csv(os.path.join("..", "data", "CTS", "tankdata.csv"))
# U = np.array(df[['uEst']]).astype(np.float32)
# Y_sys = np.array(df[['yEst']]).astype(np.float32)
# system = 'CTS'
# # -------EMPS system--------
# df_data = pd.read_csv(os.path.join("..", "data", "EMPS","DATA_EMPS_SC.csv"))
# time_exp = np.array(df_data['time_exp']).astype(np.float32)
# Y_sys = np.array(df_data[['q_meas']]).astype(np.float32)
# U = np.array(df_data[['u_in']]).astype(np.float32)
# system = 'EMPS'


N = len(Y_sys)
start_time = time.time()
# ------- set up PEM -----
n = 2  # dim of X
t = n + n + n  # canonical form, dim_y = 1
online = PEM(n, t, N)  # NN_header.
# parameters initialization
online.P_old2[:, :] = 0.9
online.Psi_old2[:, :] = 0.9
np.random.seed(3)   # CTS 1554419047, RLC 3
online.Thehat_old = np.random.rand(t, 1)
online.Xhat_old = np.random.rand(n, 1)


# --------- set up NN dataset--------
class Data(Dataset):
    def __init__(self):
        self.U_u = torch.from_numpy(U.astype(np.float32))
        self.Y_nn = torch.from_numpy(Y_sys.astype(np.float32))
        self.n_samples = N

    def __getitem__(self, index):
        return self.U_u[index], self.Y_nn[index]

    def __len__(self):
        return self.n_samples


# ----- batches ----
batch_size = 200  # sequence length in each batch, 200: WH, EMPS;  100: CTS,  CED, RLC,
dataset = Data()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0)
dataiter = iter(train_loader)
data = dataiter.next()
U_u, Y_nn = data
# define NN model
num_epoch = 50
input_size = 2
output_size = 1
hidden = 60
model = NN_header.NNFilter(input_size, hidden, output_size)
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
LOSS = []  # collect MSE
num_iter = math.ceil(N/batch_size)  # number of iter in each epoch

# --------- training loop --------
for epoch in range(num_epoch):
    for i, (U_u, Y_nn) in enumerate(train_loader):
        out = model.forward(U_u, Y_nn)  # prediction = forward pass
        U_pem = out[:, 0].detach().numpy()
        Y = Y_nn[:, 0].detach().numpy()
        online.pemt(Y, U_pem)
        yhat = online.Yhat_data[:, np.newaxis]
        yhat = torch.from_numpy(yhat.astype(np.float32))
        loss = NN_header.mse(Y_nn, yhat)
        LOSS.append(loss.item())
        error = NN_header.mse(out, loss)
        error.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1) % 5 == 0:  # unpack before print
            print(f'epoch {epoch+1}/{num_epoch}, step {i+1}/{num_iter}: loss= {loss.item():.5f}')

train_time = time.time() - start_time
print(f"\nTrain time: {train_time}")
# Save model
if not os.path.exists("models"):
    os.makedirs("models")
model_filename = f"{system}_batch{batch_size}_epo{num_epoch}"
torch.save(model.state_dict(), os.path.join("models", model_filename))

# ----- plot trained model ------
with torch.no_grad():
    out = model.forward(dataset.U_u, dataset.Y_nn)  #
U_pem = out[:, 0].detach().numpy()
online.pemt(Y_sys, U_pem)
error_out = Y_sys[:, 0] - online.Yhat_data
print("R^2 = ", pem.R2(Y_sys[:, 0], online.Yhat_data))
# ----------- save training results------------
# df1 = pd.DataFrame(U, columns=['U'])
# df1 = pd.concat([df1, pd.DataFrame(Y_sys, columns=['Y_sys'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame(online.Yhat_data, columns=['Yhat_filter'])], axis=1)
# df1.to_csv(f'results_{system}_batch_train.csv', index=False)

plt.figure(1)
plt.plot(LOSS, 'k', label='NN loss')
plt.xlabel('Iteration')
plt.legend()
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(Y_sys, 'g', label='$Y_{sys}$')
ax[0].plot(online.Yhat_data, 'r--', label='$\hat{Y}$')
ax[0].legend()
ax[1].plot(U, 'k', label='u')
ax[1].set_xlabel('Time')  #  (s)
ax[1].legend()
plt.figure(4)
plt.plot(error_out, 'k', label='prediction error')
plt.xlabel('Time') #  (s)
plt.legend()
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(U, 'k', label='real u')
ax[0].legend()
ax[1].plot(U_pem, 'k', label='filtered u')
ax[1].set_xlabel('Time') # (s)
ax[1].legend()
plt.show()




