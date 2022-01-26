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
          'legend.loc': 'upper right',
          'axes.labelsize': 11,
          'axes.labelpad': 0.5,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11
          }
pylab.rcParams.update(params)

# # # # -------read CED signals --------
# df = pd.read_csv(os.path.join("..", "data", "CED", "DATAPRBS.csv"))
# Y_sys = np.array(df['z1']).astype(np.float32)
# U = np.array(df['u1']).astype(np.float32)
# system = 'ced'

# # ---- load data WH
# df = pd.read_csv(os.path.join("..", "data", "WH", "WienerHammerBenchmark.csv"))
# U = np.array(df['uBenchMark'][6000:10000]).astype(np.float32)
# Y_sys = np.array(df['yBenchMark'][6000:10000]).astype(np.float32)
# system = 'WH'
# # ---- load data F16
# df = pd.read_csv(os.path.join("..", "data", "F16", "F16Data_SineSw_Level1.csv"))
# Y_sys = np.array(df['Acceleration2'][40000:60000]).astype(np.float32)
# U = np.array(df['Force'][40000:60000]).astype(np.float32)
# system = 'F16'
# #
# # ----------RLC ---------------
# df = pd.read_csv(os.path.join("..", "data", "RLC", "RLC_data_id.csv"))
# U = np.array(df['V_IN']).astype(np.float32)
# Y_sys = np.array(df['V_C']).astype(np.float32)
# system = 'RLC'
# Y_sys = NN_header.normalize(Y_sys, 1)
# U = NN_header.normalize(U, 1)
# # ----cascaded tanks------
# df = pd.read_csv(os.path.join("..", "data", "CTS", "tankdata.csv"))
# U = np.array(df['uEst']).astype(np.float32)
# Y_sys = np.array(df['yEst']).astype(np.float32)
# system = 'CTS'
# # -------EMPS system--------
df_data = pd.read_csv(os.path.join("..", "data", "EMPS","DATA_EMPS_SC.csv"))
time_exp = np.array(df_data['time_exp']).astype(np.float32)
Y_sys = np.array(df_data['q_meas']).astype(np.float32)
U = np.array(df_data['u_in']).astype(np.float32)
system = 'EMPS'
# ------ emps ageing ------
# df_data = pd.read_csv(os.path.join("..", "data", "EMPS", "data_generate_train.csv"))
# time_exp = np.array(df_data['time']).astype(np.float32)
# Y_sys = np.array(df_data['Y_sys']).astype(np.float32)
# U = np.array(df_data['u']).astype(np.float32)
# system = 'ageing'


N = len(Y_sys)
# print('N:', N)
# N = int(500)
# start_time = time.time()
# ------- set up PEM -----
n = 2  # dim of X
t = n + n + n  # canonical form, dim_y = 1
online = PEM(n, t, N)
# parameters initialization
online.P_old2[:, :] = 0.9
online.Psi_old2[:, :] = 0.9
np.random.seed(3)   # CTS 1554419047, RLC 3
online.Thehat_old = np.random.rand(t, 1)
online.Xhat_old = np.random.rand(n, 1)
# y=0, train PEM-----------
online.pemt(Y_sys, U*0)

print("R^2(PEM) = ", pem.R2(Y_sys, online.Yhat_data))
# plt.figure(0)
# plt.plot(Y_sys, 'g', label='measurement')
# plt.plot(online.Yhat_data, 'r', label='PEM-yhat')
# plt.xlabel('Iteration')
# plt.legend()
# plt.show()
# tensor KF, train NN---
Y_sys = Y_sys[:, np.newaxis]
U = U[:, np.newaxis]
A = torch.from_numpy(np.copy(online.Ahat))
K = torch.from_numpy(np.copy(online.Khat))
C = torch.from_numpy(np.copy(online.Chat))
X = torch.from_numpy(np.copy(online.Xhat))
B = torch.ones(n, 1)

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
batch_size = 25  # sequence length in each batch,  WH 20 ; CTS 30,  CED 20, RLC 30, EMPS 30
dataset = Data()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0)
dataiter = iter(train_loader)
data = dataiter.next()
U_u, Y_nn = data
# define NN model
num_epoch = 30
input_size = 2
output_size = 1
hidden = 60
model = NN_header.NNFilter(input_size, hidden, output_size)
learning_rate = 0.01 # emps, CTS: 0.01, else: rlc 0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
LOSS = []  # collect MSE
num_iter = math.ceil(N/batch_size)  # number of iter in each epoch
start_time = time.time()
# --------- training loop --------
for epoch in range(num_epoch):
    for i, (U_u, Y_nn) in enumerate(train_loader):

        out = model.forward(U_u, Y_nn)  # prediction = forward pass

        X_nn = torch.zeros((n, Y_nn.shape[0]), dtype=torch.double)
        Yhat = torch.zeros((1, Y_nn.shape[0]), dtype=torch.double)
        X_nn[:, [0]] = X
        for p in range(Y_nn.shape[0]):

            X_nn[:, [p]] = torch.matmul(A, X_nn[:, [p-1]]) + torch.matmul(B, out[[p], :]) + K*(Y_nn[[p], :] - Yhat[:, [p-1]])
            Yhat[:, [p]] = torch.matmul(C, X_nn[:, [p]])

        loss = NN_header.mse(Y_nn, Yhat)
        LOSS.append(loss.item())
        loss.backward(retain_graph=True)
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
    out = model.forward(dataset.U_u, dataset.Y_nn)
    X_nn = torch.zeros((n, dataset.Y_nn.shape[0]), dtype=torch.double)
    Yhat = torch.zeros((1, dataset.Y_nn.shape[0]), dtype=torch.double)
    for p in range(dataset.Y_nn.shape[0]):
        X_nn[:, [p]] = torch.matmul(A, X_nn[:, [p - 1]]) + torch.matmul(B, out[[p], :]) + K * (dataset.Y_nn[[p], :] - Yhat[:, [p - 1]])
        Yhat[:, [p]] = torch.matmul(C, X_nn[:, [p]])

Y = dataset.Y_nn[:, 0].detach().numpy()
Yhat = Yhat[0, :].detach().numpy()
out = out[:, 0].detach().numpy()
R2 = pem.R2(Y, Yhat)
print("R^2 = ", R2)
# ----------- save training results------------
# df1 = pd.DataFrame(Yhat, columns=['filter'])
# df1 = pd.concat([df1, pd.DataFrame([learning_rate], columns=['lr'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame([batch_size], columns=['batch'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame([num_epoch], columns=['epoch'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame([train_time], columns=['train_time'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame([R2], columns=['r2'])], axis=1)
# df1.to_csv(f'train_{system}.csv', index=False)

plt.figure(1)
plt.plot(LOSS, 'k', label='loss')
plt.xlabel('Iteration')
plt.legend()
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(Y_sys, 'g', label='$measurement$')
ax[0].plot(Yhat, 'r--', label='$\hat{y}$')
ax[0].legend()
ax[1].plot(U, 'k', label='u')
ax[1].set_xlabel('Time')  #  (s)
ax[1].legend()

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(U, 'k', label='u')
ax[0].grid(True)
ax[0].legend()
ax[1].plot(out, 'k', label="$u'$")
ax[1].set_xlabel('Time')
ax[1].grid(True)
ax[1].legend()
plt.show()




