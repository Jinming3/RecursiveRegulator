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
from pem import PEM
import NN_header
import pem

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
df = pd.read_csv(os.path.join("..", "data", "EMPS", "data_generate_train.csv"))
time_exp = np.array(df['time']).astype(np.float32)
Y_sys = np.array(df[['Y_sys']]).astype(np.float32)  # shape (N, 1) !!!!!
U = np.array(df[['u']]).astype(np.float32)
# vel = np.array(df['vel']).astype(np.float32)
# acc = np.array(df['acc']).astype(np.float32)
N = len(time_exp)
start_time = time.time()
# ------- set up PEM -----
n = 2  # dim of X
t = n + n + n  # canonical form, dim_y = 1
online = PEM(n, t, N)  # NN_header.
# parameters initialization
online.P_old2[:, :] = 0.9
online.Psi_old2[:, :] = 0.9
np.random.seed(3)
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
batch_size = 100  # sequence length in each batch
dataset = Data()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0)
dataiter = iter(train_loader)
data = dataiter.next()
U_u, Y_nn = data
# define NN model
input_size = 2
output_size = 1
hidden = 60
model = NN_header.NNFilter(input_size, hidden, output_size)
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
LOSS = []  # collect MSE
num_epoch = 80
num_iter = math.ceil(N / batch_size)  # number of iter in each epoch

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

        if (epoch + 1) % 5 == 0:  # unpack before print
            print(f'epoch {epoch + 1}/{num_epoch}, step {i + 1}/{num_iter}: loss= {loss.item():.5f}')

train_time = time.time() - start_time
print(f"\nTrain time: {train_time}")
# Save model
if not os.path.exists("models"):
    os.makedirs("models")
model_filename = f"ageing_batch{batch_size}"
torch.save(model.state_dict(), os.path.join("models", model_filename))

# ----- plot trained model ------
with torch.no_grad():
    out = model(dataset.U_u, dataset.Y_nn)
U_pem = out[:, 0].detach().numpy()
online.pemt(Y_sys, U_pem)
error_out = Y_sys[:, 0] - online.Yhat_data

print("R^2 = ", pem.R2(Y_sys[:, 0], online.Yhat_data))
# ----------- save training results------------
df1 = pd.concat([df, pd.DataFrame(online.Yhat_data, columns=['Yhat_pem'])], axis=1)
df1 = pd.concat([df1, pd.DataFrame(error_out, columns=['error_out'])], axis=1)

df1.to_csv('ageing_batch_train.csv', index=False)

# plt.figure(1)
# plt.plot(LOSS, 'k', label='NN loss')
# plt.xlabel('Iteration')
# plt.legend()
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(time_exp, Y_sys, 'g', label='measurement')
ax[0].plot(time_exp, online.Yhat_data, 'r--', label='estimate')
ax[0].legend()
ax[1].plot(time_exp, U, 'k', label='input')
ax[1].set_xlabel("Time (s)")
ax[1].legend()

plt.figure(4)
plt.plot(time_exp, error_out, 'k', label='prediction error')
plt.xlabel('Time (s)')
plt.legend()
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(time_exp, U, 'k', label='real u')
# ax[1].plot(time_exp, U_pem, 'k', label='filtered u')
# ax[1].set_xlabel("Time (s)")
# plt.legend()
plt.show()
