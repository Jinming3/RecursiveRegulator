import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import csv
# # -------------extract columns from txt file--------
# # node = np.loadtxt('n2temp.txt')
# # N = len(open('n2temp.txt').readlines())
# # node = np.loadtxt('n3temp.txt')
# # N = len(open('n3temp.txt').readlines())
# # node = np.loadtxt('n2humi.txt')
# # N = len(open('n2humi.txt').readlines())
# node = np.loadtxt('n3humi.txt')
# N = len(open('n3humi.txt').readlines())
# node = np.reshape(node, (N, 4))
# df = pd.DataFrame(node[:, 3]*5-5, columns=['time'])
# df = pd.concat([df, pd.DataFrame(node[:, 0]/100, columns=['Y'])], axis=1)
# df = pd.concat([df, pd.DataFrame(node[:, 1]/100, columns=['Yhat_mote'])], axis=1)
# df = pd.concat([df, pd.DataFrame(node[:, 2]/100, columns=['Vn_mote'])], axis=1)
# # df.to_csv('temp2.csv', index=False)
# # df.to_csv('temp3.csv', index=False)
# # df.to_csv('humi2.csv', index=False)
# df.to_csv('humi3.csv', index=False)
# ------extract Y[1], lower tank measurement-----------
# Y_raw = np.loadtxt('Tank2_Y_data.txt')
# Y_raw = np.reshape(Y_raw,(2, 7500))
# Y = []
# for i in range(0, 7500):
#     Y.append(Y_raw[1, i])
# np.savetxt('Y_lower.txt', Y, fmt='%.3f', delimiter='\n')
# # ------------------------convert mat file to csv selected data----------
# mat_contents = scipy.io.loadmat('DATA_EMPS.mat')
# print(mat_contents.keys())
# # df = pd.DataFrame(mat_contents['t'], columns=['time'])
# # df = pd.concat([df, pd.DataFrame(mat_contents['vir'], columns=['u_vol'])], axis=1)
# # df.to_csv('measure_EMPS.csv', index=False)
# ------------------- covert whole mat file EMPS ----------
# mat = scipy.io.loadmat('DATA_EMPS.mat')
# mat = scipy.io.loadmat('DATA_EMPS_PULSES.mat')
#
# print(mat.keys())
# mat = {k: v for k, v in mat.items() if k[0] != '_'}  # keys and values
# data = pd.DataFrame({k: pd.Series(v[:, 0]) for k, v in mat.items()})
# data.to_csv('measures_EMPS_PULSE.csv', index=False)

# # ---------- convert mat file to csv and change sampling time 0.005 ------------
# # mat_contents = scipy.io.loadmat('DATA_EMPS.mat')
# mat_contents = scipy.io.loadmat('DATA_EMPS_PULSES.mat')
#
# print(mat_contents.keys())
# time0 = np.array(mat_contents['t'])
# u0 = np.array(mat_contents['vir'])
#
#
# time = []
# u = []
# for i in range(0, time0.shape[0]):
#     if i % 5 == 0:
#         time.append(time0[i])
#         u.append(u0[i])
#
# df1 = pd.DataFrame(time, columns=['time'])
# df1 = pd.concat([df1, pd.DataFrame(u, columns=['u_vol'])], axis=1)
# # df1.to_csv('sample_EMPS_train.csv', index=False)
# # df1.to_csv('sample_EMPS_test.csv', index=False)

# ------------------- covert whole mat file EEG ----------
#
# mat = scipy.io.loadmat('Benchmark_EEG_small.mat')
# print(mat.keys())
# df = pd.DataFrame(data=mat['data'][0, :])  #
# df.to_csv('Benchmark_EEG_small.csv', index=False)


mat = scipy.io.loadmat('DATA_EMPS_PULSES.mat')
print(mat.keys())
df = pd.DataFrame(data=mat['data'][0, :])  #
df.to_csv('DATA_EMPS_PULSES.csv', index=False)


