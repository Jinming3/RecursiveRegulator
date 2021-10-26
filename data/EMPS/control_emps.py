"""
generate EMPS signals; ageing system
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pylab

params = {
    'figure.figsize': (5, 3.7),
    'legend.fontsize': 10,
    'legend.labelspacing': 0.05,
    'legend.loc': 'upper right',
    'axes.labelsize': 11,
    'axes.labelpad': 0.1,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11

}  # 'axes.grid': True,
pylab.rcParams.update(params)


def normalize(x, r=1):
    """
    normalize an array
    :param x: array
    :param r: new array of [-r, r]
    :return: new array
    """
    out = []
    mini = np.amin(x)
    maxi = np.amax(x)
    for j in range(len(x)):
        # norm = (x[i] - mini) / (maxi - mini)  # [0, 1]
        norm = 2 * r * (x[j] - mini) / (maxi - mini) - r
        out.append(norm)
    return np.array(out)


class EMPS(object):
    def __init__(self, dt, pos, vel, acc, u):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.dt = dt
        self.u = u  # control signal voltage

    def measure(self, pos_ref, noise_process=0.0, noise_measure=0.0):
        self.u = kp * kv * (pos_ref - self.pos) - kv * self.vel
        # if self.u > satu:  # Maximum voltage (saturation)
        #     self.u = satu
        # if self.u < -satu:
        #     self.u = -satu
        force = gt * self.u
        self.acc = force / M - Fv / M * self.vel - Fc / M * np.sign(self.vel) - offset / M + np.random.randn() * noise_process
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
# satu = 10  # saturation
df = pd.read_csv("control_EMPS.csv")
time_exp = np.array(df['t']).astype(np.float32)
p_ref = np.array(df['p_ref']).astype(np.float32)
p_ref1 = np.array(df['p_ref1']).astype(np.float32)
pulse = np.array(df['pulse'])


# # -------------------------------------------------
# # ------- time-invariant system for training ------
# # -------- sampling time 0.005s ------------
# time = []
# p_ref0 = []
# for k in range(0, time_exp.shape[0]):
#     if k % 5 == 0:
#         time.append(time_exp[k])
#         p_ref0.append(p_ref[k])
# N0 = len(time)  # 4969
# M, Fv, Fc, offset = 95.1089, 203.5034, 20.3935, -3.1648
# dt = 0.005
# sampling = EMPS(dt, pos=0, vel=0, acc=0, u=0)
#
# velocity = []
# acceleration = []
# Y_sys = []
# U = []
# for i in range(N0):
#     p_control = p_ref0[i]
#     y = sampling.measure(p_control)
#     Y_sys.append(y)
#     U.append(sampling.u)
#     velocity.append(sampling.vel)
#     acceleration.append(sampling.acc)
#
# Y_sys = normalize(Y_sys, 1)
# # # # ----------- save data --------------
# # df0 = pd.DataFrame(time, columns=['time'])
# # df0 = pd.concat([df0, pd.DataFrame(U, columns=['u'])], axis=1)
# # df0 = pd.concat([df0, pd.DataFrame(Y_sys, columns=['Y_sys'])], axis=1)
# # df0 = pd.concat([df0, pd.DataFrame(velocity, columns=['vel'])], axis=1)
# # df0 = pd.concat([df0, pd.DataFrame(acceleration, columns=['acc'])], axis=1)
# # df0.to_csv('data_generate_train.csv', index=False)
#
#
# plt.figure(1)
# plt.subplot(211)
# plt.plot(Y_sys, 'g', label='measurement-position')  # time,
# plt.legend()
# plt.subplot(212)
# plt.plot(U, 'k', label='input-force')
# plt.xlabel('Time')  # (s)
# plt.legend()
# plt.figure(2)
# plt.subplot(211)
# plt.plot(velocity, 'y', label='velocity')  # time,
# plt.legend()
# plt.subplot(212)
# plt.plot(acceleration, 'r', label='acceleration')
# plt.xlabel('Time')  # (s)
# plt.legend()
# plt.show()

# ----------------------------------------------
# ------------ ageing system for inference, dt = 0.001s  ------------
# ---- don't forget to comment out above sampling -----
N = len(time_exp)  # 24841
dt = 0.001
velocity = []
acceleration = []
Y_sys = []
U = []
ageing = np.array([5000, 15000])

# ---- original system ---
offset = -3.1648
M = 95.1089
Fv = 203.5034
Fc = 20.3935
sampling = EMPS(dt, pos=0, vel=0, acc=0, u=0)
for i in range(ageing[0]):
    y = sampling.measure(p_ref1[i])
    Y_sys.append(y)
    U.append(sampling.u)
    velocity.append(sampling.vel)
    acceleration.append(sampling.acc)
# --- system changing ---
M = M * 0.7
Fv = Fv * 0.7
Fc = Fc * 0.8
for i in range(ageing[0], ageing[1]):
    y = sampling.measure(p_ref1[i], noise_process=10**-3, noise_measure=10**-4)
    Y_sys.append(y)
    U.append(sampling.u)
    velocity.append(sampling.vel)
    acceleration.append(sampling.acc)
offset = offset*0.7
M = M * 0.5
Fv = Fv * 0.6
Fc = Fc * 0.75
for i in range(ageing[1], N):
    y = sampling.measure(p_ref1[i], noise_process=10**-2, noise_measure=10**-4)
    Y_sys.append(y)
    U.append(sampling.u)
    velocity.append(sampling.vel)
    acceleration.append(sampling.acc)
Y_sys = normalize(Y_sys, 1)
# # ----------- save ------------
# df1 = pd.DataFrame(time_exp, columns=['time'])
# df1 = pd.concat([df1, pd.DataFrame(U, columns=['u'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame(Y_sys, columns=['Y_sys'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame(velocity, columns=['vel'])], axis=1)
# df1 = pd.concat([df1, pd.DataFrame(acceleration, columns=['acc'])], axis=1)
# df1.to_csv('data_generate_ageing.csv', index=False)

Y_sys = np.asarray(Y_sys)
U = np.asarray(U)
time = np.arange(N)
plt.figure(1)
plt.subplot(211)
plt.plot(time, Y_sys, 'g', label='measurement-position')  # time_exp,
plt.plot(time[ageing], Y_sys[ageing], 'rx')
plt.legend(loc='lower right')
plt.subplot(212)
plt.plot(time, U, 'k', label='input-force')
plt.plot(time[ageing], U[ageing], 'rx', label='ageing')
plt.xlabel('Time')  # (s)
plt.legend()
plt.figure(2)
plt.subplot(211)
plt.plot(velocity, 'y', label='velocity')  # time,
plt.legend()
plt.subplot(212)
plt.plot(acceleration, 'r', label='acceleration')
plt.xlabel('Time')  # (s)
plt.legend()
plt.show()
