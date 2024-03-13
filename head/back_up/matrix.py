# """"
# calculate discrete from continuous, observable controllable matrix
# """
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete, lti, dlti, dstep
import sympy as sym


# # A = np.array([[0, 1], [0, -Fv/M]])
# # B = np.array([[0], [1/M]])
# # C = np.array([[1, 0]])  # == Cd
# # D = np.array([[0.]])  # == Dd
# # dt = 0.001
# #
# # d_system = cont2discrete((A, B, C, D), dt)
# #
# #
# # Ad = d_system[0]
# # Bd = d_system[1]
#
# contrl = np.concatenate((Bd, np.dot(Ad, Bd)), axis=1)  # controllable matrix
# observ = np.concatenate((C, np.dot(C, Ad)), axis=0)  # observable matrix
#
# Q = observ
# Ao = np.dot(np.dot(Q, Ad), np.linalg.inv(Q))
# Bo = np.dot(Q, Bd)
# Co = np.dot(C, np.linalg.inv(Q))
#
# #  ---------------- sketch board ----------
# # gt, kp, kv = 35.15, 160.18, 243.45
# # M, Fv, offset = 95.1089, 203.5034, -3.1648
# # Fc = 20.3935
# #
# # dt = 0.005
# # T = 10
# # N = int(T/dt)
# # y_all = []
# # for i in range(N):
# #     M = M - (0.001*i*dt)
# #     y_all.append(M)
# #
# #
# # time_all = np.arange(N) * dt
# # plt.figure()
# # plt.plot(time_all, y_all)

#--- string representation----
u = sym.Symbol('u')
l = sym.Symbol('l')
a = sym.Symbol('a')
b = sym.Symbol('b')
x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')
x3 = sym.Symbol('x3')
X = sym.Matrix([[x1], [x2], [x3]])
du = sym.Symbol('du')
dl = sym.Symbol('dl')
A = sym.Matrix([[u, 0, 0], [0, l, -l], [0, 0, 2*u]])
A2 = sym.Matrix([[u+du, 0, 0], [0, l+dl, -l-dl], [0, 0, 2*u+2*du]])
P_bar = A2-A

A_P_bar = np.dot(P_bar, A.inv())

phi1 = sym.Matrix([[1, 0, 0], [0, 1, 0]])  # phi^-1
P = np.dot(np.dot(phi1,A_P_bar),phi1.T)


# out = np.dot(phi1,np.dot(A, X))
# true=np.dot(phi1, np.dot(A2, X))  # true
# e = true-out
# # P = e@out.inv()
# # out_test= out+np.dot(P, out)  # test
# compen = np.dot(P, out)













