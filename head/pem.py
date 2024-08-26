
import numpy as np


def R2(Y_sys, Yhat):
    """
    R-square metrics
    :param Y_sys: size N sequence
    :param Yhat: size N sequence
    :return:
    """
    s1 = np.sum((Y_sys - Yhat) ** 2)
    mean = np.mean(Y_sys)
    s2 = np.sum((Y_sys - mean) ** 2)

    return 1.0 - s1 / s2


def mse(Y_sys, Yhat):
    s = np.sum((Y_sys - Yhat) ** 2)
    m = s / len(Y_sys)
    return m


def normalize(x, r=5):
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


class PEM(object):  #
    def __init__(self, n, t, N, ur=1, m=1, r=1):
        self.n = n  # dim of X
        self.ur = ur #dim of U
        self.t = t  # dim of Theta
        self.N = N  # total data size
        self.m = m  # dimension of output Y
        self.r = r  # dimension of input U
        self.Thehat_data = np.zeros((self.N, self.t))
        self.Xhat_data = np.zeros((self.N, self.n))  # collect state estimates
        self.Yhat_data = np.zeros(self.N)  # collect prediction
        self.VN_data = np.zeros(self.N)  # prediction mean squared errors
        self.Xhat = np.zeros((self.n, 1))
        self.E_data = []  # collect n step ahead prediction error
        self.Ahat = np.eye(self.n, self.n, 1)  # canonical form
        self.Ahat_old = np.eye(self.n, self.n, 1)
        self.Bhat = np.zeros((self.n, self.ur))
        self.Chat = np.eye(1, self.n)  # [1, 0]  fixed!
        self.Chat_old = np.eye(1, self.n)  # [1, 0]
        self.Khat = np.zeros((self.n, self.m))
        self.Khat_old = np.zeros((self.n, self.m))
        self.Y = np.zeros((m, 1))
        self.Y_old = np.zeros((m, 1))
        self.Yhat = np.zeros((m, 1))
        self.Yhat_old = np.zeros((m, 1))
        self.U_old = np.zeros((self.ur, 1))
        self.Thehat = np.zeros((self.t, 1))

        self.Thehat_old = np.random.rand(self.t, 1) * 0.1
        # --------------------------------------------
        self.P_old2 = np.eye(t, t)*0.09
        self.Psi_old2 = np.eye(t, 1)*0.9
        self.Xhat_old = np.zeros((self.n, 1))

        # ---------------------------------------------
        self.I = np.eye(1)
        self.Xhatdot0 = np.zeros((self.n, self.t))
        self.Xhatdot_old = np.zeros((self.n, self.t))

    # ------------>>> test >>>>>>---------

    def pem_one(self, Y_sys, U, on):  # dependent funtion !!!!!!! Bhat =n*ur
        """
        :param Y_sys: size 1 (embedded) sequence, system raw measurements
        :param U: input raw data
        :param on: true == iteration updating Thehat
        :return:
        """

        self.Yhat = np.dot(self.Chat_old, self.Xhat)
        # assign theta-hat
        for a in range(self.n):
            self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            self.Ahat_old[self.n - 1, a] = self.Thehat_old[a, 0]
        for b in range(self.n):
            for b0 in range(self.ur):
                self.Bhat[b, b0] = self.Thehat[self.n + b0 + b * self.ur, 0]

        for h in range(self.n):
            self.Khat[h, 0] = self.Thehat[self.n + self.n*self.ur + h, 0]
            self.Khat_old[h, 0] = self.Thehat_old[self.n + self.n*self.ur + h, 0]
        # ---------------PEM iteration-------------------------

        if on:
            self.Y[:] = Y_sys[:]  # read in transmission
            for i0 in range(self.n):  # derivative of A
                self.Xhatdot0[self.n - 1, i0] = self.Xhat_old[i0, 0]
            for i1 in range(self.n):  # of B
                for i10 in range(self.ur):
                    self.Xhatdot0[i1, self.n  + i1 * self.ur + i10] = self.U_old[i10]#, 0]
            for i2 in range(self.n):  # of K
                self.Xhatdot0[i2, self.n + self.n*self.ur + i2] = self.Y_old - self.Yhat_old

            Xhatdot = self.Xhatdot0 + np.dot(self.Ahat_old, self.Xhatdot_old) - np.dot(self.Khat_old[:, [0]],
                                                                                       self.Psi_old2.T)
            Psi_old = np.dot(self.Chat_old, Xhatdot).T
            J = self.I + np.dot(np.dot(Psi_old.T, self.P_old2), Psi_old)

            P_old = self.P_old2 - np.dot(np.dot(np.dot(self.P_old2, Psi_old), np.linalg.pinv(J)),
                                         np.dot(Psi_old.T, self.P_old2))

            self.Thehat = self.Thehat_old + np.dot(np.dot(P_old, Psi_old), (self.Y - self.Yhat))
            # update thehat
            for a in range(self.n):
                self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            for b in range(self.n):
                for b0 in range(self.ur):
                    self.Bhat[b, b0] = self.Thehat[self.n + b0 + b * self.ur, 0]
            for h in range(self.n):
                self.Khat[h, 0] = self.Thehat[self.n + self.n * self.ur + h, 0]

            if len(U.shape) ==1:
                U = U.reshape(-1, 1)
            Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat, U) + self.Khat * (self.Y - self.Yhat)

            Yhat_new = np.dot(self.Chat, Xhat_new)
            # update every parameter which is time-variant
            self.Xhat_old = np.copy(self.Xhat)
            self.Xhat = np.copy(Xhat_new)
            self.Ahat_old = np.copy(self.Ahat)
            self.Khat_old = np.copy(self.Khat)
            self.Xhatdot_old = np.copy(Xhatdot)
            self.Psi_old2 = np.copy(Psi_old)
            self.U_old = np.copy(U)
            self.Thehat_old = np.copy(self.Thehat)
            self.P_old2 = np.copy(P_old)
            # squared prediction errors
            self.Y_old = np.copy(self.Y)
            self.Yhat_old = np.copy(self.Yhat)
            self.fix = np.copy(self.Y - self.Yhat)
            self.Yhat = np.copy(Yhat_new)

        if not on:  # check if to stop # only ssm
            if len(U.shape) ==1:
                U = U.reshape(-1, 1)
            Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat, U)
            Yhat_new = np.dot(self.Chat, Xhat_new)
            self.Xhat_old = np.copy(self.Xhat)
            self.Xhat = np.copy(Xhat_new)
            self.U_old = np.copy(U)
            self.Y_old = np.copy(self.Y)
            self.Yhat_old = np.copy(self.Yhat)
            self.Yhat = np.copy(Yhat_new)





