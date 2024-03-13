"""
single-output pem of canonical form 
Functions:
pemt -- with threshold to stop and restart, redefined N for batch-calculation, set thres=0, nonstop PEM
peml -- packet-loss reconstruction
pemt_pkf -- stop and transmit using PKF
"""
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


# ----------- >>> test --------
"""
class PEM_step(object):  # extract Ahat for NN
    def __init__(self, n, t, N):
        self.n = n  # dim of X
        self.t = t  # dim of Theta
        self.N = N  # total data size

        self.Xhat_data = np.zeros((N, n))  # collect state estimates
        self.Yhat_data = np.zeros(N)  # collect prediction
        self.VN_data = np.zeros(N)  # prediction mean squared errors
        self.Xhat = np.zeros((n, 1))
        self.Xhat_old = np.zeros((n, 1))
        self.Ahat = np.eye(n, n, 1)  # canonical form
        self.Ahat_old = np.eye(n, n, 1)
        self.Bhat = np.zeros((n, 1))
        self.Chat = np.eye(1, n)  # [1, 0]  fixed!
        self.Chat_old = np.eye(1, n)  # [1, 0]
        self.Khat = np.zeros((n, 1))
        self.Khat_old = np.zeros((n, 1))
        self.Y = np.zeros((1, 1))
        self.Y_old = np.zeros((1, 1))
        self.Yhat = np.zeros((1, 1))
        self.Yhat_old = np.zeros((1, 1))
        self.U_old = np.zeros(1)
        self.Thehat = np.zeros((t, 1))
        self.Thehat_old = np.zeros((t, 1))
        self.P_old2 = np.eye(t, t)
        self.Psi_old2 = np.eye(t, 1)
        self.I = np.eye(1)
        self.Xhatdot0 = np.zeros((n, t))
        self.Xhatdot_old = np.zeros((n, t))

    # ------------ ---- - non stop PEM --------- -------------------
    def forward(self, Y_sys, U, xn):  # xn: xhat from NN
        k = 0
        VN0 = 0
        self.N = Y_sys.shape[0]  # reshape if batch calculation
        self.Xhat_data = np.zeros((self.N, self.n))  # collect state estimates
        self.Yhat_data = np.zeros(self.N)  # collect prediction
        self.VN_data = np.zeros(self.N)  # prediction mean squared errors
        # self.Yhat_old = np.dot(self.Chat_old, self.Xhat_old)

        # assign theta-hat
        for a in range(self.n):
            self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            self.Ahat_old[self.n - 1, a] = self.Thehat_old[a, 0]
        for b in range(self.n):
            self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
        for h in range(self.n):
            self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
            self.Khat_old[h, 0] = self.Thehat_old[self.n + self.n + h, 0]

        self.Yhat_old = np.dot(self.Chat_old, np.dot(self.Ahat_old, xn))


        # ---------------PEM iteration-------------------------
        q = 0
        while q < self.N:

            self.Y[:] = Y_sys[q]  # read in transmission
            for i0 in range(self.n):  # derivative of A
                self.Xhatdot0[self.n - 1, i0] = self.Xhat_old[i0, 0]
            for i1 in range(self.n):  # of B
                self.Xhatdot0[i1, self.n + i1] = self.U_old[0]
            for i2 in range(self.n):  # of K
                self.Xhatdot0[i2, self.n + self.n + i2] = self.Y_old - self.Yhat_old

            Xhatdot = self.Xhatdot0 + np.dot(self.Ahat_old, self.Xhatdot_old) - np.dot(self.Khat_old[:, [0]],
                                                                                       self.Psi_old2.T)
            Psi_old = np.dot(self.Chat_old, Xhatdot).T
            J = self.I + np.dot(np.dot(Psi_old.T, self.P_old2), Psi_old)
            P_old = self.P_old2 - np.dot(np.dot(np.dot(self.P_old2, Psi_old), np.linalg.pinv(J)),
                                         np.dot(Psi_old.T, self.P_old2))

            self.Yhat = np.dot(self.Chat, np.dot(self.Ahat, xn))

            self.Thehat = self.Thehat_old + np.dot(np.dot(P_old, Psi_old), (self.Y - self.Yhat))

            # update thehat
            for a in range(self.n):
                self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            for b in range(self.n):
                self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
            for h in range(self.n):
                self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]

            # self.Yhat = np.dot(self.Chat, np.dot(self.Ahat, xn))  # ytilde = A * yhat

            Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (self.Y - self.Yhat)
            Yhat_new = np.dot(self.Chat, Xhat_new)
            # update every parameter which is time-variant
            self.Xhat_old = np.copy(self.Xhat)
            self.Xhat = np.copy(Xhat_new)
            self.Xhat_data[q, :] = np.copy(self.Xhat[:, 0])
            self.Ahat_old = np.copy(self.Ahat)
            self.Khat_old = np.copy(self.Khat)
            self.Xhatdot_old = np.copy(Xhatdot)
            self.Psi_old2 = np.copy(Psi_old)
            self.U_old[:] = np.copy(U[q])
            self.Thehat_old = np.copy(self.Thehat)
            self.P_old2 = np.copy(P_old)
            # squared prediction errors
            E = self.Y - self.Yhat
            sqE = np.dot(E.T, E)
            VN0 = VN0 + sqE
            k = k + 1
            VN = VN0 / k

            self.Y_old = np.copy(self.Y)
            self.Yhat_old = np.copy(self.Yhat)
            self.Yhat = np.copy(Yhat_new)
            # ---------- save data-----------------
            self.Yhat_data[q] = self.Yhat[0]

            self.VN_data[q] = VN
            q = q + 1
        # return self.Ahat

"""

# ----------<<<<test------------

class PEM(object):  #
    def __init__(self, n, t, N):
        self.n = n  # dim of X
        self.t = t  # dim of Theta
        self.N = N  # total data size

        self.Xhat_data = np.zeros((N, n))  # collect state estimates
        self.Yhat_data = np.zeros(N)  # collect prediction
        self.VN_data = np.zeros(N)  # prediction mean squared errors
        self.Xhat = np.zeros((n, 1))
        self.Xhat_old = np.zeros((n, 1))
        self.Ahat = np.eye(n, n, 1)  # canonical form
        self.Ahat_old = np.eye(n, n, 1)
        self.Bhat = np.zeros((n, 1))
        self.Chat = np.eye(1, n)  # [1, 0]  fixed!
        self.Chat_old = np.eye(1, n)  # [1, 0]
        self.Khat = np.zeros((n, 1))
        self.Khat_old = np.zeros((n, 1))
        self.Y = np.zeros((1, 1))
        self.Y_old = np.zeros((1, 1))
        self.Yhat = np.zeros((1, 1))
        self.Yhat_old = np.zeros((1, 1))
        self.U_old = np.zeros(1)
        self.Thehat = np.zeros((t, 1))
        self.Thehat_old = np.zeros((t, 1))
        self.P_old2 = np.eye(t, t)
        self.Psi_old2 = np.eye(t, 1)
        self.I = np.eye(1)
        self.Xhatdot0 = np.zeros((n, t))
        self.Xhatdot_old = np.zeros((n, t))

    # ------------>>> test >>>>>>---------
    def pem_rest(self, Y_sys, U, on=True):  # dependent funtion !!!!!!!
        """
        on-off PEM, threshold and slot in parent function!!! when rest, no reading y
        similar to forward pem
        :param Y_sys: size-N sequence, system raw measurements
        :param U: N*1 array, input raw data
        :param on: true == iteration updating Thehat
        :return:
        """
        self.N = Y_sys.shape[0]
        self.Yhat = np.dot(self.Chat_old, self.Xhat)
        self.Xhat_data = np.zeros((self.N, self.n))
        self.Yhat_data = np.zeros(self.N)  # collect prediction
        # assign theta-hat
        for a in range(self.n):
            self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            self.Ahat_old[self.n - 1, a] = self.Thehat_old[a, 0]
        for b in range(self.n):
            self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
        for h in range(self.n):
            self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
            self.Khat_old[h, 0] = self.Thehat_old[self.n + self.n + h, 0]
        # ---------------PEM iteration-------------------------
        q = 0
        while q < self.N:
            if on:
                self.Y[:] = Y_sys[q]  # read in transmission
                for i0 in range(self.n):  # derivative of A
                    self.Xhatdot0[self.n - 1, i0] = self.Xhat_old[i0, 0]
                for i1 in range(self.n):  # of B
                    self.Xhatdot0[i1, self.n + i1] = self.U_old[0]
                for i2 in range(self.n):  # of K
                    self.Xhatdot0[i2, self.n + self.n + i2] = self.Y_old - self.Yhat_old

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
                    self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
                for h in range(self.n):
                    self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]

                Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (self.Y - self.Yhat)
                Yhat_new = np.dot(self.Chat, Xhat_new)
                # update every parameter which is time-variant
                self.Xhat_old = np.copy(self.Xhat)
                self.Xhat = np.copy(Xhat_new)
                self.Xhat_data[q, :] = np.copy(self.Xhat[:, 0])
                self.Ahat_old = np.copy(self.Ahat)
                self.Khat_old = np.copy(self.Khat)
                self.Xhatdot_old = np.copy(Xhatdot)
                self.Psi_old2 = np.copy(Psi_old)
                self.U_old[:] = np.copy(U[q])
                self.Thehat_old = np.copy(self.Thehat)
                self.P_old2 = np.copy(P_old)
                # squared prediction errors
                self.Y_old = np.copy(self.Y)
                self.Yhat_old = np.copy(self.Yhat)
                self.Yhat = np.copy(Yhat_new)
                # save data-----------------
                self.Yhat_data[q] = self.Yhat[0]

            if not on:  # check if to stop # only KF-predictor no updating
                fix = self.Y - self.Yhat
                # print(fix)
                Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat,
                                                                 U[q]) + self.Khat * fix  # (self.Y - self.Yhat)  #
                Yhat_new = np.dot(self.Chat, Xhat_new)
                self.Xhat_old = np.copy(self.Xhat)
                self.Xhat = np.copy(Xhat_new)
                self.Xhat_data[q, :] = self.Xhat[:, 0]
                self.U_old[:] = U[q]
                self.Y_old = np.copy(self.Y)
                self.Yhat_old = np.copy(self.Yhat)
                self.Yhat = np.copy(Yhat_new)
                # save
                self.Yhat_data[q] = self.Yhat[0]

            q = q + 1

    # ----------------------- >> back up ---------------------------------

    # def pem_rest(self, Y_sys, U, on=True):  # dependent funtion !!!!!!!
    #     """
    #     on-off PEM, threshold and slot in parent function!!! when rest, no reading y
    #     similar to forward pem
    #     :param Y_sys: size-N sequence, system raw measurements
    #     :param U: N*1 array, input raw data
    #     :param on: true == iteration update Thehat
    #     :return:
    #     """
    #     self.N = Y_sys.shape[0]
    #     self.Yhat = np.dot(self.Chat_old, self.Xhat)
    #     self.Xhat_data = np.zeros((self.N, self.n))
    #     self.Yhat_data = np.zeros(self.N)  # collect prediction
    #     # assign theta-hat
    #     for a in range(self.n):
    #         self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
    #         self.Ahat_old[self.n - 1, a] = self.Thehat_old[a, 0]
    #     for b in range(self.n):
    #         self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
    #     for h in range(self.n):
    #         self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
    #         self.Khat_old[h, 0] = self.Thehat_old[self.n + self.n + h, 0]
    #     # ---------------PEM iteration-------------------------
    #     q = 0
    #     while q < self.N:
    #         self.Y[:] = Y_sys[q]  # read in transmission
    #         for i0 in range(self.n):  # derivative of A
    #             self.Xhatdot0[self.n - 1, i0] = self.Xhat_old[i0, 0]
    #         for i1 in range(self.n):  # of B
    #             self.Xhatdot0[i1, self.n + i1] = self.U_old[0]
    #         for i2 in range(self.n):  # of K
    #             self.Xhatdot0[i2, self.n + self.n + i2] = self.Y_old - self.Yhat_old
    #
    #         Xhatdot = self.Xhatdot0 + np.dot(self.Ahat_old, self.Xhatdot_old) - np.dot(self.Khat_old[:, [0]],
    #                                                                                    self.Psi_old2.T)
    #         Psi_old = np.dot(self.Chat_old, Xhatdot).T
    #         J = self.I + np.dot(np.dot(Psi_old.T, self.P_old2), Psi_old)
    #         P_old = self.P_old2 - np.dot(np.dot(np.dot(self.P_old2, Psi_old), np.linalg.pinv(J)),
    #                                      np.dot(Psi_old.T, self.P_old2))
    #
    #         self.Thehat = self.Thehat_old + np.dot(np.dot(P_old, Psi_old), (self.Y - self.Yhat))
    #         # update thehat
    #         for a in range(self.n):
    #             self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
    #         for b in range(self.n):
    #             self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
    #         for h in range(self.n):
    #             self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
    #
    #         Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (self.Y - self.Yhat)
    #         Yhat_new = np.dot(self.Chat, Xhat_new)
    #         # update every parameter which is time-variant
    #         self.Xhat_old = np.copy(self.Xhat)
    #         self.Xhat = np.copy(Xhat_new)
    #         self.Xhat_data[q, :] = np.copy(self.Xhat[:, 0])
    #         self.Ahat_old = np.copy(self.Ahat)
    #         self.Khat_old = np.copy(self.Khat)
    #         self.Xhatdot_old = np.copy(Xhatdot)
    #         self.Psi_old2 = np.copy(Psi_old)
    #         self.U_old[:] = np.copy(U[q])
    #         self.Thehat_old = np.copy(self.Thehat)
    #         self.P_old2 = np.copy(P_old)
    #         # squared prediction errors
    #         self.Y_old = np.copy(self.Y)
    #         self.Yhat_old = np.copy(self.Yhat)
    #         self.Yhat = np.copy(Yhat_new)
    #         # save data-----------------
    #         self.Yhat_data[q] = self.Yhat[0]
    #
    #         while q < self.N and not on:  # check if to stop # only KF-predictor no updating
    #             fix = self.Y - self.Yhat
    #             self.stop.append(q)
    #             self.theta.append(self.Thehat[:, 0])
    #             q = q + 1
    #             # print(fix)
    #             Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat, U[q]) + self.Khat * fix  # (self.Y - self.Yhat)  #
    #             Yhat_new = np.dot(self.Chat, Xhat_new)
    #             self.Xhat_old = np.copy(self.Xhat)
    #             self.Xhat = np.copy(Xhat_new)
    #             self.Xhat_data[q, :] = self.Xhat[:, 0]
    #             self.U_old[:] = U[q]
    #             self.Y_old = np.copy(self.Y)
    #             self.Yhat_old = np.copy(self.Yhat)
    #             self.Yhat = np.copy(Yhat_new)
    #             # save
    #             self.Yhat_data[q] = self.Yhat[0]
    #             if on:  # to update again
    #                 self.restart.append(q + 1)
    #                 break
    #             q = q + 1
    #         q = q + 1
    # ----<<< back up -----

    def pem_R2(self, Y_sys, U, threshold1=0.96, threshold2=0.98, slot=80):
        """
        on-off PEM, variance threshold=0 nonstop, when rest, no reading y
        :param Y_sys: size-N sequence, system raw measurements
        :param U: N*1 array, input raw data
        :param threshold1: variance of a window of R2
        :param threshold2: > threshold1
        :param slot: sensitivity window of R2
        :return:
        """
        self.N = Y_sys.shape[0]
        self.Yhat = np.dot(self.Chat_old, self.Xhat)
        self.Xhat_data = np.zeros((self.N, self.n))
        self.Yhat_data = np.zeros(self.N)  # collect prediction
        # for on-off points collecting
        self.restart = []
        self.stop = []
        self.theta = []  # collect identified parameters at stops
        # assign theta-hat
        for a in range(self.n):
            self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            self.Ahat_old[self.n - 1, a] = self.Thehat_old[a, 0]
        for b in range(self.n):
            self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
        for h in range(self.n):
            self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
            self.Khat_old[h, 0] = self.Thehat_old[self.n + self.n + h, 0]
        # ---------------PEM iteration-------------------------
        q = 0
        while q < self.N:
            self.Y[:] = Y_sys[q]  # read in transmission
            for i0 in range(self.n):  # derivative of A
                self.Xhatdot0[self.n - 1, i0] = self.Xhat_old[i0, 0]
            for i1 in range(self.n):  # of B
                self.Xhatdot0[i1, self.n + i1] = self.U_old[0]
            for i2 in range(self.n):  # of K
                self.Xhatdot0[i2, self.n + self.n + i2] = self.Y_old - self.Yhat_old

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
                self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
            for h in range(self.n):
                self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]

            Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (self.Y - self.Yhat)
            Yhat_new = np.dot(self.Chat, Xhat_new)
            # update every parameter which is time-variant
            self.Xhat_old = np.copy(self.Xhat)
            self.Xhat = np.copy(Xhat_new)
            self.Xhat_data[q, :] = np.copy(self.Xhat[:, 0])
            self.Ahat_old = np.copy(self.Ahat)
            self.Khat_old = np.copy(self.Khat)
            self.Xhatdot_old = np.copy(Xhatdot)
            self.Psi_old2 = np.copy(Psi_old)
            self.U_old[:] = np.copy(U[q])
            self.Thehat_old = np.copy(self.Thehat)
            self.P_old2 = np.copy(P_old)
            # squared prediction errors

            self.Y_old = np.copy(self.Y)
            self.Yhat_old = np.copy(self.Yhat)
            self.Yhat = np.copy(Yhat_new)
            # save data-----------------
            self.Yhat_data[q] = self.Yhat[0]
            match = R2(Y_sys[q - slot:q], self.Yhat_data[q - slot:q])
            # print(f'R2 is {match} at data {q}')
            if match > threshold2:  # check if to stop
                fix = self.Y - self.Yhat
                self.stop.append(q)
                print(f'stop at {q}, with R2={match} ')
                self.theta.append(self.Thehat[:, 0])
                q = q + 1
                while q < self.N:  # only KF-predictor no updating

                    # print(fix)
                    Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat,
                                                                     U[q]) + self.Khat * fix  # (self.Y - self.Yhat)  #
                    Yhat_new = np.dot(self.Chat, Xhat_new)
                    self.Xhat_old = np.copy(self.Xhat)
                    self.Xhat = np.copy(Xhat_new)
                    self.Xhat_data[q, :] = self.Xhat[:, 0]
                    self.U_old[:] = U[q]
                    self.Y_old = np.copy(self.Y)
                    self.Yhat_old = np.copy(self.Yhat)
                    self.Yhat = np.copy(Yhat_new)
                    # save
                    self.Yhat_data[q] = self.Yhat[0]
                    match = R2(Y_sys[q - slot:q], self.Yhat_data[q - slot:q])
                    # print(f'R2 is {match} at data {q}')
                    if match < threshold1:  # to identify again
                        self.restart.append(q + 1)
                        print(f'restart at {q}, with R2={match} ')
                        break
                    q = q + 1
            q = q + 1
        self.theta = np.asarray(self.theta)  # a list of arrays. if nonstop, use self.Thehat

    # ------<<<<<<<<<< test <<<< ------------------------

    # ------------ ---- - non stop PEM --------- -------------------
    def forward(self, Y_sys, U):
        k = 0
        VN0 = 0
        self.N = Y_sys.shape[0]  # reshape if batch calculation
        self.Xhat_data = np.zeros((self.N, self.n))  # collect state estimates
        self.Yhat_data = np.zeros(self.N)  # collect prediction
        self.VN_data = np.zeros(self.N)  # prediction mean squared errors
        self.Yhat_old = np.dot(self.Chat_old, self.Xhat_old)
        self.Thehat_data = np.zeros((self.N, self.t))  # not useful in step optimization

        # assign theta-hat
        for a in range(self.n):
            self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            self.Ahat_old[self.n - 1, a] = self.Thehat_old[a, 0]
        for b in range(self.n):
            self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
        for h in range(self.n):
            self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
            self.Khat_old[h, 0] = self.Thehat_old[self.n + self.n + h, 0]
        # ---------------PEM iteration-------------------------
        q = 0
        while q < self.N:

            self.Y[:] = Y_sys[q]  # read in transmission
            for i0 in range(self.n):  # derivative of A
                self.Xhatdot0[self.n - 1, i0] = self.Xhat_old[i0, 0]
            for i1 in range(self.n):  # of B
                self.Xhatdot0[i1, self.n + i1] = self.U_old[0]
            for i2 in range(self.n):  # of K
                self.Xhatdot0[i2, self.n + self.n + i2] = self.Y_old - self.Yhat_old

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
                self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
            for h in range(self.n):
                self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]

            Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (self.Y - self.Yhat)
            Yhat_new = np.dot(self.Chat, Xhat_new)
            # update every parameter which is time-variant
            self.Xhat_old = np.copy(self.Xhat)
            self.Xhat = np.copy(Xhat_new)
            self.Xhat_data[q, :] = np.copy(self.Xhat[:, 0])
            self.Ahat_old = np.copy(self.Ahat)
            self.Khat_old = np.copy(self.Khat)
            self.Xhatdot_old = np.copy(Xhatdot)
            self.Psi_old2 = np.copy(Psi_old)
            self.U_old[:] = np.copy(U[q])
            self.P_old2 = np.copy(P_old)
            self.Thehat_old = np.copy(self.Thehat)

            # squared prediction errors
            E = self.Y - self.Yhat
            sqE = np.dot(E.T, E)
            VN0 = VN0 + sqE
            k = k + 1
            VN = VN0 / k

            self.Y_old = np.copy(self.Y)
            self.Yhat_old = np.copy(self.Yhat)
            self.Yhat = np.copy(Yhat_new)
            # ---------- save data-----------------
            self.Yhat_data[q] = self.Yhat[0]
            self.Thehat_data[q, :] = np.copy(self.Thehat[:, 0])  # not useful in step optimization
            self.VN_data[q] = VN
            q = q + 1

    # ---------  with functionality in Thesis ------------
    def pemt(self, Y_sys, U, threshold1=0, threshold2=0, slot=100):
        """
        on-off PEM, variance threshold=0 nonstop
        :param Y_sys: size-N sequence, system raw measurements
        :param U: N*1 array, input raw data
        :param threshold1: variance of a window of VN
        :param threshold2: > threshold1
        :param slot: window of VN
        :return:
        """
        self.N = Y_sys.shape[0]
        # self.N = len(Y_sys)
        self.Yhat = np.dot(self.Chat_old, self.Xhat)
        self.Xhat_data = np.zeros((self.N, self.n))
        self.Yhat_data = np.zeros(self.N)  # collect prediction
        self.VN_data = np.zeros(self.N)
        k = 0
        VN0 = 0
        # for on-off points collecting
        self.stop = []
        self.restart = []
        self.theta = []  # collect identified parameters at stops
        # assign theta-hat
        for a in range(self.n):
            self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            self.Ahat_old[self.n - 1, a] = self.Thehat_old[a, 0]
        for b in range(self.n):
            self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
        for h in range(self.n):
            self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
            self.Khat_old[h, 0] = self.Thehat_old[self.n + self.n + h, 0]
        # ---------------PEM iteration-------------------------
        q = 0
        while q < self.N:
            self.Y[:] = Y_sys[q]  # read in transmission
            for i0 in range(self.n):  # derivative of A
                self.Xhatdot0[self.n - 1, i0] = self.Xhat_old[i0, 0]
            for i1 in range(self.n):  # of B
                self.Xhatdot0[i1, self.n + i1] = self.U_old[0]
            for i2 in range(self.n):  # of K
                self.Xhatdot0[i2, self.n + self.n + i2] = self.Y_old - self.Yhat_old

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
                self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
            for h in range(self.n):
                self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]

            Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (self.Y - self.Yhat)
            Yhat_new = np.dot(self.Chat, Xhat_new)
            # update every parameter which is time-variant
            self.Xhat_old = np.copy(self.Xhat)
            self.Xhat = np.copy(Xhat_new)
            self.Xhat_data[q, :] = np.copy(self.Xhat[:, 0])
            self.Ahat_old = np.copy(self.Ahat)
            self.Khat_old = np.copy(self.Khat)
            self.Xhatdot_old = np.copy(Xhatdot)
            self.Psi_old2 = np.copy(Psi_old)
            self.U_old[:] = np.copy(U[q])
            self.Thehat_old = np.copy(self.Thehat)
            self.P_old2 = np.copy(P_old)
            # squared prediction errors
            E = self.Y - self.Yhat
            sqE = np.dot(E.T, E)
            VN0 = VN0 + sqE
            k = k + 1
            VN = VN0 / k
            self.Y_old = np.copy(self.Y)
            self.Yhat_old = np.copy(self.Yhat)
            self.Yhat = np.copy(Yhat_new)
            # save data-----------------
            self.Yhat_data[q] = self.Yhat[0]
            self.VN_data[q] = VN
            slot_vn = self.VN_data[q - slot:q]  # check if to stop
            if np.var(slot_vn) < threshold1:
                self.stop.append(q)
                self.theta.append(self.Thehat[:, 0])
                q = q + 1
                while q < self.N:  # only KF-predictor
                    self.Y[:] = Y_sys[q]
                    Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat, U[q]) + self.Khat * (self.Y - self.Yhat)
                    Yhat_new = np.dot(self.Chat, Xhat_new)
                    self.Xhat_old = np.copy(self.Xhat)
                    self.Xhat = np.copy(Xhat_new)
                    self.Xhat_data[q, :] = self.Xhat[:, 0]
                    self.U_old[:] = U[[q]]
                    E = self.Y - self.Yhat
                    sqE = np.dot(E.T, E)
                    VN0 = VN0 + sqE
                    k = k + 1
                    VN = VN0 / k
                    self.Y_old = np.copy(self.Y)
                    self.Yhat_old = np.copy(self.Yhat)
                    self.Yhat = np.copy(Yhat_new)
                    # save
                    self.Yhat_data[q] = self.Yhat[0]
                    self.VN_data[q] = VN
                    slot_vn = self.VN_data[q - slot:q]
                    if np.var(slot_vn) > threshold2:  # to identify again
                        self.restart.append(q + 1)
                        break
                    q = q + 1
            q = q + 1
        self.theta = np.asarray(self.theta)  # a list of arrays. if nonstop, use self.Thehat
        # return theta #self.stop, self.restart,

    def peml(self, Y_sys, U):
        """
        for packet loss situation, implemented on centre node
        :param Y_sys: dim=2*N array
        :param U: dim=N array
        :return:
        """
        s_old = np.zeros(1)  # new Y, received observations considering loss
        gamma_old = 1  # data received
        k = 0
        VN0 = 0
        self.Yhat = np.dot(self.Chat_old, self.Xhat)
        # assign theta-hat
        for a in range(self.n):
            self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            self.Ahat_old[self.n - 1, a] = self.Thehat_old[a, 0]
        for b in range(self.n):
            self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
        for h in range(self.n):
            self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
            self.Khat_old[h, 0] = self.Thehat_old[self.n + self.n + h, 0]

        # ---------------PEM iteration-------------------------
        for q in range(self.N):
            if Y_sys[1, [q]] == 1:
                gamma = 1
                self.Y[0] = Y_sys[0, [q]]  # read in transmission
            else:
                gamma = 0
            s = gamma * self.Y + (1 - gamma) * self.Yhat

            for i0 in range(self.n):  # derivative of A
                self.Xhatdot0[self.n - 1, i0] = self.Xhat_old[i0, 0]
            for i1 in range(self.n):  # of B
                self.Xhatdot0[i1, self.n + i1] = gamma_old * self.U_old[0]
            for i2 in range(self.n):  # of K
                self.Xhatdot0[i2, self.n + self.n + i2] = gamma_old * (s_old - self.Yhat_old)

            Xhatdot = self.Xhatdot0 + np.dot(self.Ahat_old, self.Xhatdot_old) - gamma_old * np.dot(self.Khat_old,
                                                                                                   self.Psi_old2.T)
            Psi_old = np.dot(self.Chat_old, Xhatdot).T
            J = self.I + np.dot(np.dot(Psi_old.T, self.P_old2), Psi_old)
            P_old = self.P_old2 - np.dot(np.dot(np.dot(self.P_old2, Psi_old), np.linalg.pinv(J)),
                                         np.dot(Psi_old.T, self.P_old2))
            # self.Yhat = np.dot(self.Chat_old, self.Xhat)
            self.Thehat = self.Thehat_old + np.dot(np.dot(P_old, Psi_old), (s - self.Yhat))
            # update thehat
            for a in range(self.n):
                self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            for b in range(self.n):
                self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
            for h in range(self.n):
                self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]

            Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat, gamma * U[q]) + gamma * np.dot(self.Khat,
                                                                                                       (s - self.Yhat))
            Yhat_new = np.dot(self.Chat, Xhat_new)
            # update every parameter which is time-variant
            self.Xhat_old = np.copy(self.Xhat)
            self.Xhat = np.copy(Xhat_new)
            self.Xhat_data[q, :] = self.Xhat[:, 0]
            self.Ahat_old = np.copy(self.Ahat)
            self.Khat_old = np.copy(self.Khat)
            self.Xhatdot_old = np.copy(Xhatdot)
            self.P_old2 = np.copy(P_old)
            self.Psi_old2 = np.copy(Psi_old)
            self.U_old[:] = U[q]
            self.Thehat_old = np.copy(self.Thehat)
            # squared prediction errors
            E = self.Y - self.Yhat
            sqE = np.dot(E.T, E)
            VN0 = VN0 + sqE
            k = k + 1
            VN = VN0 / k
            gamma_old = gamma
            s_old = np.copy(s)
            self.Yhat_old = np.copy(self.Yhat)
            self.Yhat = np.copy(Yhat_new)
            # save data
            self.Yhat_data[q] = self.Yhat[0]
            self.VN_data[q] = VN
        return self.Yhat_data, self.VN_data

    def pkf(self, y, u, thres_pkf):
        self.A = np.eye(n)
        self.B = np.zeros((n, 1))
        self.C = np.zeros((1, n))
        self.x = np.zeros((n, 1))
        self.K = np.zeros((n, 1))
        self.xlp = np.zeros((n, 1))
        # prior estimate
        x_prior = np.dot(self.A, self.x) + np.dot(self.B, u)
        # posteriori estimate
        yr = y - np.dot(self.C, x_prior)
        x_post = x_prior + np.dot(self.K, yr)
        self.x = np.copy(x_post)
        yhat = np.dot(self.C, self.x)
        # linear predictor, synchronous in both leaf and head node
        self.xlp = np.dot(self.A, self.xlp) + np.dot(self.B, u)
        yp = np.dot(self.C, self.xlp)
        # prediction error of the predictor
        error = yp - yhat
        # self.errors.append(error)
        if abs(error) > thres_pkf:  # transmit
            self.trans.append(1)
            self.recon.append(yhat[0])
            self.xlp = np.copy(self.x)
        else:  # not transmit
            self.trans.append(0)
            self.recon.append(yp)

    def pemt_pkf(self, Y_sys, U, thres_pkf, threshold1=0, threshold2=0, slot=100):
        """
        on-ff PEM and PKF trans, variance threshold_1_2=0 nonstop
        :param Y_sys: N sequence, system raw measurements
        :param U: N*1 array, input raw data
        :param threshold1: variance of a window of VN
        :param threshold2: > threshold1
        :param slot: window of VN
        :return:
        """
        self.Yhat = np.dot(self.Chat_old, self.Xhat)
        self.recon = []  # reconstruction in head node
        self.trans = []  # transmitted = 1, for transmit rate
        self.errors = []
        self.rate = []
        self.bits = []
        k = 0
        VN0 = 0
        # assign theta-hat
        for a in range(self.n):
            self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            self.Ahat_old[self.n - 1, a] = self.Thehat_old[a, 0]
        for b in range(self.n):
            self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
        for h in range(self.n):
            self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
            self.Khat_old[h, 0] = self.Thehat_old[self.n + self.n + h, 0]
        # ---------------PEM iteration-------------------------
        q = 0
        while q < self.N:
            self.Y[:] = Y_sys[q]  # read in transmission
            for i0 in range(self.n):  # derivative of A
                self.Xhatdot0[self.n - 1, i0] = self.Xhat_old[i0, 0]
            for i1 in range(self.n):  # of B
                self.Xhatdot0[i1, self.n + i1] = self.U_old[0]
            for i2 in range(self.n):  # of K
                self.Xhatdot0[i2, self.n + self.n + i2] = self.Y_old - self.Yhat_old

            Xhatdot = self.Xhatdot0 + np.dot(self.Ahat_old, self.Xhatdot_old) - np.dot(self.Khat_old[:, [0]],
                                                                                       self.Psi_old2.T)
            Psi_old = np.dot(self.Chat_old, Xhatdot).T
            J = self.I + np.dot(np.dot(Psi_old.T, self.P_old2), Psi_old)
            P_old = self.P_old2 - np.dot(np.dot(np.dot(self.P_old2, Psi_old), np.linalg.pinv(J)),
                                         np.dot(Psi_old.T, self.P_old2))
            # self.Yhat = np.dot(self.Chat_old, self.Xhat)
            self.Thehat = self.Thehat_old + np.dot(np.dot(P_old, Psi_old), (self.Y - self.Yhat))
            # update thehat
            for a in range(self.n):
                self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            for b in range(self.n):
                self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
            for h in range(self.n):
                self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
            Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat, U[q]) + self.Khat * (self.Y - self.Yhat)
            Yhat_new = np.dot(self.Chat, Xhat_new)
            # update every parameter which is time-variant
            self.Xhat_old = np.copy(self.Xhat)
            self.Xhat = np.copy(Xhat_new)
            self.Xhat_data[q, :] = self.Xhat[:, 0]
            self.Ahat_old = np.copy(self.Ahat)
            self.Khat_old = np.copy(self.Khat)
            self.Xhatdot_old = np.copy(Xhatdot)
            self.Psi_old2 = np.copy(Psi_old)
            self.U_old[:] = U[q]
            self.Thehat_old = np.copy(self.Thehat)
            self.P_old2 = np.copy(P_old)
            # squared prediction errors
            E = self.Y - self.Yhat
            sqE = np.dot(E.T, E)
            VN0 = VN0 + sqE
            k = k + 1
            VN = VN0 / k
            self.Y_old = np.copy(self.Y)
            self.Yhat_old = np.copy(self.Yhat)
            self.Yhat = np.copy(Yhat_new)
            # save data-----------------
            self.Yhat_data[q] = self.Yhat[0]
            self.VN_data[q] = VN
            slot_vn = self.VN_data[q - slot:q]  # check if to stop
            if np.var(slot_vn) < threshold1:
                self.stop.append(q)
                #  initialize PKF
                self.A = np.copy(self.Ahat)
                self.B = np.copy(self.Bhat)
                self.C = np.copy(self.Chat)
                self.K = np.copy(self.Khat)
                self.x = np.copy(self.Xhat)
                self.xlp = np.copy(self.Xhat)
                q = q + 1
                while q < self.N:  # only KF-predictor
                    self.Y[:] = Y_sys[q]
                    u = U[q]
                    self.pkf(self.Y, u, thres_pkf)
                    Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat, U[q]) + self.Khat * (self.Y - self.Yhat)
                    Yhat_new = np.dot(self.Chat, Xhat_new)
                    self.Xhat_old = np.copy(self.Xhat)
                    self.Xhat = np.copy(Xhat_new)
                    self.Xhat_data[q, :] = self.Xhat[:, 0]
                    self.U_old[:] = U[[q]]
                    E = self.Y - self.Yhat
                    sqE = np.dot(E.T, E)
                    VN0 = VN0 + sqE
                    k = k + 1
                    VN = VN0 / k
                    self.Y_old = np.copy(self.Y)
                    self.Yhat_old = np.copy(self.Yhat)
                    self.Yhat = np.copy(Yhat_new)
                    # save
                    self.Yhat_data[q] = self.Yhat[0]
                    self.VN_data[q] = VN
                    slot_vn = self.VN_data[q - slot:q]
                    if np.var(slot_vn) > threshold2:  # to identify again
                        self.restart.append(q + 1)
                        break
                    q = q + 1
            q = q + 1
        self.rate = np.count_nonzero(self.trans) / len(self.recon)  # transmission rate
        self.bits = np.dot(self.rate, np.size(self.recon))
        # self.errors = np.asarray(self.errors)
