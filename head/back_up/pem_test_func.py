"""
single-output pem of canonical form 
Functions:
pemt -- with threshold to stop and restart, redefined N for batch-calculation, set thres=0, nonstop PEM
peml -- packet-loss reconstruction
pemt_pkf -- stop and transmit using PKF
"""
import numpy as np
import torch


class PEM(object):  #
    def __init__(self, n, t, N):
        self.n = n  # dim of X
        self.t = t  # dim of Theta
        self.N = N  # total data size

        self.Xhat_data = np.zeros((N, n))  # collect state estimates
        self.Yhat_data = np.zeros(N)  # collect prediction
        # self.Yhat_data = []
        self.VN_data = np.zeros(N)  # prediction mean squared errors
        self.Xhat = np.zeros((n, 1))
        self.Xhat_old = np.zeros((n, 1))
        self.Ahat = np.eye(n, n, 1)
        self.Ahat_old = np.eye(n, n, 1)
        self.Bhat = np.zeros((n, 1))
        self.Chat = np.eye(1, n)  # [1, 0]
        self.Chat_old = np.eye(1, n)  # [1, 0]
        self.Khat = np.zeros((n, 1))
        self.Khat_old = np.zeros((n, 1))
        self.Y = np.zeros(1)
        self.Y_old = np.zeros(1)
        self.Yhat = np.zeros(1)
        self.Yhat_old = np.zeros(1)
        self.U_old = np.zeros(1)
        self.Thehat = np.zeros((t, 1))
        self.Thehat_old = np.zeros((t, 1))
        self.P_old2 = np.eye(t, t)
        self.Psi_old2 = np.eye(t, 1)
        self.I = np.eye(1)
        self.Xhatdot0 = np.zeros((n, t))
        self.Xhatdot_old = np.zeros((n, t))

    # ------------ ---- - recent --------- -------------------
    def forward(self, Y_sys, U):
        k = 0
        VN0 = 0
        self.N = Y_sys.shape[0]
        self.Xhat_data = np.zeros((self.N, self.n))  # collect state estimates
        self.Yhat_data = np.zeros(self.N)  # collect prediction
        self.VN_data = np.zeros(self.N)  # prediction mean squared errors
        self.Yhat_old = np.dot(self.Chat_old, self.Xhat_old)

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
            # Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat, U[q]) + self.Khat * (self.Y - self.Yhat)
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
        return self.Xhat_data

    # ------------- ----------- test , deletable ---------------------- --------------
    def state(self, Y_sys, U):   # only Kalman predictor, no theta updating
        self.N = U.shape[0]
        self.Yhat_data = np.zeros(self.N)
        q = 0
        while q < self.N:
            self.Y[:] = Y_sys[q]
            Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q]+ self.Khat * (self.Y - self.Yhat)
            Yhat_new = np.dot(self.Chat, Xhat_new)
            self.Xhat = np.copy(Xhat_new)
            self.Yhat = np.copy(Yhat_new)
            self.Yhat_data[q] = Yhat_new[0]
            q = q + 1

    def kf(self, Y_sys, U):  # only Kalman filter, no theta updating, for upload from another theta
        self.N = U.shape[0]
        self.Xhat = self.Xhat_old
        self.Thehat = self.Thehat_old
        self.Xhat_data = np.zeros((self.N, self.n))
        self.Yhat_data = np.zeros(self.N)
        self.VN_data = np.zeros(self.N)

        k = 0
        VN0 = 0
        for a in range(self.n):
            self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
        for b in range(self.n):
            self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
        for h in range(self.n):
            self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
        q = 0
        while q < self.N:
            self.Y[:] = Y_sys[q]

            Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (self.Y - self.Yhat_old)
            Yhat_new = np.dot(self.Chat, Xhat_new)
            # update every parameter which is time-variant
            self.Xhat_old = np.copy(self.Xhat)
            self.Xhat = np.copy(Xhat_new)
            self.Xhat_data[q, :] = self.Xhat[:, 0]
            self.U_old[:] = U[q]
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
            q = q + 1

    def pemfullprd(self, Y_sys, U, r):  # wrong???
        """
        fully predictive; r-step ahead, theta updating
        """
        self.N = Y_sys.shape[0]
        # self.Xhat_data = np.zeros((self.N, self.n))
        self.Xhat = np.copy(self.Xhat_old)
        self.Thehat = np.copy(self.Thehat_old)
        self.Yhat_data = np.zeros(self.N)  # collect prediction
        self.VN_data = np.zeros(self.N)
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
            if q % r == 0:
                self.Y[:] = Y_sys[q]
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
                Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (self.Y - self.Yhat)
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
                q = q + 1

            else:
                # self.Y[:] = self.Yhat_old[:]
                # Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (self.Y - self.Yhat)
                Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (self.Yhat - self.Yhat_old)
                Yhat_new = np.dot(self.Chat, Xhat_new)
                # update every parameter which is time-variant
                self.Xhat_old = np.copy(self.Xhat)
                self.Xhat = np.copy(Xhat_new)
                self.Xhat_data[q, :] = self.Xhat[:, 0]
                self.U_old[:] = U[q]
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
                q = q + 1

    def fullprd(self, Y_sys, U, r):  # wrong
        """
        fully predictive; r-step ahead, theta not updating

        """
        self.N = U.shape[0]
        # self.Xhat_data = np.zeros((self.N, self.n))
        self.Xhat_data = self.Xhat_old
        self.Thehat = self.Thehat_old
        self.Yhat_data = np.zeros(self.N)  # collect prediction
        self.VN_data = np.zeros(self.N)
        k = 0
        VN0 = 0
        # assign theta-hat
        for a in range(self.n):
            self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
        for b in range(self.n):
            self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
        for h in range(self.n):
            self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
        # ---------------PEM iteration-------------------------
        q = 0
        while q < self.N:
            if q % r == 0:
                self.Y[:] = Y_sys[q]
            else:
                self.Y[:] = self.Yhat_old[:]

            # Xhat_new = np.dot(self.Ahat, self.Xhat) * self.Bhat * U[q] + self.Khat * (self.Y - self.Yhat) # Ax*Bu

            Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (self.Y - self.Yhat)
            Yhat_new = np.dot(self.Chat, Xhat_new)
            # update every parameter which is time-variant
            self.Xhat_old = np.copy(self.Xhat)
            self.Xhat = np.copy(Xhat_new)
            self.Xhat_data[q, :] = self.Xhat[:, 0]
            self.U_old[:] = U[q]
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
            q = q + 1

    def pemtmul(self, Y_sys, U, threshold1=0, threshold2=0, slot=100):  # not math proved
        """
        AX * Bu wrong ?????????????????
        on-off PEM, variance threshold=0 nonstop
        :param Y_sys: size-N sequence, system raw measurements
        :param U: N*1 array, input raw data
        :param threshold1: variance of a window of VN
        :param threshold2: > threshold1
        :param slot: window of VN
        :return:
        """
        self.N = Y_sys.shape[0]
        self.Xhat_data = np.zeros((self.N, self.n))
        self.Yhat_data = np.zeros(self.N)  # collect prediction
        self.VN_data = np.zeros(self.N)
        k = 0
        VN0 = 0
        # for on-off points
        self.stop = []
        self.restart = []
        theta = []  # collect identified parameters at stops
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
            Xhat_new = np.dot(self.Ahat, self.Xhat) * np.dot(self.Bhat, U[q]) + self.Khat * (self.Y - self.Yhat)
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
                theta.append(self.Thehat[:, 0])
                q = q + 1
                while q < self.N:  # only KF-predictor
                    self.Y[:] = Y_sys[q]
                    Xhat_new = np.dot(self.Ahat, self.Xhat) * np.dot(self.Bhat, U[q]) + self.Khat * (self.Y - self.Yhat)
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
        theta = np.asarray(theta)  # a list of arrays. if nonstop, use self.Thehat
        return self.stop, self.restart  # , theta

    def pem_offset(self, Y_sys, U):
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

        self.Xhat_data = np.zeros((self.N, self.n))
        self.Yhat_data = np.zeros(self.N)  # collect prediction
        self.VN_data = np.zeros(self.N)
        k = 0
        VN0 = 0
        # for on-off points
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
            Psi_old = np.dot(self.Chat_old, Xhatdot).T  # + 1  # theta6
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
            # Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat, U[q]) + self.Khat * (self.Y - self.Yhat)
            Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (
                        self.Y - self.Yhat)  # + np.array([[self.Thehat[-2, 0]], [self.Thehat[-1, 0]]])
            Yhat_new = np.dot(self.Chat, Xhat_new)  # + self.Thehat[6, 0]
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

            q = q + 1
        self.theta = np.asarray(self.theta)  # a list of arrays. if nonstop, use self.Thehat
        # return theta #self.stop, self.restart,

    # --------- early ------------
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
        # for on-off points
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
            # self.Yhat = np.dot(self.Chat_old, self.Xhat)
            self.Thehat = self.Thehat_old + np.dot(np.dot(P_old, Psi_old), (self.Y - self.Yhat))
            # update thehat
            for a in range(self.n):
                self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            for b in range(self.n):
                self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
            for h in range(self.n):
                self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
            # Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat, U[q]) + self.Khat * (self.Y - self.Yhat)
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
        for packet loss situation, implemented on centre
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


# ------------- ----------- test , deletable ---------------
# class PEMFull:  # about C???
#     def __init__(self, n, t, N):
#         self.n = n  # dim of X
#         self.t = t  # dim of Theta
#         self.N = N  # total data size
#
#         self.Xhat_data = np.zeros((N, n))  # collect state estimates
#         self.Yhat_data = np.zeros(N)  # collect prediction
#         self.VN_data = np.zeros(N)  # prediction mean squared errors
#         self.Xhat = np.zeros((n, 1))
#         self.Xhat_old = np.zeros((n, 1))
#         self.Ahat = np.zeros((n, n))
#         self.Ahat_old = np.zeros((n, n))
#         self.Bhat = np.zeros((n, 1))
#         self.Chat = np.zeros((1, n))
#         self.Chat_old = np.zeros((1, n))
#         self.Khat = np.zeros((n, 1))
#         self.Khat_old = np.zeros((n, 1))
#         self.Y = np.zeros(1)
#         self.Y_old = np.zeros(1)
#         self.Yhat = np.zeros(1)
#         self.Yhat_old = np.dot(self.Chat_old, self.Xhat)
#         self.U_old = np.zeros(1)
#         self.Thehat = np.zeros((t, 1))
#         self.Thehat_old = np.zeros((t, 1))
#         self.P_old2 = np.eye(t, t)
#         self.Psi_old2 = np.eye(t, 1)
#         self.I = np.eye(1)
#         self.Xhatdot0 = np.zeros((n, t))
#         self.Xhatdot_old = np.zeros((n, t))
#
#     def forward(self, Y_sys, U):
#         self.N = Y_sys.shape[0]
#         self.Xhat_data = np.zeros((self.N, self.n))
#         self.Yhat_data = np.zeros(self.N)  # collect prediction
#         self.VN_data = np.zeros(self.N)
#         k = 0
#         VN0 = 0
#         # assign theta-hat
#         for a in range(self.n):
#             for a1 in range(self.n):
#                 self.Ahat[a, a1] = self.Thehat[a * self.n + a1, 0]
#                 self.Ahat_old[a, a1] = self.Thehat_old[a * self.n + a1, 0]
#         for b in range(self.n):
#             self.Bhat[b, 0] = self.Thehat[self.n * self.n + b, 0]
#         for h in range(self.n):
#             self.Khat[h, 0] = self.Thehat[self.n * self.n + self.n + h, 0]
#             self.Khat_old[h, 0] = self.Thehat_old[self.n * self.n + self.n + h, 0]
#         for c in range(self.n):
#             self.Chat[0, c] = self.Thehat[self.n * self.n + self.n + self.n + c, 0]
#             self.Chat_old[0, c] = self.Thehat_old[self.n * self.n + self.n + self.n + c, 0]
#         # ---------------PEM iteration-------------------------
#         q = 0
#         while q < self.N:
#             self.Y[:] = Y_sys[q]  # read in transmission
#
#             for i0 in range(self.n):
#                 for i01 in range(self.n):
#                     self.Xhatdot0[i0, i0 * self.n + i01] = self.Xhat_old[i01, 0]
#             for i1 in range(self.n):  # of B
#                 self.Xhatdot0[i1, self.n * self.n + i1] = self.U_old[0]
#             for i2 in range(self.n):  # of K
#                 self.Xhatdot0[i2, self.n * self.n + self.n + i2] = self.Y_old - self.Yhat_old
#
#             Xhatdot = self.Xhatdot0 + np.dot(self.Ahat_old, self.Xhatdot_old) - np.dot(self.Khat_old[:, [0]],
#                                                                                        self.Psi_old2.T)
#             dc = np.zeros((self.t, 1))
#             dc[8:10, 0] = self.Xhat_old[:, 0]
#             Psi_old = np.dot(self.Chat_old, Xhatdot).T + dc
#             J = self.I + np.dot(np.dot(Psi_old.T, self.P_old2), Psi_old)
#             P_old = self.P_old2 - np.dot(np.dot(np.dot(self.P_old2, Psi_old), np.linalg.pinv(J)),
#                                          np.dot(Psi_old.T, self.P_old2))
#             # self.Yhat = np.dot(self.Chat_old, self.Xhat)
#             self.Thehat = self.Thehat_old + np.dot(np.dot(P_old, Psi_old), (self.Y - self.Yhat))
#             # update thehat
#             for a in range(self.n):
#                 for a1 in range(self.n):
#                     self.Ahat[a, a1] = self.Thehat[a * self.n + a1, 0]
#             for b in range(self.n):
#                 self.Bhat[b, 0] = self.Thehat[self.n * self.n + b, 0]
#             for h in range(self.n):
#                 self.Khat[h, 0] = self.Thehat[self.n * self.n + self.n + h, 0]
#             for c in range(self.n):
#                 self.Chat[0, c] = self.Thehat[self.n * self.n + self.n + self.n + c, 0]
#             # Xhat_new = np.dot(self.Ahat, self.Xhat) + np.dot(self.Bhat, U[q]) + self.Khat * (self.Y - self.Yhat)
#             Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * U[q] + self.Khat * (
#                     self.Y - self.Yhat)  # + np.array([[self.Thehat[-2, 0]], [self.Thehat[-1, 0]]])
#             Yhat_new = np.dot(self.Chat, Xhat_new)  # + self.Thehat[6, 0]
#             # update every parameter which is time-variant
#             self.Xhat_old = np.copy(self.Xhat)
#             self.Xhat = np.copy(Xhat_new)
#             self.Xhat_data[q, :] = np.copy(self.Xhat[:, 0])
#             self.Ahat_old = np.copy(self.Ahat)
#             self.Khat_old = np.copy(self.Khat)
#             self.Chat_old = np.copy(self.Chat)
#             self.Xhatdot_old = np.copy(Xhatdot)
#             self.Psi_old2 = np.copy(Psi_old)
#             self.U_old[:] = np.copy(U[q])
#             self.Thehat_old = np.copy(self.Thehat)
#             self.P_old2 = np.copy(P_old)
#             # squared prediction errors
#             E = self.Y - self.Yhat
#             sqE = np.dot(E.T, E)
#             VN0 = VN0 + sqE
#             k = k + 1
#             VN = VN0 / k
#             self.Y_old = np.copy(self.Y)
#             self.Yhat_old = np.copy(self.Yhat)
#             self.Yhat = np.copy(Yhat_new)
#             # save data-----------------
#             self.Yhat_data[q] = self.Yhat[0]
#             self.VN_data[q] = VN
#             q = q + 1


class PEM_increment(object):
    """
    A,B, K, single measurement to update AB
    """

    def __init__(self, n, t):
        self.n = n
        self.t = t

        self.Xhat = np.zeros((self.n, 1))
        self.Xhat_old = np.zeros((self.n, 1))
        self.Ahat = np.eye(self.n, self.n, 1)
        self.Ahat_old = np.eye(self.n, self.n, 1)
        self.Bhat = np.zeros((self.n, 1))
        self.Chat = np.array([[1, 0]])
        self.Chat_old = np.array([[1, 0]])
        self.Khat = np.zeros((self.n, 1))
        self.Khat_old = np.zeros((self.n, 1))
        self.Y_old = np.zeros(1)
        self.Yhat_old = np.zeros(1)
        self.U_old = np.zeros(1)
        self.Thehat = np.zeros((self.t, 1))
        self.Thehat_old = np.zeros((self.t, 1))
        self.P_old2 = np.eye(self.t, self.t)
        self.Psi_old2 = np.eye(self.t, 1)
        self.I = np.eye(1)
        self.Xhatdot0 = np.zeros((self.n, self.t))
        self.Xhatdot_old = np.zeros((self.n, self.t))
        # initialize outside forward, only the first is random, rest iterative
        for a in range(self.n):
            self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
            self.Ahat_old[self.n - 1, a] = self.Thehat_old[a, 0]
        for b in range(self.n):
            self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
        for h in range(self.n):
            self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]
            self.Khat_old[h, 0] = self.Thehat_old[self.n + self.n + h, 0]

        self.Xhat_data: List[np.array] = []
        self.J_list = []

    def forward(self, yi, ui):  # single element, not full array! xhat: prediction of NN, x

        # self.Xhat = x
        self.Yhat = np.dot(self.Chat_old, self.Xhat)

        for i0 in range(self.n):  # derivative of A
            self.Xhatdot0[self.n - 1, i0] = self.Xhat_old[i0, 0]
        for i1 in range(self.n):  # of B
            self.Xhatdot0[i1, self.n + i1] = self.U_old[0]
        for i2 in range(self.n):  # of K
            self.Xhatdot0[i2, self.n + self.n + i2] = self.Y_old - self.Yhat_old

        Xhatdot = self.Xhatdot0 + np.dot(self.Ahat_old, self.Xhatdot_old) - np.dot(self.Khat_old[:, [0]], self.Psi_old2.T)

        Psi_old = np.dot(self.Chat_old, Xhatdot).T
        J = self.I + np.dot(np.dot(Psi_old.T, self.P_old2), Psi_old)
        self.J_list.append(J)
        P_old = self.P_old2 - np.dot(np.dot(np.dot(self.P_old2, Psi_old), np.linalg.pinv(J)),np.dot(Psi_old.T, self.P_old2))

        # self.Yhat = np.dot(self.Chat_old, self.Xhat)
        self.Thehat = self.Thehat_old + np.dot(np.dot(P_old, Psi_old), (yi - self.Yhat))
        # update thehat
        for a in range(self.n):
            self.Ahat[self.n - 1, a] = self.Thehat[a, 0]
        for b in range(self.n):
            self.Bhat[b, 0] = self.Thehat[self.n + b, 0]
        for h in range(self.n):
            self.Khat[h, 0] = self.Thehat[self.n + self.n + h, 0]

        Xhat_new = np.dot(self.Ahat, self.Xhat) + self.Bhat * ui + self.Khat * (yi - self.Yhat)
        Yhat_new = np.dot(self.Chat, Xhat_new)
        # update every parameter which is time-variant
        self.Xhat_old = np.copy(self.Xhat)
        self.Xhat = np.copy(Xhat_new)
        self.Xhat_data.append(self.Xhat[:, 0])  # ???
        self.Ahat_old = np.copy(self.Ahat)
        self.Khat_old = np.copy(self.Khat)
        self.Xhatdot_old = np.copy(Xhatdot)
        self.Psi_old2 = np.copy(Psi_old)
        self.U_old[:] = np.copy(ui)
        self.Thehat_old = np.copy(self.Thehat)
        self.P_old2 = np.copy(P_old)
        self.Y_old = np.copy(yi)
        self.Yhat_old = np.copy(self.Yhat)
        self.Yhat = np.copy(Yhat_new)





class PEM_batch(object):
    """
    canonical A,B, K
    """

    def __init__(self, width, n=2, t=6):
        self.n = n
        self.t = t
        self.width = width  # batch_num
        self.Chat = np.zeros((self.width, 1, self.n))
        self.Chat[:, ...] = np.array([[1, 0]])
        self.Chat_old = np.copy(self.Chat)
        self.P_old2 = np.zeros((self.width, self.t, self.t))
        self.Psi_old2 = np.zeros((self.width, self.t, 1))

        self.Xhat = np.zeros((self.width, self.n, 1))
        self.Xhat_old = np.zeros((self.width, self.n, 1))
        self.Ahat = np.zeros((self.width, self.n, self.n))
        self.Ahat_old = np.zeros((self.width, self.n, self.n))
        self.Ahat[:, [0], :] = np.array([0, 1])
        self.Ahat_old[:, [0], :] = np.array([0, 1])
        self.Bhat = np.zeros((self.width, self.n, 1))

        self.Khat = np.zeros((self.width, self.n, 1))
        self.Khat_old = np.zeros((self.width, self.n, 1))

        self.Y_old = np.zeros((self.width, 1, 1))
        self.Yhat = np.zeros((self.width, 1, 1))  # np.matmul(self.Chat_old, self.Xhat)  #
        self.Yhat_old = np.zeros((self.width, 1, 1))
        self.U_old = np.zeros((self.width, 1, 1))
        self.Thehat = np.zeros((self.width, self.t, 1))
        self.Thehat_old = np.zeros((self.width, self.t, 1))

        self.I = np.ones(1)
        self.Xhatdot0 = np.zeros((self.width, self.n, self.t))
        self.Xhatdot_old = np.zeros((self.width, self.n, self.t))

        for a in range(self.n):
            self.Ahat[:, self.n - 1, a] = self.Thehat[:, a, 0]
            self.Ahat_old[:, self.n - 1, a] = self.Thehat_old[:, a, 0]
        for b in range(self.n):
            self.Bhat[:, b, 0] = self.Thehat[:, self.n + b, 0]
        for h in range(self.n):
            self.Khat[:, h, 0] = self.Thehat[:, self.n + self.n + h, 0]
            self.Khat_old[:, h, 0] = self.Thehat_old[:, self.n + self.n + h, 0]

    def forward(self, ui, yi):
        # self.width = yi.shape[0]
        # self.Xhat_data: List[np.array] = []
        yi = yi[:, np.newaxis, :]
        ui = ui[:, np.newaxis, :]
        for i0 in range(self.n):
            self.Xhatdot0[:, self.n - 1, i0] = self.Xhat_old[:, i0, 0]
        for i1 in range(self.n):  # of B
            self.Xhatdot0[:, i1, self.n + i1] = self.U_old[:, 0, 0]
        for i2 in range(self.n):  # of K
            self.Xhatdot0[:, i2, self.n + self.n + i2] = self.Y_old[:, 0, 0] - self.Yhat_old[:, 0, 0]
        Psi_old2T0 = self.Psi_old2.squeeze(2)
        Psi_old2T = Psi_old2T0[:, np.newaxis, :]
        Xhatdot = self.Xhatdot0 + np.matmul(self.Ahat_old, self.Xhatdot_old) - np.matmul(self.Khat_old, Psi_old2T)
        Psi_oldT = np.matmul(self.Chat_old, Xhatdot)
        Psi_old0 = Psi_oldT.squeeze(1)
        Psi_old = Psi_old0[:, :, np.newaxis]
        J = self.I + np.matmul(np.matmul(Psi_oldT, self.P_old2), Psi_old)
        Jinv = np.zeros(J.shape)
        for j in range(self.width):
            Jinv[j, :, :] = np.linalg.pinv(J[j, :, :])


        P_old = self.P_old2 - np.matmul(np.matmul(np.matmul(self.P_old2, Psi_old), Jinv),
                                        np.matmul(Psi_oldT, self.P_old2))
        # self.Yhat = np.dot(self.Chat_old, self.Xhat)
        m1 = np.matmul(np.matmul(P_old, Psi_old), (yi - self.Yhat))
        self.Thehat = np.add(self.Thehat_old, m1)
        # update thehat
        for a in range(self.n):
            self.Ahat[:, self.n - 1, a] = self.Thehat[:, a, 0]
        for b in range(self.n):
            self.Bhat[:, b, 0] = self.Thehat[:, self.n + b, 0]
        for h in range(self.n):
            self.Khat[:, h, 0] = self.Thehat[:, self.n + self.n + h, 0]
        m2 = np.matmul(self.Bhat, ui)
        m3 = np.matmul(self.Khat,(yi - self.Yhat))
        Xhat_new = np.add(np.add(np.matmul(self.Ahat, self.Xhat), m2), m3)
        Yhat_new = np.matmul(self.Chat, Xhat_new)
        # Yhat_new = Yhat_new.squeeze(1)
        # update every parameter which is time-variant
        self.Xhat_old = np.copy(self.Xhat)
        self.Xhat = np.copy(Xhat_new)
        # self.Xhat_data.append(self.Xhat)
        self.Ahat_old = np.copy(self.Ahat)
        self.Khat_old = np.copy(self.Khat)
        self.Xhatdot_old = np.copy(Xhatdot)
        self.Psi_old2 = np.copy(Psi_old)
        self.U_old = np.copy(ui)
        # self.U_old = np.copy(ui[:, 0, :])
        self.Thehat_old = np.copy(self.Thehat)
        self.P_old2 = np.copy(P_old)
        self.Y_old = np.copy(yi)
        self.Yhat_old = np.copy(self.Yhat)
        self.Yhat = np.copy(Yhat_new)
        Xhat = self.Xhat.squeeze(2)
        # save data-----------------
        # self.Xhat_data = np.stack(self.Xhat_data)
        return Xhat  # self.Xhat_data


class PEM_BatchFull(object):
    """
    full A,B, K
    """

    def __init__(self, width, n=2, t=8):
        self.n = n
        self.t = t
        self.width = width
        self.Chat = np.array([[1, 0]])
        self.Chat_old = np.array([[1, 0]])
        self.P_old2 = np.zeros((self.width, self.t, self.t))
        self.Psi_old2 = np.zeros((self.width, self.t, 1))

        self.Xhat = np.zeros((self.width, self.n, 1))
        self.Xhat_old = np.zeros((self.width, self.n, 1))
        self.Ahat = np.zeros((self.width, self.n, self.n))
        self.Ahat_old = np.zeros((self.width, self.n, self.n))
        self.Bhat = np.zeros((self.width, self.n, 1))

        self.Khat = np.zeros((self.width, self.n, 1))
        self.Khat_old = np.zeros((self.width, self.n, 1))

        self.Y_old = np.zeros((self.width, 1, 1))
        self.Yhat = np.zeros((self.width, 1, 1))  # np.matmul(self.Chat_old, self.Xhat)  #
        self.Yhat_old = np.zeros((self.width, 1, 1))
        self.U_old = np.zeros((self.width, 1, 1))
        self.Thehat = np.zeros((self.width, self.t, 1))
        self.Thehat_old = np.zeros((self.width, self.t, 1))

        self.I = np.ones(1)
        self.Xhatdot0 = np.zeros((self.width, self.n, self.t))
        self.Xhatdot_old = np.zeros((self.width, self.n, self.t))

        for a in range(self.n):
            for a1 in range(self.n):
                self.Ahat[:, a, a1] = self.Thehat[:, a * self.n + a1, 0]
                self.Ahat_old[:, a, a1] = self.Thehat_old[:, a * self.n + a1, 0]
        for b in range(self.n):
            self.Bhat[:, b, 0] = self.Thehat[:, self.n * self.n + b, 0]
        for h in range(self.n):
            self.Khat[:, h, 0] = self.Thehat[:, self.n * self.n + self.n + h, 0]
            self.Khat_old[:, h, 0] = self.Thehat_old[:, self.n * self.n + self.n + h, 0]

    def forward(self, ui, yi):

        # self.Xhat_data: List[np.array] = []
        yi = yi[:, np.newaxis, :]
        ui = ui[:, np.newaxis, :]
        for i0 in range(self.n):
            for i01 in range(self.n):
                self.Xhatdot0[:, i0, i0 * self.n + i01] = self.Xhat_old[:, i01, 0]
        for i1 in range(self.n):  # of B
            self.Xhatdot0[:, i1, self.n * self.n + i1] = self.U_old[:, 0, 0]
        for i2 in range(self.n):  # of K
            self.Xhatdot0[:, i2, self.n * self.n + self.n + i2] = self.Y_old[:, 0, 0] - self.Yhat_old[:, 0, 0]
        Psi_old2T0 = self.Psi_old2.squeeze(2)
        Psi_old2T = Psi_old2T0[:, np.newaxis, :]
        Xhatdot = self.Xhatdot0 + np.matmul(self.Ahat_old, self.Xhatdot_old) - np.matmul(self.Khat_old,
                                                                                         Psi_old2T)
        Psi_oldT = np.matmul(self.Chat_old, Xhatdot)
        Psi_old0 = Psi_oldT.squeeze(1)
        Psi_old = Psi_old0[:, :, np.newaxis]
        J = self.I + np.matmul(np.matmul(Psi_oldT, self.P_old2), Psi_old)

        P_old = self.P_old2 - np.matmul(np.matmul(np.matmul(self.P_old2, Psi_old), np.linalg.pinv(J)),
                                        np.matmul(Psi_oldT, self.P_old2))
        # self.Yhat = np.dot(self.Chat_old, self.Xhat)
        self.Thehat = self.Thehat_old + np.matmul(np.matmul(P_old, Psi_old), (yi - self.Yhat))
        # update thehat
        for a in range(self.n):
            for a1 in range(self.n):
                self.Ahat[:, a, a1] = self.Thehat[:, a * self.n + a1, 0]
        for b in range(self.n):
            self.Bhat[:, b, 0] = self.Thehat[:, self.n * self.n + b, 0]
        for h in range(self.n):
            self.Khat[:, h, 0] = self.Thehat[:, self.n * self.n + self.n + h, 0]

        Xhat_new = np.matmul(self.Ahat, self.Xhat) + np.matmul(self.Bhat, ui) + np.matmul(self.Khat,
                                                                                          (yi - self.Yhat))
        Yhat_new = np.matmul(self.Chat, Xhat_new)
        # Yhat_new = Yhat_new.squeeze(1)
        # update
        self.Xhat_old = np.copy(self.Xhat)
        self.Xhat = np.copy(Xhat_new)
        # self.Xhat_data.append(self.Xhat)
        self.Ahat_old = np.copy(self.Ahat)
        self.Khat_old = np.copy(self.Khat)
        self.Xhatdot_old = np.copy(Xhatdot)
        self.Psi_old2 = np.copy(Psi_old)
        self.U_old = np.copy(ui)
        # self.U_old = np.copy(ui[:, 0, :])
        self.Thehat_old = np.copy(self.Thehat)
        self.P_old2 = np.copy(P_old)
        self.Y_old = np.copy(yi)
        self.Yhat_old = np.copy(self.Yhat)
        self.Yhat = np.copy(Yhat_new)
        # save data-----------------
        # self.Xhat_data = np.stack(self.Xhat_data)
        Xhat = self.Xhat.squeeze(2)
        return Xhat  # self.Xhat_data


# class PEM_batchflex(object):
#     """
#     only A,B, K
#     """
#     def __init__(self, n=2, t=8):
#         self.n = n
#         self.t = t
#         self.Chat = np.array([[1, 0]])
#         self.Chat_old = np.array([[1, 0]])
#         self.P_old2 = np.zeros((self.t, self.t))
#         self.Psi_old2 = np.zeros((self.t, 1))
#
#         self.Xhat = np.zeros((self.n, 1))
#         self.Xhat_old = np.zeros((self.n, 1))
#         self.Ahat = np.zeros((self.n, self.n))
#         self.Ahat_old = np.zeros((self.n, self.n))
#         self.Bhat = np.zeros((self.n, 1))
#
#         self.Khat = np.zeros((self.n, 1))
#         self.Khat_old = np.zeros((self.n, 1))
#
#         self.Y_old = np.zeros((1, 1))
#         self.Yhat = np.zeros((1, 1))
#         self.Yhat_old = np.zeros((1, 1))
#         self.U_old = np.zeros((1, 1))
#         self.Thehat = np.zeros((self.t, 1))
#         self.Thehat_old = np.zeros((self.t, 1))
#
#         self.I = np.ones(1)
#         self.Xhatdot0 = np.zeros((self.n, self.t))
#         self.Xhatdot_old = np.zeros((self.n, self.t))
#
#
#         for a in range(self.n):
#             for a1 in range(self.n):
#                 self.Ahat[a, a1] = self.Thehat[a * self.n+a1, 0]
#                 self.Ahat_old[a, a1] = self.Thehat_old[a * self.n+a1, 0]
#         for b in range(self.n):
#             self.Bhat[b, 0] = self.Thehat[self.n * self.n + b, 0]
#         for h in range(self.n):
#             self.Khat[h, 0] = self.Thehat[self.n * self.n + self.n + h, 0]
#             self.Khat_old[h, 0] = self.Thehat_old[self.n * self.n+ self.n + h, 0]
#
#
#     def forward(self, ui, yi):
#         width = len(yi)
#         self.Xhatdot0 = np.zeros((width, self.n, self.t))
#         self.Xhatdot_old = np.zeros((width, self.n, self.t))
#         self.Psi_old20 = np.zeros((width, self.t, 1))
#         self.Psi_old20[:, ...] = self.Psi_old2
#
#         # self.Xhat_old = np.repeat(self.Xhat_old, width, axis=1)
#         # self.Xhat_old = self.Xhat_old.T
#         # self.Xhat_old = self.Xhat_old[:,:, np.newaxis]
#
#         # self.U_old = self.U_old[np.newaxis, :, :]
#         # self.Y_old = self.Y_old[np.newaxis, :, :]
#         # self.Yhat_old = self.Yhat_old[np.newaxis, :, :]
#         # self.Xhat_data: List[np.array] = []
#         yi = yi[:, np.newaxis, :]
#         ui = ui[:, np.newaxis, :]
#         for i0 in range(self.n):
#             for i01 in range(self.n):
#                 self.Xhatdot0[:, i0, i0 * self.n + i01] = self.Xhat_old[:, i01, 0]
#         for i1 in range(self.n):  # of B
#             self.Xhatdot0[:, i1, self.n * self.n + i1] = self.U_old[:, 0, 0]
#         for i2 in range(self.n):  # of K
#             self.Xhatdot0[:, i2, self.n * self.n + self.n + i2] = self.Y_old[:, 0, 0] - self.Yhat_old[:, 0, 0]
#         Psi_old2T0 = self.Psi_old20.squeeze(2)
#         Psi_old2T = Psi_old2T0[:, np.newaxis, :]
#         Xhatdot = self.Xhatdot0 + np.matmul(self.Ahat_old, self.Xhatdot_old) - np.matmul(self.Khat_old, Psi_old2T)
#         Psi_oldT = np.matmul(self.Chat_old, Xhatdot)
#         Psi_old0 = Psi_oldT.squeeze(1)
#         Psi_old = Psi_old0[:, :, np.newaxis]
#         J = self.I + np.matmul(np.matmul(Psi_oldT, self.P_old2), Psi_old)
#
#         P_old = self.P_old2 - np.matmul(np.matmul(np.matmul(self.P_old2, Psi_old), np.linalg.pinv(J)),
#                                          np.matmul(Psi_oldT, self.P_old2))
#         # self.Yhat = np.dot(self.Chat_old, self.Xhat)
#         self.Thehat = self.Thehat_old + np.matmul(np.matmul(P_old, Psi_old), (yi - self.Yhat))
#         # update thehat
#         for a in range(self.n):
#             for a1 in range(self.n):
#                 self.Ahat[:, a, a1] = self.Thehat[:, a * self.n + a1, 0]
#         for b in range(self.n):
#             self.Bhat[:, b, 0] = self.Thehat[:, self.n * self.n + b, 0]
#         for h in range(self.n):
#             self.Khat[:, h, 0] = self.Thehat[:, self.n * self.n + self.n + h, 0]
#
#         Xhat_new = np.matmul(self.Ahat, self.Xhat) + np.matmul(self.Bhat, ui) + np.matmul(self.Khat, (yi - self.Yhat)) # - y_step
#         Yhat_new = np.matmul(self.Chat, Xhat_new)
#         # Yhat_new = Yhat_new.squeeze(1)
#         # update every parameter which is time-variant
#         self.Xhat_old = np.copy(self.Xhat)
#         self.Xhat = np.copy(Xhat_new)
#         # self.Xhat_data.append(self.Xhat)
#         self.Ahat_old = np.copy(self.Ahat)
#         self.Khat_old = np.copy(self.Khat)
#         self.Xhatdot_old = np.copy(Xhatdot)
#         self.Psi_old2 = np.copy(Psi_old)
#         self.U_old = np.copy(ui)
#         # self.U_old = np.copy(ui[:, 0, :])
#         self.Thehat_old = np.copy(self.Thehat)
#         self.P_old2 = np.copy(P_old)
#         self.Y_old = np.copy(yi)
#         self.Yhat_old = np.copy(self.Yhat)
#         self.Yhat = np.copy(Yhat_new)
#             # save data-----------------
#         # self.Xhat_data = np.stack(self.Xhat_data)
#         return self.Xhat #self.Xhat_data


# class PEM_batch(object):
#     """
#     only A,B, K
#     """
#     def __init__(self, n=2, t=8):
#         self.n = n
#         self.t = t
#         self.Chat = np.array([[1, 0]])
#         self.Chat_old = np.array([[1, 0]])
#         self.P_old2 = np.zeros((self.t, self.t))
#         self.Psi_old2 = np.zeros((self.t, 1))
#
#         self.Xhat = np.zeros(( self.n, 1))
#         self.Xhat_old = np.zeros((self.n, 1))
#         self.Ahat = np.zeros((self.n, self.n))
#         self.Ahat_old = np.zeros((self.n, self.n))
#         self.Bhat = np.zeros((self.n, 1))
#
#         self.Khat = np.zeros((self.n, 1))
#         self.Khat_old = np.zeros((self.n, 1))
#
#         self.Y_old = np.zeros((1, 1))
#         self.Yhat = np.zeros((1, 1)) #np.matmul(self.Chat_old, self.Xhat)  #
#         self.Yhat_old = np.zeros((1, 1))
#         self.U_old = np.zeros((1, 1))
#         self.Thehat = np.zeros((self.t, 1))
#         self.Thehat_old = np.zeros((self.t, 1))
#
#         self.I = np.ones(1)
#         self.Xhatdot0 = np.zeros((self.n, self.t))
#         self.Xhatdot_old = np.zeros((self.n, self.t))
#
#         for a in range(self.n):
#             for a1 in range(self.n):
#                 self.Ahat[a, a1] = self.Thehat[a * self.n+a1, 0]
#                 self.Ahat_old[a, a1] = self.Thehat_old[a * self.n+a1, 0]
#         for b in range(self.n):
#             self.Bhat[b, 0] = self.Thehat[self.n * self.n + b, 0]
#         for h in range(self.n):
#             self.Khat[h, 0] = self.Thehat[self.n * self.n + self.n + h, 0]
#             self.Khat_old[h, 0] = self.Thehat_old[self.n * self.n+ self.n + h, 0]
#
#
#     def forward(self, ui, yi):
#         width = yi.shape[0]
#         self.Xhatdot0 = np.zeros((width, self.n, self.t))
#         self.Xhatdot_old = np.zeros((width, self.n, self.t))
#         self.Psi_old20 = np.zeros((width, self.t, 1))
#         self.Psi_old20[:, ...] = self.Psi_old2
#         self.U_old = self.U_old[np.newaxis, :, :]
#         self.Y_old = self.Y_old[np.newaxis, :, :]
#         self.Yhat_old = self.Yhat_old[np.newaxis, :, :]
#
#         # self.Xhat_data: List[np.array] = []
#         yi = yi[:, np.newaxis, :]
#         ui = ui[:, np.newaxis, :]
#
#         for i0 in range(self.n):
#             for i01 in range(self.n):
#                 self.Xhatdot0[:, i0, i0 * self.n + i01] = self.Xhat_old[i01, 0]
#         for i1 in range(self.n):  # of B
#             self.Xhatdot0[:, i1, self.n * self.n + i1] = self.U_old[:, 0, 0]
#         for i2 in range(self.n):  # of K
#             self.Xhatdot0[:, i2, self.n * self.n + self.n + i2] = self.Y_old[:, 0, 0] - self.Yhat_old[:, 0, 0]
#         Psi_old2T0 = self.Psi_old20.squeeze(2)
#
#         Psi_old2T = Psi_old2T0[:, np.newaxis, :]
#         Xhatdot = self.Xhatdot0 + np.matmul(self.Ahat_old, self.Xhatdot_old) - np.matmul(self.Khat_old, Psi_old2T)
#         Psi_oldT = np.matmul(self.Chat_old, Xhatdot)
#         Psi_old0 = Psi_oldT.squeeze(1)
#         Psi_old = Psi_old0[:, :, np.newaxis]
#         J = self.I + np.matmul(np.matmul(Psi_oldT, self.P_old2), Psi_old)
#
#         P_old = self.P_old2 - np.matmul(np.matmul(np.matmul(self.P_old2, Psi_old), np.linalg.pinv(J)),
#                                          np.matmul(Psi_oldT, self.P_old2))
#         # self.Yhat = np.dot(self.Chat_old, self.Xhat)
#         self.Thehat = self.Thehat_old + np.matmul(np.matmul(P_old, Psi_old), (yi - self.Yhat))
#         # update thehat
#         for a in range(self.n):
#             for a1 in range(self.n):
#                 self.Ahat[a, a1] = self.Thehat[a * self.n + a1, 0]
#         for b in range(self.n):
#             self.Bhat[b, 0] = self.Thehat[self.n * self.n + b, 0]
#         for h in range(self.n):
#             self.Khat[h, 0] = self.Thehat[self.n * self.n + self.n + h, 0]
#
#         Xhat_new = np.matmul(self.Ahat, self.Xhat) + np.matmul(self.Bhat, ui) + np.matmul(self.Khat, (yi - self.Yhat)) # - y_step
#         Yhat_new = np.matmul(self.Chat, Xhat_new)
#         # Yhat_new = Yhat_new.squeeze(1)
#         # update every parameter which is time-variant
#         self.Xhat_old = np.copy(self.Xhat)
#         self.Xhat = np.copy(Xhat_new)
#         # self.Xhat_data.append(self.Xhat)
#         self.Ahat_old = np.copy(self.Ahat)
#         self.Khat_old = np.copy(self.Khat)
#         self.Xhatdot_old = np.copy(Xhatdot)
#         self.Psi_old2 = np.copy(Psi_old)
#         self.U_old = np.copy(ui)
#         # self.U_old = np.copy(ui[:, 0, :])
#         self.Thehat_old = np.copy(self.Thehat)
#         self.P_old2 = np.copy(P_old)
#         self.Y_old = np.copy(yi)
#         self.Yhat_old = np.copy(self.Yhat)
#         self.Yhat = np.copy(Yhat_new)
#             # save data-----------------
#         # self.Xhat_data = np.stack(self.Xhat_data)
#         return self.Xhat #self.Xhat_data


# ----- other ---------
def R2(Y_sys, Yhat):
    """
    R-square metrics
    :param Y_sys: size N sequence
    :param Yhat: size N sequence
    :return:
    """

    if Yhat.dtype == torch.Tensor:
        s1 = torch.sum((Y_sys - Yhat) ** 2)
        mean = torch.mean(Y_sys)
        s2 = torch.sum((Y_sys - mean) ** 2)
    else:
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
