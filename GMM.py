from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


class GMM:
    def __init__(self, k, method='random_mean_std', max_iter=300, tol=1e-6):
        self.k = k
        self.method = method
        self.max_iter = max_iter

    def init_centers(self, X):
        if self.method == 'random_mean_std':
            pass  # generating K random means and std-s
        if self.method == 'random_mean':
            pass  # generate K random means
        if self.method == 'k-means':
            kmeansM = KMeans(self.k)
            x_i_clusters = kmeansM.fit_predict(X)

            mean_arr = kmeansM.cluster_centers_
            cov_arr = []
            pi_arr = []
            for i in range(self.k):
                X_i = X[x_i_clusters == i]
                cov_arr.append(X_i.T)
                pi_arr.append(X_i.shape[0]/X.shape[0])

                # if m is quantity of features, k is number of clusters
                # mean_arr size is k*m
                # cov_arr size is k*m*m
                # pi_arr size is k

            return mean_arr, np.array(cov_arr), np.array(pi_arr)

            #  pass # generate initial points by KMeans algo
        if self.method == 'random_divide':
            pass  # divide data into K clusters randomly
        if self.method == 'random_gammas':
            pass  # generate random gamma matrix

    def fit(self, X):
        self.mean_arr, self.cov_arr, self.pi_arr = self.init_centers(X)
        self.loss = self.loss(...)

        for _ in range(self.max_iter):
            self.gamma_mtrx = self.expectation(X)
            self.mean_arr, self.cov_arr, self.pi_arr = self.maximization(X)

            loss = self.loss(...)
            if loss == self.loss:  # add tolerance comparison
                break
            self.loss = loss
            self.mean_arr = mean_arr
            self.cov_arr = cov_arr
            self.pi_arr = pi_arr

    def loss(self, X, mean, cov, pi):
        pass

    def pdf(self, x, mean, cov):
        proba = (1 / (cov * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * (((x - mean) / cov) ** 2))
        return proba

    def expectation(self, X):
        n, m = X.shape
        gamma_mtrx = np.zeros((n, self.k))

        for k in range(self.k):
            for i in range(n):
                gamma_mtrx[i][k] = (self.pi_arr[k] * self.pdf(X[i], self.mean_arr[k], self.cov_arr[k]))
                summ = 0
                for j in range(self.k):
                    summ += (self.pi_arr[j] * self.pdf(X[i], self.mean_arr[j], self.cov_arr[j]))
                gamma_mtrx[i][k] /= summ

        return gamma_mtrx

    def maximization(self, X):
        for k in range(self.k):
            Nk = 0
            new_mean_sum = 0
            new_sigma_sum = 0
            for i in range(X.shape[0]):
                Nk += self.gamma_mtrx[i][k]
                new_mean_sum += (self.gamma_mtrx[i][k] * X[i])

            self.mean_arr[k] = new_mean_sum/Nk




        return mean_arr, cov_arr, pi_arr

    def predict(self, X):
        return

    def predict_proba(self, X):
        # return predictions using expectation function
        return