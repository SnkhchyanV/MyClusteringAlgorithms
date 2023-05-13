from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import numpy as np


class GMM:
    def __init__(self, k, method='k-means', max_iter=300, tol=1e-6):
        self.k = k
        self.method = method
        self.max_iter = max_iter
        self.mean_arr = None
        self.cov_arr = None
        self.pi_arr = None
        self.gamma_mtrx = None
        self.loss = None
        self.tol = tol

    def init_centers(self, X):
        if self.method == 'random_mean_std':
            mean_arr = np.zeros((self.k, X.shape[1]))
            cov_arr = np.zeros((self.k, X.shape[1], X.shape[1]))

            for i in range(self.k):
                mean_arr[i] = np.random.choice(X[:, 0])
                cov_arr[i] = np.diag(np.random.rand(X.shape[1]))

            pi_arr = np.ones(self.k) / self.k

            return mean_arr, cov_arr, pi_arr
        if self.method == 'random_mean':
            mean_arr = np.zeros((self.k, X.shape[1]))
            cov_arr = np.zeros((self.k, X.shape[1], X.shape[1]))

            for i in range(self.k):
                mean_arr[i] = np.random.choice(X[:, 0])
                cov_arr[i] = np.diag(np.ones(X.shape[1]))

            pi_arr = np.ones(self.k) / self.k

            return mean_arr, cov_arr, pi_arr
        if self.method == 'k-means':
            kmeansM = KMeans(self.k)
            x_i_clusters = kmeansM.fit_predict(X)

            mean_arr = kmeansM.cluster_centers_
            cov_arr = []
            pi_arr = []
            for i in range(self.k):
                X_i = X[x_i_clusters == i]
                cov_arr.append(np.cov(X_i.T))
                pi_arr.append(X_i.shape[0]/X.shape[0])

                # if m is quantity of features, k is number of clusters
                # mean_arr size is k*m
                # cov_arr size is k*m*m
                # pi_arr size is k

            return mean_arr, np.array(cov_arr), np.array(pi_arr)

        if self.method == 'random_divide':
            N = X.shape[0]
            np.random.shuffle(X)
            batch_size = N / self.k
            mean_arr = np.zeros((self.k, X.shape[1]))
            cov_arr = np.zeros((self.k, X.shape[1], X.shape[1]))
            pi_arr = np.ones(self.k) / self.k

            for k in range(self.k):
                batch = X[k * batch_size: (k + 1) * batch_size]
                mean_arr[k] = np.mean(batch, axis=0)
                cov_arr[k] = np.diag(np.var(batch, axis=0))

            return mean_arr, cov_arr, pi_arr
        if self.method == 'random_gammas':
            self.gamma_mtrx = np.random.rand(self.k, X.shape[0])
            self.gamma_mtrx /= np.sum(self.gamma_mtrx, axis=0)
            self.mean_arr = np.zeros((self.k, X.shape[1]))
            self.cov_arr = np.zeros((self.k, X.shape[1], X.shape[1]))
            self.pi_arr = np.zeros((self.k, 1))
            self.__maximization(X)
            return self.mean_arr,self.cov_arr, self.pi_arr


    def fit(self, X):
        self.mean_arr, self.cov_arr, self.pi_arr = self.init_centers(X)
        # self.loss = self.loss(...)
        self.gamma_mtrx = np.zeros((self.k, X.shape[0]))
        for _ in range(self.max_iter):
            self.__expectation(X)
            self.__maximization(X)

           new_loss = self.loss + self.loss_NLL(X, self.mean_arr, self.cov_arr, self.pi_arr)
           if new_loss <=self.tol:
                break
           self.loss = new_loss
            
           # self.loss = self.loss_NLL(X, self.mean_arr, self.cov_arr, self.pi_arr)
           # if self.loss <= self.tol:  # add tolerance comparison
           #     break

    def loss_NLL(self, X, mean_arr, cov_arr, pi_arr):
        lh = np.zeros((self.k, X.shape[0]))
        for k in range(self.k):
            lh[k] = pi_arr[k] * self.__pdf(X, mean_arr[k], cov_arr[k])
            # summ = np.sum((pi_arr * self.__pdf(X, mean_arr, cov_arr)),axis=0)
        l_sum = np.sum(lh, axis=0)
        log_sum = np.sum(np.log(l_sum), axis=0)
        return -log_sum

    def __pdf(self, X, mean, cov):
        proba = multivariate_normal.pdf(X, mean, cov)
        return proba

    def __expectation(self, X):
        for k in range(self.k):
            self.gamma_mtrx[k] = self.pi_arr[k] * self.__pdf(X, self.mean_arr[k], self.cov_arr[k])
        sum_gammas = np.sum(self.gamma_mtrx, axis=0)
        self.gamma_mtrx /= sum_gammas

    def __maximization(self, X):
        Nk = np.sum(self.gamma_mtrx, axis=1)
        for k in range(self.k):
            self.mean_arr[k] = np.sum((self.gamma_mtrx[k].reshape(-1, 1)*X), axis=0)
            self.mean_arr[k] /= Nk[k]

            temp = X - self.mean_arr[k]
            self.cov_arr[k] = np.dot((self.gamma_mtrx[k] * temp.T), temp)
            self.cov_arr[k] /= Nk[k]

            self.pi_arr[k] = Nk[k]/X.shape[0]

    def predict(self, X):
        self.__expectation(X)
        y = np.argmax(self.gamma_mtrx, axis=0)
        return y

    def predict_proba(self, X):
        self.__expectation(X)
        return self.gamma_mtrx.T
