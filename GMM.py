from sklearn.cluster import KMeans
import numpy as np


class GMM:
    def __init__(self, k, method='random_mean_std', max_iter=300, tol=1e-6):
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
        # self.loss = self.loss(...)

        for _ in range(self.max_iter):
            self.gamma_mtrx = self.__expectation(X)
            self.mean_arr, self.cov_arr, self.pi_arr = self.__maximization(X)

            loss = self.loss_NLL(X, self.mean_arr, self.cov_arr, self.pi_arr)
            if loss <= self.tol:  # add tolerance comparison
                break

            self.loss = loss

    def loss_NLL(self, X, mean_arr, cov_arr, pi_arr):
        log_sum = 0
        for i in range(X.shape[0]):
            summ = 0
            for k in range(self.k):
                summ += pi_arr[k] * self.__pdf(X, mean_arr[k], cov_arr[k])
                # summ = np.sum((pi_arr * self.__pdf(X, mean_arr, cov_arr)),axis=0)
            log_sum += np.log(summ)

        return -log_sum

    def __pdf(self, x, mean, cov):
        proba = (1 / (cov * np.sqrt(2 * np.pi))) * np.exp((-1 / 2) * (((x - mean) / cov) ** 2))
        return proba

    def __expectation(self, X):
        n, m = X.shape
        gamma_mtrx = np.zeros((n, self.k))

        for k in range(self.k):
            for i in range(n):
                gamma_mtrx[i][k] = (self.pi_arr[k] * self.__pdf(X[i], self.mean_arr[k], self.cov_arr[k]))
                summ = 0
                for j in range(self.k):
                    summ += (self.pi_arr[j] * self.__pdf(X[i], self.mean_arr[j], self.cov_arr[j]))
                gamma_mtrx[i][k] /= summ

        return gamma_mtrx

    def __maximization(self, X):
        mean_arr = np.zeros((self.k, X.shape[1]))
        cov_arr = np.zeros((self.k, X.shape[1], X.shape[1]))
        pi_arr = np.zeros((self.k, 1))
        Nk = np.sum(self.gamma_mtrx, axis=0)
        Nk /= X.shape[0]
        # mean_arr = np.sum ((self.gamma_mtrx.dot(X)) , axis=0 )
        for k in range(self.k):
            for i in range(X.shape[0]):
                mean_arr[k] += self.gamma_mtrx[i][k]*X[i]
            mean_arr[k] /= Nk[k]

            for i in range(X.shape[0]):
                cov_arr[k] = self.gamma_mtrx[i][k] * ((X[i] - mean_arr[k]).dot((X[i]-mean_arr[k]).T))
            cov_arr[k] /= Nk[k]
            pi_arr[k] = Nk[k]/X.shape[0]

        return mean_arr, cov_arr, pi_arr

    def predict(self, X):
        prediction_proba = self.__expectation(X)
        y = np.argmax(prediction_proba, axis=0)
        return y

    def predict_proba(self, X):
        # return predictions using expectation function
        return self.__expectation(X)
