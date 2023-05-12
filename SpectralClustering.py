import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


class SpectralClustering:
    def __init__(self, n_clusters=8, *, affinity='rbf', assign_labels='kmeans', kernel_params=None):
        self.__n_clusters = n_clusters
        self.__affinity = affinity
        self.__assign_labels = assign_labels
        self.__kernel_params = kernel_params
        self.__affinity_mtrx = None
        self.labels = None

    def fit(self, X):

        if self.__affinity == 'rbf':
            affinity_mtrx = self.__init_affinity_rbf(X, self.__kernel_params)

        d = np.sum(affinity_mtrx, axis=0)
        L = np.zeros_like(affinity_mtrx)
        L = L - affinity_mtrx
        for i in range(len(d)):
            L[i][i] = d[i]

        self.__affinity_mtrx = L
        eig_w, eig_v = np.linalg.eigh(self.__affinity_mtrx)
        h_mtrx = eig_v[:, 0:self.__n_clusters]

        if self.__assign_labels == 'kmeans':
            KMeans_clustering = KMeans(self.__n_clusters, n_init='auto')
            self.labels = KMeans_clustering.fit_predict(h_mtrx)

        return self

    def fit_predict(self, X):
        return self.fit(X).labels

    def __init_affinity_rbf(self, X, kernel_params):
        n = X.shape[0]
        affinity_mtrx = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                affinity_mtrx[i, j] = self.__Gaussian_Kernel(X[i], X[j], kernel_params)

        return affinity_mtrx

    def __Gaussian_Kernel(self, xi, xj, sigma):
        diff = np.linalg.norm(xi-xj)
        w_i_j = diff**2/(sigma**2)
        return np.exp(-w_i_j)
