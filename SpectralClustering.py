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
            self.__affinity_mtrx = self.__init_affinity_rbf(X, self.__kernel_params)

        eig_w, eig_v = np.linalg.eig(self.__affinity_mtrx)
        h_mtrx = np.ones((X.shape[0], 1))
        max_ev_list = set()
        for i in range(self.__n_clusters):
            max_id = eig_w.argmax(axis=0)
            max_ev_list.add(id)
            h_mtrx = np.hstack((h_mtrx, eig_v[max_id].reshape(-1, 1)))
            if max_id in max_ev_list:
                np.delete(eig_w, max_id, axis=1)

        if self.__assign_labels == 'kmeans':
            KMeans_clustering = KMeans(self.__n_clusters, n_init='auto')
            self.labels = KMeans_clustering.fit_predict(X)
        if self.__assign_labels == 'DBSCAN':
            clusterer = DBSCAN()
            self.labels = clusterer.fit_predict(h_mtrx)

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels

    def __init_affinity_rbf(self, X, kernel_params):
        n = X.shape[0]
        affinity_mtrx = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                affinity_mtrx[i, j] = self.__Gaussian_Kernel(X[i], X[j], kernel_params)

        return affinity_mtrx

    def __Gaussian_Kernel(self, xi, xj, sigma):
        diff = np.sum((xi - xj)**2)
        w_i_j = (-diff)/(sigma**2)
        return np.exp(w_i_j)
