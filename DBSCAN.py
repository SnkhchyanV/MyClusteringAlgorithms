import numpy as np


class DBSCAN:
    def __init__(self, epsilon=0.1, point_q=2):
        self.ep = epsilon
        self.k = point_q
        self.noise_points = set()
        self.core_points = set()
        self.border_points = set()
        self.predicted_labels = None
        self.distance_matrix = None

    def fit(self, X):
        index = np.random.randint(0, X.shape[0] - 1)
        self.core_points.add(index)

        for i in range(X.shape[0]):
            self.noise_points.add(i)

        self.distance_matrix = self.__distances(X)
        for i in self.noise_points:
            for j in self.noise_points:
                if i != j and self.distance_matrix[i, j] < self.ep:
                    self.border_points.add(i)
                    self.border_points.add(j)

        for i in self.border_points.copy():
            neighbors_q = 0
            for j in range(X.shape[0]):
                if self.distance_matrix[i, j] < self.ep:
                    neighbors_q += 1
                    if neighbors_q >= self.k:
                        self.border_points.remove(i)
                        self.core_points.add(i)
                        break

        self.predicted_labels = np.full(X.shape[0], -1)
        k = 0
        for i in self.core_points:
            if self.predicted_labels[i] == -1:
                cluster = self.__clustering(X, i)
                for j in cluster:
                    self.predicted_labels[j] = k
                k += 1

    def __clustering(self, X, i):
        cluster = set()
        queue = set()
        queue.add(i)

        while queue:
            j = queue.pop()
            if j not in cluster:
                cluster.add(j)
                for k in range(X.shape[0]):
                    if self.distance_matrix[j, k] < self.ep:
                        if k in self.core_points:
                            queue.add(k)
                        elif k in self.border_points:
                            cluster.add(k)
        return cluster

    def __distances(self, X):
        n = X.shape[0]
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i][j] = np.sqrt(np.sum((X[i]-X[j])**2))
        return distance_matrix

    def predict(self):
        return self.predicted_labels
