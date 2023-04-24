import numpy as np
import random


class MyKMeans:
    def __init__(self, n_clusters=8, *, init="random", max_iter=300):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        if self.init == "random":
            self.centroids = self.__init_centers_randomly(X)
        elif self.init == "kmeans++":
            self.centroids = self.__init_centers_KMeanspp(X)

        for _ in range(self.max_iter):
            proba = self.__expectation(X)
            self.__maximization(X, proba)

    def predict(self, X):
        return self.__expectation(X)

    def __expectation(self, X):
        y = np.zeros((X.shape[0], 1))  # this array index is point index, and a value in that index is cluster index
        for i, x in np.enumerate(X):
            # Computing distances of x point from all centroids
            # then writing in y[i] the index of centroid which has minimum distance from given point
            # i here is also the index of value in dataframe
            y[i] = np.argmin(self.__distances(x, self.centroids))
        return y

    def __maximization(self, X, y):
        for i in range(self.n_clusters):
            self.centroids[i] = np.sum(X[y == i], axes=0)

    def __distances(self, point, centroids):
        # Euclidian distances of point from centroids
        distances = []
        for i in range(centroids.shape[0]):
            distances.append(np.sqrt(np.sum((point - centroids[i])**2)))
        return np.array(distances)

    def __init_centers_randomly(self, X):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            while centroids[i] != centroids.any():
                centroids[i] = np.random.choice(X)
        return centroids

    ## Is it right realization ?
    def __init_centers_KMeanspp(self, X):
        centroids = list()
        centroids.append(np.random.choice(X))
        for i in range(self.n_clusters):
            min_square_distances = list()
            for j, x in np.enumerate(X):
                min_square_distances.append((np.min(self.__distances(x, np.array(centroids))))**2)
            sumdx = sum(min_square_distances)
            rnd = random.random()*sumdx
            sum_v = 0
            for k in range(X.shape[0]):
                sum_v += min_square_distances[i]
                if sum_v > rnd:
                    centroids.append(X[i])
                    break

        return np.array(centroids)
