import numpy as np
from hmmlearn import hmm
import scipy
from sklearn.cluster import AgglomerativeClustering
import itertools


class HMMclustering():
    def __init__(self, n_states):
        self.n_states = n_states
        return

    def fit(self, X,n_clusters):
        models = [hmm.GaussianHMM(n_components=self.n_states) for _ in range(len(X))]
        for i in range(len(X)):
            models[i].fit(np.array(X[i])[:, None])

        def distance(i, j):
            return -(models[i].score(X[j][:, None])+models[j].score(X[i][:, None]))/2
        '''
        condensed_distance_matrix = np.array([distance(i, j) for i, j in itertools.combinations(range(len(X)), 2)])
        self.condensed_distance_matrix = condensed_distance_matrix
        '''

        self.distance_matrix = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                k = distance(i, j)
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = k

        self.clustering = AgglomerativeClustering(n_clusters=n_clusters,affinity="precomputed",linkage="single").fit(self.distance_matrix)

    #self.clustering.labels_