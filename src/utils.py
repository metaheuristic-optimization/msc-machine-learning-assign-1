import numpy as np

class Utils:

    def euclideanDistance(self, a, b):
        return np.sqrt(((a - b)**2).sum(-1))

    def manhattanDistance(self, a, b):
        return np.abs(a - b).sum(-1)

    def minkowskiDistance(self, a, b, p_value = 1):
        return np.abs(((a - b) / p_value) / (1/ p_value)).sum(-1)

    def calculateDistances(self, a, b, algo = 'euclidean', p=1):
        if algo == 'euclidean':
            dist = self.euclideanDistance(a, b)
        elif algo == 'manhattan':
            dist = self.manhattanDistance(a, b)
        elif algo == 'minkowski':
            dist = self.minkowskiDistance(a, b, p)

        sorted = np.argsort(dist)[np.in1d(np.argsort(dist),np.where(dist),1)]

        return dist, sorted
