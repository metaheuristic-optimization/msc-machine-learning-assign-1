import numpy as np

class Utils:

    def euclideanDistance(self, a, b):
        return np.sqrt(((a - b)**2).sum(-1))

    def manhattanDistance(self, a, b):
        return np.abs(a - b).sum(-1)

    def minkowskiDistance(self, a, b, p_value = 1):
        return np.abs(((a - b) / p_value) / (1/ p_value)).sum(-1)