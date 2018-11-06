import numpy as np

"""
    Utility function
"""
class Utils:

    """
        Calculate euclidean distance
    """
    def euclideanDistance(self, a, b):
        return np.sqrt(((a - b)**2).sum(-1))

    """
        Calculate manhattan distance
    """
    def manhattanDistance(self, a, b):
        return np.abs(a - b).sum(-1)

    """
        Calculate minkowski distance
    """
    def minkowskiDistance(self, a, b, p_value = 1):
        return np.abs(((a - b) / p_value) / (1/ p_value)).sum(-1)