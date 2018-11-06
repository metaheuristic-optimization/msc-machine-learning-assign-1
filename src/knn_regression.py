import numpy as np
import pandas as pd
from src.utils import Utils

"""
    K-Nearest-neighbours for regression
"""
class KNNRegression:

    """
        Constructor with configurable parameters

        datasetFile should be the path to the dataset we want to load.
        k_value is used for selecting n nearest neighbours.
        distanceAlg is the algorithm we will use for calculating the distance between points. Algorithms supported are
            - euclidean
            - manhattan
            - minkowski
        p value is used for the minkowski algorithm
    """
    def __init__(self, datasetFile, k_value, distanceAlg='euclidean', p=1):
        self.readData(datasetFile)
        self.k_value = k_value
        self.distanceAlg = distanceAlg
        self.p = p
        self.utils = Utils()

    """
        Read the data into a data-frame using pandas
    """
    def readData(self, datasetFile):
        self.dataset = pd.read_csv(datasetFile, header=None)

    """
        Predict the data points for each row in the dataset and calculate the accuracy of the model
    """
    def predict(self):
        averages = []
        for index, row in self.dataset.iterrows():
            dist, sorted = self.calculateDistances(self.dataset.values, row.values)

            averages.append(self.calcuclateAverages(sorted))

        epsilon = np.square(self.dataset[12] - averages).sum()
        print('Epsilon is {0}'.format(epsilon))

        # total sum of squares (TSS) = ((y_average - y_predicted) ^2 ).sum()
        average_of_training_data = np.average(self.dataset[12])

        tss = np.square(average_of_training_data-averages).sum()
        print('TSS is {0}'. format(tss))

        accuracy = 1 - (epsilon/tss)
        print('R squared coefficient is {0}'. format(accuracy))
        print('Accuracy of the model is: {0}'.format(accuracy*100))

        return accuracy * 100

    """
       Calculates the distance based on a user provided algorithm. The supported algorithms are 
           - euclidean
           - manhattan
           - minkowski
   """
    def calculateDistances(self, a, b):
        if self.distanceAlg == 'euclidean':
            dist = self.utils.euclideanDistance(a, b)
        elif self.distanceAlg == 'manhattan':
            dist = self.utils.manhattanDistance(a, b)
        elif self.distanceAlg == 'minkowski':
            dist = self.utils.minkowskiDistance(a, b, self.p)

        """
            Sort the indexes of distances from best to worst
        """
        sorted = np.argsort(dist)[np.in1d(np.argsort(dist), np.where(dist), 1)]

        return dist, sorted

    """
        Get th average sum of each of the values in the sorted indexes array
    """
    def calcuclateAverages(self, sorted):

        kClosest = sorted[:self.k_value]
        values = []

        for i in kClosest:
            values.append(self.dataset.values[i][12])

        return np.average(values)


