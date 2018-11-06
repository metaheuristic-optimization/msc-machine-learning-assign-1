import numpy as np
import pandas as pd
import operator
from src.utils import Utils

"""
    K-Nearest-neighbours for classification
"""
class KNN:

    """
        Set the columns of the pandas data-frame for each of the features of the dataset
    """
    columns = ['bi_rads', 'age', 'shape', 'margin', 'density', 'severity']

    """
        Constructor with configurable parameters
    
        datasetFile should be the path to the dataset we want to load.
        k_value is used for selecting n nearest neighbours.
        distanceAlg is the algorithm we will use for calculating the distance between points. Algorithms supported are
            - euclidean
            - manhattan
            - minkowski
        classificationAlg is the classification algorithm we will use. Currently this class supports 2 methods.
            - vote
            - weighted distance
        p value is used for the minkowski algorithm
    """
    def __init__(self, datasetFile, k_value, distanceAlg='euclidean', classificationAlg='vote', p=1):
        self.readData(datasetFile)
        self.k_value = k_value
        self.distanceAlg = distanceAlg
        self.classificationAlg = classificationAlg
        self.p = p
        self.utils = Utils()

    """
        Read the data into a data-frame using pandas
    """
    def readData(self, datasetFile):
        self.dataset = pd.read_csv(datasetFile, names = self.columns, header=None)

        """
            Small bit of cleaning here as there was an outlier skewing the results. Therefore we will remove
            any value in the 'bi_rads' column that is greater than 5
        """
        self.dataset.drop(self.dataset.index[self.dataset['bi_rads'] > 5], inplace=True)

    """
        This is a wrapper function and entry point for classifying and getting the accuracy of the classification.
    """
    def run(self):
        correct = 0
        incorrect = 0

        """
            Loop through the data frame and get the distance between every row to every other row
        """
        for index, row in self.dataset.iterrows():
            dist, sorted = self.calculateDistances(self.dataset.values, row.values)

            """
                Choose which classification algorithm to use
            """
            if self.classificationAlg == 'vote':
                classification = self.getVoteClassification(sorted, self.dataset.values, self.k_value)
            elif self.classificationAlg == 'weighted':
                classification = self.getWeightedClassification(dist, sorted, self.dataset.values, self.k_value)

            """
                Check if the classification is correct or incorrect
            """
            if classification == row.values[5]:
                correct += 1
            else:
                incorrect += 1

        """
            Calculate the overall accuracy
        """
        accuracy = (correct / (correct + incorrect)) * 100

        return accuracy

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
        sorted = np.argsort(dist)[np.in1d(np.argsort(dist),np.where(dist),1)]

        return dist, sorted

    """
        Calculate the classification based on a vote of k values.
        
        The majority vote will be used as the classification
        
        If there is a tie we will assume that the first classification is correct
    """
    def getVoteClassification(self, sorted, dataset, k_value):
        votes = {}
        kClosest = sorted[:k_value]

        for i in kClosest:
            key = dataset[i][5]
            if not key in votes:
                votes[key] = 1
            else:
                votes[key] += 1

        return max(votes.items(), key=operator.itemgetter(1))[0]

    """
        Calculate the classification based on a weighted distance. The largest weight will be used to classify our
        data point.
    """
    def getWeightedClassification(self, dist, sorted, dataset, k_value):
        classes = {}
        weights = {}

        kClosest = sorted[:k_value]

        """
            Loop through the to k closest values and store each classification in a new dict data structure for easy
            lookup. We will end up with a structure like
            
            {
                1.0: [1, 2, 3],
                0.0: [1, 2, 2, 4]
            }
        """
        for i in kClosest:
            key = dataset[i][5]

            if not key in classes:
                classes[key] = [dist[i]]
            else:
                classes[key].append(dist[i])

        """
            Loop through our dictionary and calculate the total weight for each classification. We will end up with
            a dict data structure like 
            
            {
                1.0: 1.83
                0.0: 2.08
            }
        """
        for classification in classes:
            total = 0
            for i in classes[classification]:
                total += (1 / i)

            weights[classification] = total

        largestWeight = 0
        selectedClassification = None

        """
            Loop through the weights data dictionary and find the key (classification) with the largest weight.
            We will then use the largest weight to classify the data point
        """
        for classification in weights:
            if weights[classification] > largestWeight:
                largestWeight = weights[classification]
                selectedClassification = classification

        return selectedClassification