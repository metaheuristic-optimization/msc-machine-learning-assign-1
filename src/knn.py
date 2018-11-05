import numpy as np
import pandas as pd
import operator
from src.utils import Utils

class KNN:

    columns = ['bi_rads', 'age', 'shape', 'margin', 'density', 'severity']

    def __init__(self, trainingFile, k_value, distanceAlg='euclidean', classificationAlg='vote', p=1):
        self.readData(trainingFile)
        self.k_value = k_value
        self.distanceAlg = distanceAlg
        self.classificationAlg = classificationAlg
        self.p = p
        self.utils = Utils()

    def readData(self, trainingFile):
        self.trainingSet = pd.read_csv(trainingFile, names = self.columns, header=None)

        self.trainingSet.drop(self.trainingSet.index[self.trainingSet['bi_rads'] > 5], inplace=True)

    def run(self):
        correct = 0
        incorrect = 0

        for index, row in self.trainingSet.iterrows():
            dist, sorted = self.calculateDistances(self.trainingSet.values, row.values)

            if self.classificationAlg == 'vote':
                classification = self.getVoteClassification(sorted, self.trainingSet.values, self.k_value)
            elif self.classificationAlg == 'weighted':
                classification = self.getWeightedClassification(dist, sorted, self.trainingSet.values, self.k_value)

            if classification == row.values[5]:
                correct += 1
            else:
                incorrect += 1

        accuracy = (correct / (correct + incorrect)) * 100

        return accuracy

    def calculateDistances(self, a, b):
        if self.distanceAlg == 'euclidean':
            dist = self.utils.euclideanDistance(a, b)
        elif self.distanceAlg == 'manhattan':
            dist = self.utils.manhattanDistance(a, b)
        elif self.distanceAlg == 'minkowski':
            dist = self.utils.minkowskiDistance(a, b, self.p)

        sorted = np.argsort(dist)[np.in1d(np.argsort(dist),np.where(dist),1)]

        return dist, sorted

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

    def getWeightedClassification(self, dist, sorted, dataset, k_value):
        classes = {}
        weights = {}

        kClosest = sorted[:k_value]

        for i in kClosest:
            key = dataset[i][5]

            if not key in classes:
                classes[key] = [dist[i]]
            else:
                classes[key].append(dist[i])

        for classification in classes:
            total = 0
            for i in classes[classification]:
                total += (1 / i)

            weights[classification] = total

        largestWeight = 0
        selectedClassification = None

        for classification in weights:
            if weights[classification] > largestWeight:
                largestWeight = weights[classification]
                selectedClassification = classification

        return selectedClassification