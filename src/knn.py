import numpy as np
import pandas as pd
import operator

class KNN:

    columns = ['bi_rads', 'age', 'shape', 'margin', 'density', 'severity']

    def __init__(self, trainingFile, k, distanceAlg='euclidean', classificationAlg='vote', p=1):
        self.readData(trainingFile)
        self.k = k
        self.distanceAlg = distanceAlg
        self.classificationAlg = classificationAlg
        self.p = p

    def readData(self, trainingFile):
        self.trainingSet = pd.read_csv(trainingFile, names = self.columns, header=None)

        self.trainingSet.drop(self.trainingSet.index[self.trainingSet['bi_rads'] > 5], inplace=True)

    def run(self):
        correct = 0
        incorrect = 0

        for index, row in self.trainingSet.iterrows():
            dist, sorted = self.calculateDistances(self.trainingSet.values, row.values)

            if self.classificationAlg == 'vote':
                classification = self.getVoteClassification(sorted)
            elif self.classificationAlg == 'weighted':
                classification = self.getWeightedClassification(dist, sorted)

            if classification == row.values[5]:
                correct += 1
            else:
                incorrect += 1

        accuracy = (correct / (correct + incorrect)) * 100

        return accuracy

    def calculateDistances(self, a, b):
        if self.distanceAlg == 'euclidean':
            dist = self.euclideanDistance(a, b)
        elif self.distanceAlg == 'manhattan':
            dist = self.manhattanDistance(a, b)
        elif self.distanceAlg == 'minkowski':
            dist = self.minkowskiDistance(a, b, self.p)

        sorted = np.argsort(dist)[np.in1d(np.argsort(dist),np.where(dist),1)]

        return dist, sorted

    def euclideanDistance(self, a, b):
        return np.sqrt(((a - b)**2).sum(-1))

    def manhattanDistance(self, a, b):
        return np.abs(a - b).sum(-1)

    def minkowskiDistance(self, a, b, p_value = 1):
        return np.abs(((a - b) / p_value) / (1/ p_value)).sum(-1)

    def getVoteClassification(self, sorted):
        votes = {}
        kClosest = sorted[:self.k]

        for i in kClosest:
            key = self.trainingSet.values[i][5]
            if not key in votes:
                votes[key] = 1
            else:
                votes[key] += 1

        return max(votes.items(), key=operator.itemgetter(1))[0]

    def getWeightedClassification(self, dist, sorted):
        classes = {}
        weights = {}

        kClosest = sorted[:self.k]

        for i in kClosest:
            key = self.trainingSet.values[i][5]

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
