import numpy as np
import pandas as pd
import operator

class KNN:

    columns = ['bi_rads', 'age', 'shape', 'margin', 'density', 'severity']

    def __init__(self, trainingFile, k):
        self.readData(trainingFile)
        self.k = k

    def readData(self, trainingFile):
        self.trainingSet = pd.read_csv(trainingFile, names = self.columns, header=None)

        self.trainingSet.drop(self.trainingSet.index[self.trainingSet['bi_rads'] > 5], inplace=True)

    def run(self):
        correct = 0
        incorrect = 0

        for index, row in self.trainingSet.iterrows():
            dist, sorted = self.calculateDistances(self.trainingSet.values, row.values)
            classification = self.getWeightedClassification(dist, sorted)

            if classification == row.values[5]:
                correct += 1
            else:
                incorrect += 1

        accuracy = (correct / (correct + incorrect)) * 100

        print('Accuracy is: ', accuracy, '%')

    def calculateDistances(self, A, B):
        dist = np.sqrt(((A - B)**2).sum(-1))
        sorted = np.argsort(dist)[np.in1d(np.argsort(dist),np.where(dist),1)]

        return dist, sorted

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
