import numpy as np
import pandas as pd
import operator

class KNN:

    def __init__(self, trainingFile, dataFile, k):
        self.readData(trainingFile, dataFile)
        self.k = k

    def readData(self, trainingFile, dataFile):
        self.trainingSet = pd.read_csv(trainingFile, header=None)
        self.dataSet = pd.read_csv(dataFile, header=None)

    def run(self):
        correct = 0
        incorrect = 0

        for index, row in self.trainingSet.iterrows():
            dist, sorted = self.calculateDistances(self.trainingSet.values, row.values)
            classification = self.getClassification(sorted)

            # print('Comparing', classification, ' to ', row.values[5])
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

    def getClassification(self, sorted):
        votes = {}
        closest = sorted[:self.k]

        for i in closest:
            key = self.trainingSet.values[i][5]
            if not key in votes:
                votes[key] = 1
            else:
                votes[key] += 1

        return max(votes.items(), key=operator.itemgetter(1))[0]

