import numpy as np
from pandas import read_csv

class KNN:

    def __init__(self, trainingFile, dataFile, k):
        self.readData(trainingFile, dataFile)
        self.k = k

    def readData(self, trainingFile, dataFile):
        self.trainingSet = read_csv(trainingFile)
        self.dataSet = read_csv(dataFile)

    def run(self):
        for index, row in self.trainingSet.iterrows():
            self.calculateDistances(self.trainingSet.values, row.values)

    def calculateDistances(self, A, B):
        #print(A)
        #print(B)
        dist = np.sqrt(((A - B)**2).sum(-1))

        print(dist)
        # dists = np.hypot(A[:, 0, np.newaxis]-B[:, 0], A[:, 1, np.newaxis]-B[:, 1])

        #print(dists)
        #dist = (A - B)**2
        #dist = np.sum(dist, axis=1)
        #dist = np.sqrt(dist)
        #print(np.sqrt(np.sum((A - B) ** 2)))

    def getNeighbours(self):
        pass

