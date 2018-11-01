import numpy as np
import pandas as pd

class KNN:

    def __init__(self, trainingFile, dataFile, k):
        self.readData(trainingFile, dataFile)
        self.k = k

    def readData(self, trainingFile, dataFile):
        self.trainingSet = pd.read_csv(trainingFile, header=None)
        self.dataSet = pd.read_csv(dataFile, header=None)

    def run(self):
        for index, row in self.trainingSet.iterrows():
            self.calculateDistances(self.trainingSet.values, row.values)

    def calculateDistances(self, A, B):
        dist = np.sqrt(((A - B)**2).sum(-1))
        sorted = np.argsort(dist)[np.in1d(np.argsort(dist),np.where(dist),1)];

        return dist, sorted

    def getNeighbours(self):
        pass

