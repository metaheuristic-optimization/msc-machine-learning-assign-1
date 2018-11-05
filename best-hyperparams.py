from itertools import product
from src.knn import KNN

params = {
    'k': range(1, 20),
    'distanceAlg': ['euclidean', 'manhattan', 'minkowski'],
    'classificationAlg': ['vote', 'weighted'],
    'p': range(1, 4)
}

def main():

    best = 0
    bestParams = None

    for vals in product(*params.values()):
        knn = KNN('datasets/trainingData2.csv', **dict(zip(params, vals)))
        accuracy = knn.run()
        if accuracy > best:
            best = accuracy
            bestParams = vals

    print('Best parameters: ', bestParams)
    print('Best accuracy: ', best)

if __name__ == "__main__":
    main()
