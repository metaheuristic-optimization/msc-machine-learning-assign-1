from itertools import product
from src.knn_regression import KNNRegression

params = {
    'k_value': range(1, 10),
    'distanceAlg': ['euclidean', 'manhattan', 'minkowski'],
    'p': range(1, 5)
}

def main():

    best = 0
    bestParams = None

    for vals in product(*params.values()):
        knn = KNNRegression('datasets/regression/trainingData.csv', **dict(zip(params, vals)))
        accuracy = knn.predict()
        if accuracy > best:
            best = accuracy
            bestParams = vals

    print('Best parameters: ', bestParams)
    print('Best accuracy: ', best)

if __name__ == "__main__":
    main()
