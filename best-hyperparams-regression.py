from itertools import product
from src.knn_regression import KNNRegression

params = {
    'k_value': range(1, 10),
    'distanceAlg': ['euclidean', 'manhattan', 'minkowski'],
    'p': range(1, 5)
}

"""
    Attempt to find the best hyper parameters for our regression problem. This uses a brute force method
    which will run through every combination of the params above. The accuracy of the model is used to find the best
    parameters. When the program completes it will display th combination of best parameters. NOTE as this will run
    through every possible combination, it can take a while to run depending on your hardware.
"""
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
