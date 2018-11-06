from itertools import product
from src.knn import KNN

params = {
    'k_value': range(1, 10),
    'distanceAlg': ['euclidean', 'manhattan', 'minkowski'],
    'classificationAlg': ['vote', 'weighted'],
    'p': range(1, 5)
}

"""
    Attempt to find the best hyper parameters for our classification problem. This uses a brute force method
    which will run through every combination of the params above. The accuracy of the model is used to find the best
    parameters. When the program completes it will display th combination of best parameters. NOTE as this will run
    through every possible combination, it can take a while to run depending on your hardware.
"""
def main():

    best = 0
    bestParams = None

    for vals in product(*params.values()):
        knn = KNN('datasets/classification/testData.csv', **dict(zip(params, vals)))
        accuracy = knn.run()
        if accuracy > best:
            best = accuracy
            bestParams = vals

    print('Best parameters: ', bestParams)
    print('Best accuracy: ', best)

if __name__ == "__main__":
    main()
