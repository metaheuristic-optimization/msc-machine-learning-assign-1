from src.knn_regression import KNNRegression

def main():
    """
        Edit these values to use different combinations of hyperparameters.

        The following distance algorithms are supported
            - euclidean
            - manhattan
            - minkowski
    """
    dataset = 'datasets/regression/trainingData.csv'
    k_value = 3
    distanceAlg = 'euclidean'
    p_value = 1

    """
        Create the KNN model and display the accuracy
    """
    knn = KNNRegression(dataset, k_value, distanceAlg, p_value)

    accuracy = knn.predict()

    print('Accuracy is: {0:.4f}%'.format(accuracy))

if __name__ == "__main__":
    main()
