from src.knn import KNN

def main():
    """
        Edit these values to use different combinations of hyperparameters.

        The following distance algorithms are supported
            - euclidean
            - manhattan
            - minkowski

        The following classification algorithms are supported
            - vote
            - weighted distance
    """
    dataset = 'datasets/classification/trainingData2.csv'
    k_value = 3
    distanceAlg = 'euclidean'
    classificationAlg = 'vote'
    p_value = 1

    """
        Create the KNN model and display the accuracy
    """
    knn = KNN(dataset, k_value, distanceAlg, classificationAlg, p_value)

    accuracy = knn.run()

    print('Accuracy is: ', accuracy, '%')

if __name__ == "__main__":
    main()
