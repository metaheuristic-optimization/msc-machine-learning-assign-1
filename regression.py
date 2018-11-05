from src.knn_regression import KNNRegression

def main():
    k = 3
    knn = KNNRegression('datasets/regression/testData.csv', k, 'minkowski', 10)
    accuracy = knn.predict()

    print('Accuracy is: ', accuracy, '%')

if __name__ == "__main__":
    main()
