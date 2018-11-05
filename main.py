from src.knn import KNN

def main():
    k = 3
    knn = KNN('datasets/trainingData2.csv', k, 'euclidean', 'vote')
    accuracy = knn.run()

    print('Accuracy is: ', accuracy, '%')

if __name__ == "__main__":
    main()
