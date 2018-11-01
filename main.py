from src.knn import KNN

def main():
    k = 20
    knn = KNN('datasets/trainingData2.csv', 'datasets/testData2.csv', k)
    knn.run()

if __name__ == "__main__":
    main()
