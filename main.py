from src.knn import KNN

def main():
    k = 6
    knn = KNN('datasets/trainingData2.csv', k)
    knn.run()

if __name__ == "__main__":
    main()
