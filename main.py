from src.knn import KNN

def main():
    k = 3
    knn = KNN('datasets/test.csv', 'datasets/testData2.csv', k)
    knn.run()


if __name__ == "__main__":
    main()
