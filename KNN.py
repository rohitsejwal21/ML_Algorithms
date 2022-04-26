import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def euclidean_distance(x1, x2):
    # Euclidean Distance
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        
        #1. Euclidean distance between x and all training points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        #2. K nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #3. Majority Vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

if __name__ == "__main__":
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    K = [3, 5, 7, 9]
    for k in K:
        clf = KNN(k)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print("KNN classification accuracy with k = ", k, " is: ", accuracy_score(y_test, predictions))
