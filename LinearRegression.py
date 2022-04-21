import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class LinearRegression:
    
    #Linear regression class
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #1. init wwights and bias parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        #2. Gradient Descent

        for _ in range(self.n_iters):

            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate gradient
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # Change weights accordingly
            self.weights = self.weights - self.lr * dw
            self.bias = self.weights - self.lr * db

    def predict(self, X):

        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

if __name__ == "__main__":

    X, y = datasets.make_regression(n_samples = 1000, n_features = 1, noise = 20, random_state = 42)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

    LR = [0.001, 0.01, 0.1, 1]
    for lr in LR:
        clf = LinearRegression(lr, n_iters=1000)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print("Linear Regression MSE with learning rate = ", lr, " is: ", mean_squared_error(y_test, predictions))
