import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=10000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # compute gradient decent
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        self.y_pred = y_pred
        

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
    def mae(self, y, y_pred):
        total = 0
        n = len(y)
        for i in range(0,n):
            #finding the absolute difference
            diff = abs(y[i] - y_pred[i])
            
            #obtain average
            total += diff
        MAE = total/n
        return MAE
    
    def mse(self, y, y_pred):
        total = 0
        n = len(y)
        for i in range(0,n):
            #finding the difference
            diff = y[i] - y_pred[i]
            
            #taking square of the difference
            squared_diff = diff**2 
            
            #obtain average
            total += squared_diff
        MSE = total/n
        return MSE