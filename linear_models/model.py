import numpy as np


class LinearRegression:

    def __init__(self, fit_bias=True):
        self.beta = None
        self.fit_bias = fit_bias

    def fit_affine_proj(self, X, y):

        """
        fit the regression with MLE
        :param X:  nd array with shape NxM, N samples of M dim features
        :param y: nd array with shape NxK, N samples of K dim labels
        :return:
        """
        #
        # if self.fit_bias:
        #     X = np.c_[np.ones(X.shape[0]), X]

        inverse = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        self.beta = np.dot(inverse, y)

    def fit_sgd(self, X, y, lr, iter):

        samples, features = X.shape  # []
        self.beta = np.zeros(shape=(features,))
        self.bias = 0
        costs = []
        # print self.weights

        for i in range(iter):
            # step 1: compute y_predict

            y_predict = np.dot(X, self.beta)

            # step 2:calculate cost function
            cost = (1.0 / samples) * np.sum((y_predict - y) ** 2)
            costs.append(cost)

            # if i % 100 == 0:
            print ("Cost at iteration %d: %f" % (i, cost))

            # step 3: compute gradients
            dJ_dw = (2.0 / samples) * np.dot(X.T, (y_predict - y))
            # dJ_db = (2.0 / samples) * np.sum(y_predict - y)

            # step 4: update params
            self.beta = self.beta - lr * dJ_dw
            # self.bias = self.bias - lr * dJ_db

        return self.beta, costs


    def predict(self, X):

        # if self.fit_bias:
        #     X = np.c_[np.ones(X.shape[0]), X]

        return np.dot(X, self.beta)

