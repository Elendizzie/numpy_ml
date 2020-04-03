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

        inverse = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        self.beta = np.dot(inverse, y)

    def fit_sgd(self, X, y, lr, iter):
        """
        Fit the regression with stochastic gradient descent
        :param X: same as above
        :param y: same as above
        :param lr: sgd learning rate
        :param iter: iteration before converge
        :return:
        """

        samples, features = X.shape  # []
        self.beta = np.zeros(shape=(features,))
        self.bias = 0
        costs = []

        for i in range(iter):
            # step 1: compute y_predict
            y_predict = np.dot(X, self.beta)

            # step 2:calculate cost function
            cost = (1.0 / samples) * np.sum((y_predict - y) ** 2)
            costs.append(cost)

            # if i % 100 == 0:
            print("Cost at iteration %d: %f" % (i, cost))

            # step 3: compute gradients
            dJ_dw = (2.0 / samples) * np.dot(X.T, (y_predict - y))
            # dJ_db = (2.0 / samples) * np.sum(y_predict - y)

            # step 4: update params
            self.beta = self.beta - lr * dJ_dw
            # self.bias = self.bias - lr * dJ_db

        return self.beta, costs

    def predict(self, X):
        return np.dot(X, self.beta)


class LogisticRegression:
    def __init__(self, reg='l2', gamma=0):
        """
        A simple logistic regression fit with sgd with regularization
        :param reg: l1 or l2 regularization
        :param gamma: float in [0, 1], larger values indicate larger penalties,
        and 0 indicates no penalty
        """

        error_msg = "Reg type must be 'l1' or 'l2', but instead got: {}".format(reg)
        assert reg in ['l1', 'l2'], error_msg

        self.reg = reg
        self.gamma = gamma
        self.beta = None

    def loss_penalty(self, X, y, y_pred):
        """
        Implement cross-entropy loss and L1/L2 regularization
        :param X:
        :param y:
        :param y_pred:
        :return:
        """
        N, M = X.shape
        order = 2 if self.reg == 'l2' else 1

        ce_loss = -np.log(y_pred[y == 1]).sum() - np.log(y_pred[y == 1]).sum()
        penalty = 0.5 * self.gamma * np.linalg.norm(self.beta, ord=order) ** 2

        return ce_loss + penalty / N

    def param_update(self, X, y, y_pred):

        """
        Calculate the gradient of loss_penalty function
        :param X:
        :param y:
        :param y_pred:
        :return:
        """
        N, M = X.shape

        return -np.dot(y - y_pred, X) / N

    def fit_sgd(self, X, y, thresh=1e-7, lr=1e-2, iters=100):
        """
        train with gradient descent
        :param X: ndarray of NXM, N samples, M dimenstions
        :param y: ndarray of Nx1, N labels
        :param lr: learning rate for sgd
        :param iters: maximum num of iteration before converge
        :return:
        """

        loss_prev = np.inf
        self.beta = np.random.rand(X.shape[1])

        for iter in range(iters):
            y_pred = self.sigmoid(np.dot(X, self.beta))
            loss = self.loss_penalty(X, y, y_pred)

            print("Cost at iteration %d: %f" % (iter, loss))

            if loss_prev - loss < thresh:
                return
            loss_prev = loss

            self.beta -= lr * self.param_update(X, y, y_pred)

    def predict(self, X):
        return self.sigmoid(np.dot(X, self.beta))

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))
