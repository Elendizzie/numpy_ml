import numpy as np
from tree_models import model

from sklearn.datasets import make_regression
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import matplotlib.pyplot as plt

def process_decision_tree():
    n_samples = 100
    n_feats = 7
    max_depth = 5

    # classifier = np.random.choice([True, False])
    # if classifier:
    # create classification problem

    n_classes = 3
    X, Y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_feats)

    # plt.scatter(X[:, 0], Y)
    # plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # initialize model
    def loss(yp, y):
        return 1 - accuracy_score(yp, y)

    criterion = np.random.choice(["entropy", "gini"])
    DT1 = model.DecisionTree(classifier=True,
                              max_depth=max_depth,
                              criterion=criterion)

    DT1.fit(X, Y)

    y_pred_train = DT1.predict(X_train)
    loss_train = loss(Y_train, y_pred_train)

    y_pred_test = DT1.predict(X_test)
    loss_test = loss(Y_test, y_pred_test)

    print("Classifier={}, criterion={}".format(True, criterion))
    print("max_depth={}, n_feats={}, n_samples={}".format(max_depth, n_feats, n_samples))
    print("loss_train={}, loss_test={}".format(loss_train, loss_test))

    X_ax = np.linspace(
        np.min(X_test.flatten()) - 1, np.max(X_test.flatten()) + 1, 100).reshape(-1, 1)

    y_pred_test_plot = DT1.predict(X_ax)

    plt.scatter(X_test.flatten(), Y_test.flatten(), c="b", alpha=0.5)
    plt.plot( X_ax.flatten(),
            y_pred_test_plot.flatten(),
            #  linewidth=0.5,
            label="DT".format(max_depth),
            color="yellowgreen",)
    plt.show()

def main():
    process_decision_tree()


if __name__ == '__main__':
    main()
