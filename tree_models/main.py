import numpy as np
from tree_models import model

from sklearn.datasets import make_regression
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import matplotlib.pyplot as plt

def process_decision_tree():
    n_samples = np.random.randint(2, 100)
    n_feats = np.random.randint(2, 100)
    max_depth = np.random.randint(1, 5)

    # classifier = np.random.choice([True, False])
    # if classifier:
    # create classification problem

    n_classes = np.random.randint(2, 10)
    X, Y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_feats)

    plt.scatter(X[:, 0], Y)
    plt.show()

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3)

    # initialize model
    def loss(yp, y):
        return 1 - accuracy_score(yp, y)

    criterion = np.random.choice(["entropy", "gini"])
    DT1 = model.DecisionTree(classifier=True,
                              max_depth=max_depth,
                              criterion=criterion)

    DT1.fit(X, Y)

def main():
    process_decision_tree()


if __name__ == '__main__':
    main()
