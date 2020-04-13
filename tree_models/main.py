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

    classifier = False

    if classifier:

        n_classes = 5
        X, Y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_feats)

        plt.scatter(X[:, 0], Y)
        plt.show()

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        # initialize model
        def loss(yp, y):
            return 1 - accuracy_score(yp, y)

        criterion = np.random.choice(["entropy", "gini"])

    else:

        X, Y = make_regression(n_samples=n_samples, n_features=1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        criterion = "mse"
        loss = mean_squared_error

    DT1 = model.DecisionTree(classifier=classifier,
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

    if classifier:
        for i in np.unique(Y_test):
            plt.scatter(
                X_test[y_pred_test == i, 1].flatten(),
                X_test[y_pred_test == i, 2].flatten(),
            )
        plt.show()
    else:
        X_ax = np.linspace(
            np.min(X_test.flatten()) - 1, np.max(X_test.flatten()) + 1, 100).reshape(-1, 1)
        y_pred_test_plot = DT1.predict(X_ax)

        plt.scatter(X_test.flatten(), Y_test.flatten(), c="b", alpha=0.5)
        plt.plot(X_ax.flatten(),
                 y_pred_test_plot.flatten(),
                 label="DT".format(max_depth),
                 color="yellowgreen", )
        plt.show()


def process_random_forest():
    n_samples = 100
    n_trees = 5
    n_feats = 7
    max_depth = 5

    classifier = False

    if classifier:
        n_classes = 5
        X, Y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_feats)

        plt.scatter(X[:, 0], Y)
        plt.show()

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        # initialize model
        def loss(yp, y):
            return 1 - accuracy_score(yp, y)

        criterion = np.random.choice(["entropy", "gini"])

    else:

        X, Y = make_regression(n_samples=n_samples, n_features=1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        criterion = "mse"
        loss = mean_squared_error

    RF = model.RandomForest(n_trees=n_trees,
                            n_feats=n_feats,
                            max_depth=max_depth,
                            classifier=classifier,
                            criterion=criterion)

    RF.fit(X, Y)

    y_pred_train = RF.predict(X)
    loss_train = loss(Y, y_pred_train)

    y_pred_test = RF.predict(X_test)
    loss_test = loss(Y_test, y_pred_test)

    print("Classifier={}, criterion={}".format(True, criterion))
    print("n_trees={}, max_depth={}, n_feats={}, n_samples={}".format(n_trees, max_depth, n_feats, n_samples))

    if classifier:
        print("loss_train={}%, loss_test={}%".format(loss_train * 100, loss_test * 100))
    else:
        print("loss_train={}%, loss_test={}%".format(loss_train / X.shape[0], loss_test / X_test.shape[0]))

    if classifier:
        for i in np.unique(Y_test):
            plt.scatter(
                X_test[y_pred_test == i, 1].flatten(),
                X_test[y_pred_test == i, 2].flatten(),
            )
        plt.show()
    else:
        X_ax = np.linspace(
            np.min(X_test.flatten()) - 1, np.max(X_test.flatten()) + 1, 100).reshape(-1, 1)
        y_pred_test_plot = RF.predict(X_ax)

        plt.scatter(X_test.flatten(), Y_test.flatten(), c="b", alpha=0.5)
        plt.plot(X_ax.flatten(),
                 y_pred_test_plot.flatten(),
                 label="DT".format(max_depth),
                 color="yellowgreen", )
        plt.show()


def process_gb_decision_tree():
    n_ex = np.random.randint(2, 100)
    n_feats = np.random.randint(2, 100)
    n_trees = np.random.randint(2, 100)
    max_depth = np.random.randint(1, 5)

    classifier = np.random.choice([True, False])

    if classifier:
        # create classification problem
        n_classes = np.random.randint(2, 10)
        X, Y = make_blobs(
            n_samples=n_ex,
            centers=n_classes,
            n_features=n_feats
        )
        X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3)

        # initialize model
        def loss(yp, y):
            return 1 - accuracy_score(yp, y)

        # initialize model
        criterion = np.random.choice(["entropy", "gini"])
        GB = model.GradientBoostedDecisionTree(
            classifier=classifier,
            max_depth=max_depth,
            lr=0.1,
            loss="crossentropy",
            step_size="constant",
        )

    else:
        # create regeression problem
        X, Y = make_regression(n_samples=n_ex, n_features=n_feats)
        X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3)

        # initialize model
        criterion = "mse"
        loss = mean_squared_error
        GB = model.GradientBoostedDecisionTree(
            max_depth=max_depth,
            classifier=classifier,
            lr=0.1,
            loss="mse",
            step_size="constant"
        )

    print("Trial 1 ")
    print("\tClassifier={}, criterion={}".format(classifier, criterion))
    print("\tmax_depth={}, n_feats={}, n_ex={}".format(max_depth, n_feats, n_ex))
    if classifier:
        print("\tn_classes: {}".format(n_classes))

    GB.train_sgd(X, Y)

    y_predict_train = GB.predict(X)
    y_predict_test = GB.predict(X_test)


def main():
    # process_decision_tree()
    # process_random_forest()
    process_gb_decision_tree()


if __name__ == '__main__':
    main()
