import numpy as np
from linear_models import model
from sklearn.datasets import make_regression
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss

import matplotlib.pyplot as plt


def generate_regression_data(n_samples,
                             n_features,
                             n_targets,
                             bias,
                             noise_std=1,
                             seed=0):
    X, y, coef = make_regression(n_samples=n_samples,
                                 n_features=n_features,
                                 n_targets=n_targets,
                                 bias=bias,
                                 noise=noise_std,
                                 coef=True,
                                 random_state=seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    return X_train, X_test, y_train, y_test, coef


def generate_classification_data(n_samples,
                                 n_classes,
                                 n_features,
                                 seed=0):
    X, y = make_blobs(n_samples, n_features, n_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    return X_train, X_test, y_train, y_test


def process_LinearRegression():
    np.random.seed(12345)

    std = np.random.randint(0, 1000)
    intercept = np.random.rand() * np.random.randint(-300, 300)
    X_train, X_test, y_train, y_test, coef = generate_regression_data(50, 1, 1, intercept, std, 0)

    LR1 = model.LinearRegression(fit_bias=True)
    LR1.fit_affine_proj(X_train, y_train)
    y_pred1 = LR1.predict(X_test)
    loss = np.mean((y_test - y_pred1) ** 2)
    print(loss)

    LR2 = model.LinearRegression(fit_bias=True)
    W, costs = LR2.fit_sgd(X_train, y_train, 5e-2, 50)
    y_pred2 = LR2.predict(X_test)

    xmin = min(X_test) - 0.1 * (max(X_test) - min(X_test))
    xmax = max(X_test) + 0.1 * (max(X_test) - min(X_test))
    X_plot = np.linspace(xmin, xmax, 20)
    y1_plot = LR1.predict(X_plot)
    y2_plot = LR2.predict(X_plot)

    plt.figure(1)
    plt.scatter(X_test, y_test, alpha=0.5)
    plt.plot(X_plot, y1_plot, 'r')
    plt.show()

    plt.figure(2)
    plt.scatter(X_test, y_test, alpha=0.5)
    plt.plot(X_plot, y2_plot, 'g')
    plt.show()


def process_LogisticRegression():
    np.random.seed(12345)

    X_train, X_test, y_train, y_test = generate_classification_data(1500, 2, 1, 0)

    # plt.scatter(X_train, y_train)
    # plt.show()

    LR = model.LogisticRegression(reg='l2', gamma=0.2)
    LR.fit_sgd(X_train, y_train, thresh=1e-4, lr=0.1, iters=100000000)
    y_pred = (LR.predict(X_test) >= 0.5) * 1.0

    print(f"test accuracy: {100 - np.mean(np.abs(y_pred - y_test))}%")

    xmin = min(X_test) - 0.1 * (max(X_test) - min(X_test))
    xmax = max(X_test) + 0.1 * (max(X_test) - min(X_test))
    X_plot = np.linspace(xmin, xmax, 100)
    y_plot = LR.predict(X_plot)

    plt.figure(1)
    plt.scatter(X_test[y_pred == 0], y_test[y_pred == 0], alpha=0.5)
    plt.scatter(X_test[y_pred == 1], y_test[y_pred == 1], alpha=0.5)
    plt.plot(X_plot, y_plot, label="mine", alpha=0.75)
    plt.show()


def main():
    # process_LinearRegression()
    process_LogisticRegression()


if __name__ == '__main__':
    main()
