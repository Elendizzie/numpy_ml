import numpy as np


class DecisionTree:
    def __init__(self,
                 classifier=True,
                 max_depth=None,
                 num_feats=None,
                 criterion='entropy'):
        """

        :param classifier: For classification or regression
        :param max_depth: A point where to stop growing tree. if None, build the tree until all leaves are pure
        :param num_feats: Specifies number of features to use each data split. If None, use all on each split.
        :param criterion: Use 'entropy' or 'gini' for classification, use 'mse' for regression.
        """

        self.root = None
        self.depth = 0
        self.num_feats = num_feats
        self.criterion = criterion
        self.classifier = classifier
        self.max_depth = max_depth if max_depth else np.inf

        if classifier and classifier == 'mse':
            raise ValueError("mse is only used for regression problems")

        if not classifier and criterion in ["entropy", "gini"]:
            raise ValueError("entrpy and gini can only be used for classification problems")

    def find_best_split(self, X, Y, feat_idx):

        """
        Find the optimal splits params, and then split the data
        :param X:
        :param Y:
        :param feat_idx:
        :return:
        """

        best_gain = - np.inf
        split_idx, split_thresh = None, None

        # go through each feature, find the best idx to maximize the gain
        for i in feat_idx:
            feat_vals = X[:, i]
            levels = np.unique(feat_vals)
            thresh = (levels[:-1] + levels[1:]) / 2 if len(levels) > 1 else levels

            gains = np.array([self.impurity_gain(Y, t, feat_vals) for t in thresh])

            if gains.max() > best_gain:
                split_idx = i
                best_gain = gains.max()
                split_thresh = thresh[gains.argmax()]

        return split_idx, split_thresh

    def impurity_gain(self, Y, split_thresh, feat_vals):
        """
        Compute the gain with given thresh and features

        IG(split) = loss(parent) - weighted_avg[loss(left_child), loss(right_child)]

        :param Y:
        :param split_thresh:
        :param feat_vals:
        :return:
        """

        if self.criterion == "entropy":
            loss = self.entropy
        elif self.criterion == "gini":
            loss = self.gini
        elif self.criterion == "mse":
            loss = self.mse

        # generate split

        left = np.argwhere(feat_vals <= split_thresh).flatten()
        right = np.argwhere(feat_vals > split_thresh).flatten()

        if len(left) == 0 or len(right) == 0:
            return 0

        # compute the cost for each step
        parent_loss = loss(Y)

        n = len(Y)
        n_left = len(left)
        n_right = len(right)

        child_loss = (n_left / n) * loss(Y[left]) + (n_right / n) * loss(Y[right])

        impurity_gain = parent_loss - child_loss

        return impurity_gain

    def fit(self, X, Y):
        """
        Fit the BST to a dataset
        :param X: ndarray of shape NxM, N samples with M feature dim
        :param Y: ndarray of shape Nx1, an array of class labels for classification, a set of target values for regressions.
        :return:
        """

        self.num_classes = max(Y) + 1 if self.classifier else None
        self.num_feats = X.shape[1] if not self.num_feats else min(X.shape[1], self.num_feats)

        self.root = self.grow_tree(X, Y)

    def predict(self, X):

        """
        Predict use trained bst
        :param X: ndarray of shape NxM,
        :return: a prediction array of shape (N,)
        """

        return np.array([self.tree_traverse(x, self.root) for x in X])

    def grow_tree(self, X, Y, cur_depth=0):

        # if all labels are the same, return a leaf
        if len(set(Y)) == 1:
            if self.classifier:
                prob = np.zeros(self.num_classes)
                prob[Y[0]] = 1.0
            return Leaf(prob) if self.classifier else Leaf(Y[0])

        if cur_depth >= self.max_depth:
            v = np.mean(Y, axis=0)
            if self.classifier:
                v = np.bincount(Y, minlength=self.num_classes) / len(Y)
            return Leaf(v)

        cur_depth += 1
        self.depth = max(self.depth, cur_depth)

        N, M = X.shape
        feat_idx = np.random.choice(M, self.num_feats, replace=False)

        # greedily select the best split
        feat, thresh = self.find_best_split(X, Y, feat_idx)

        l = np.argwhere(X[:, feat] <= thresh).flatten()
        r = np.argwhere(X[:, feat] > thresh).flatten()

        left = self.grow_tree(X[l, :], Y[l], cur_depth)
        right = self.grow_tree(X[r, :], Y[r], cur_depth)

        return Node(left, right, (feat, thresh))

    def tree_traverse(self, X, node, prob=False):
        if isinstance(node, Leaf):
            if self.classifier:
                return node.value if prob else node.value.argmax()
            return node.value
        if X[node.feature] <= node.threshold:
            return self.tree_traverse(X, node.left, prob)
        return self.tree_traverse(X, node.right, prob)

    def mse(self, y):

        return np.mean(y - np.mean(y) ** 2)

    def entropy(self, y):

        """
        Entropy of a label sequence
        """
        hist = np.bincount(y)
        ps = hist / np.sum(hist)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def gini(self, y):
        """
        gini impurity, for each class K, calculate the fraction of samples with class K
        :param y:
        :return:
        """

        hist = np.bincount(y)
        N = np.sum(hist)
        return 1 - sum([(i / N) ** 2 for i in hist])


class RandomForest:

    def __init__(self,
                 n_trees,
                 max_depth,
                 n_feats,
                 classifier=True,
                 criterion="entropy"):

        """
        An ensemble of decision trees where each split is calculated using a
        random subset of the features from the input

        :param n_trees: number of decision trees to use
        :param max_depth: the max depth where to stop growing the tree
        :param n_feats: the number of features to sample on each split
        :param classifier: true for classifier and false for regression
        :param criterion: 'entropy' and 'gini' for classifier, and 'mse' for regression
        """

        self.trees = []
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.criterion = criterion
        self.classifier = classifier

    def fit(self, X, Y):
        """
        Create n_trees of bootstrapped samples from the training data and use each to fit a
        separate decision tree

        :param X:
        :param Y:
        :return:
        """

        self.trees = []

        for i in range(self.n_trees):
            X_sample, Y_sample = bootstrap_sampling(X, Y)
            DT = DecisionTree(classifier=self.classifier,
                              max_depth=self.max_depth,
                              num_feats=self.n_feats,
                              criterion=self.criterion)

            DT.fit(X_sample, Y_sample)
            self.trees.append(DT)

    def vote(self, predictions):
        """
        return the bagging prediction across all trees
        :param predictions: ndarray of shape (n_trees, N)
        :return: For classifier, returns the predicted label by majority vote of all decision trees.
        For regression, returns the average weighted output across all decision trees.
        """

        if self.classifier:
            out = [np.bincount(pred).argmax() for pred in predictions.T]
        else:
            out = [np.mean(pred) for pred in predictions.T]

        return np.array(out)

    def predict(self, X):
        """

        :param X:
        :return:
        """
        tree_prediction = np.array([[t.tree_traverse(x,t.root)for x in X] for t in self.trees])

        return self.vote(tree_prediction)

def bootstrap_sampling(X, Y):
    N, M = X.shape
    idxs = np.random.choice(N, N, replace=True)
    return X[idxs], Y[idxs]


class Node:
    def __init__(self, left, right, params):
        self.left = left
        self.right = right
        self.feature = params[0]
        self.threshold = params[1]

class Leaf:
    def __init__(self, value):
        """
        `value` is an array of class probabilities if classifier is True, else
        the mean of the region
        """
        self.value = value
