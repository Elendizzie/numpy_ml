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
        split_idx, thresh_idx = None, None

        for i in feat_idx:
            feat_vals = X[:, i]
            levels = np.unique(feat_vals)
            thresh = (levels[:-1] + levels[1:]) / 2 if len(levels) > 1 else levels

            gains = np.array([self.impurity_gain(Y, t, feat_vals) for t in thresh])

            if gains.max() > best_gain:
                split_idx = i
                best_gain = gains.max()
                split_thresh = thresh[gains.argmax()]

        return split_idx, thresh_idx

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

    def grow_tree(self, X, Y, cur_depth=0):

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

        return Node(left, right, feat, thresh)

    def tree_traverse(self, X, node, prob=False):
        if isinstance(node, Leaf):
            if self.classifier:
                return node.value if prob else node.value.argmax()
            return node.value
        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left, prob)
        return self._traverse(X, node.right, prob)

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


class Node:
    def __init__(self, left, right, feat, thresh):
        self.left = left
        self.right = right
        self.feature = feat
        self.threshold = thresh


class Leaf:
    def __init__(self, value):
        """
        `value` is an array of class probabilities if classifier is True, else
        the mean of the region
        """
        self.value = value
