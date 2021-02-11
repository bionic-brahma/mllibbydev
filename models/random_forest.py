###############################################################
# random forest algorithm developed in Collaboration
# for Risk Latte America's Inc
###############################################################


import numpy as np
from collections import Counter
from models.decision_tree import DecisionTree


def bootstrap_sample(X, y):
    """
    Performs bootstrapping the samples from the given data.
    It creates samples of the records randomly taken from the dataset
    :param X:  feature data matrix containing records in the format of list of lists
    :param y:  list of labels for that data feature matrix
    :return:  bootstrapped sample
    """
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)

    return X[idxs], y[idxs]


def most_common_label(y):
    """
    Finds the most common label from the list of labels
    :param y: list of labels
    :return: The number of occurrences of the most common label
    """
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]

    return most_common


class RandomForest:

    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        """
        Constructor for the random forest class
        :param n_trees: Number of decision trees to be used in the forest
        :param min_samples_split: minimum sample split for each each node to be considered
                at the time of spliting
        :param max_depth: Hyperparameter to be used in pruning the trees
        :param n_feats: Hyperparameter stating the number of features to loop over on.
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        """
        Fits the forest trees as per the training data.
        :param X: Feature matrix of the data. format--> [[1st rec], [2nd record],...]
        :param y: Data labels for the corresponding feature matrix. format [1st label, 2nd label,...]
        :return: None
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_feats=self.n_feats)
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        """
        Make the classification for the give input feature vector
        :param X: Feature vector to make classification on
        :return: Label for the feature vector
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]

        return np.array(y_pred)
