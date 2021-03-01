######################################################################
# This is the decision tree algorithm. Branch division is based on the
# Entropy method.
# Created in collaboration for Risk Latte Americas Inc.
######################################################################

# Imports
import ast
import json
from collections import Counter
import numpy as np


# End of imports

# Function to calculate entropy
def entropy(y):
    """
    This method calculates the entropy of the given column
    :param y: Column to calculate the entropy of.
    :return: Entropy value
    """
    # Testing if the input is an np.array
    assert isinstance(y, np.ndarray), " The input is not an np.array"

    # Input attribute should have only positive integer as lable
    assert not ((y<0).any()), "The input array includes -ve values"

    hist = np.bincount(y)
    ps = hist / len(y)

    return -np.sum([p * np.log2(p) for p in ps if p > 0])


# Class representing the node of the tree.
class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        Constructor for node class.
        You can initialise the feature, threshold, left child, right child and value of the node.
        :param feature: Best feature on which the split has been taken
        :param threshold: Value of threshold for best feature
        :param left: This is link to the left node, if there is any.
        :param right: This is the link to the right node, if there is any.
        :param value: Class label, only in case of the node is leaf node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Checks if the node is leaf node.
        :return: Boolean value. True if the node is leaf node, else false.
        """
        return self.value is not None


class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        """
        Constructor for Decision tree class.
        Creates a new instance of the DecisionTree()
        :param min_samples_split: Hyperparameter stating the minimum sample require to perform further split.
        :param max_depth: Hyperparameter stating the depth too which the tree can grow.
        :param n_feats: Hyperparameter stating the number of features to loop over on.
        """
        self.min_samplessplit = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        self.parameters = None

    def fit(self, X, y):
        """
        Fits the tree as per the training data.
        :param X: Feature matrix of the data. format--> [[1st rec], [2nd record],...]
        :param y: Data labels for the corresponding feature matrix. format [1st label, 2nd label,...]
        :return: None
        """
        self.n_feats = np.array(X).shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self.span_tree(X, y)

    def predict(self, X):
        """
        Make the classification for the give input feature vector
        :param X: Feature vector to make classification on
        :return: Label for the feature vector
        """
        return np.array([self.travel_tree(x, self.root) for x in X])

    def span_tree(self, X, y, depth=0):
        """
        The method grows the tree and increases the depth.
        :param X: Feature matrix to grow the tree.
        :param y: Labels corresponding to feature matrix
        :param depth: The depth of the tree at beginning.
        :return: The root node
        """
        n_samples, n_features = np.array(X).shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samplessplit):
            leaf_value = self.likely_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # selecting  best split according to information gain
        best_feat, best_thresh = self._best_criteria(np.array(X), y, feat_idxs)

        # span the children that result from the split
        left_idxs, right_idxs = self.split(np.array(X)[:, best_feat], best_thresh)
        left = self.span_tree(np.array(X)[left_idxs, :], np.array(y)[left_idxs], depth + 1)
        right = self.span_tree(np.array(X)[right_idxs, :], np.array(y)[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        """
        Method to select the best splitting feature among the given feature list.
        :param X: Feature matrix.
        :param y: labels for the feature matrix.
        :param feat_idxs: list of indices giving the features to look best split feature among them.
        :return: best split feature and its threshold.
        """
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.info_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def info_gain(self, y, X_column, split_thresh):
        """
        Calculates the information gain for the given column
        :param y: parent (feature column) to the column whose information gain needs to be calculated.
        :param X_column: Column whose information gain is required to be calculated.
        :param split_thresh: threshold to split the child nodes of given column (feature)
        :return: Information gain of the given column
        """
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self.split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        #print(entropy(y[left_idxs]))
        e_l, e_r = entropy(np.array(y)[left_idxs]), entropy(np.array(y)[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def split(self, X_column, split_thresh):
        """
        Splits the data in two parts on the basis of threshold value of the given feature (column)
        :param X_column: Feature on which the threshold is to be applied to make the split.
        :param split_thresh: Threshold for the feature to split the data
        :return: The indices( left indices and right indices ) for the splitted data
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def travel_tree(self, x, node):
        """
        Traversed the tree and gives out the values in DFS manner.
        :param x: data feature matrix.
        :param node: node to begin with.
        :return: values in DFS manner
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.travel_tree(x, node.left)
        return self.travel_tree(x, node.right)

    def likely_label(self, y):
        """
        Methode scans the give vector for most common label.
        :param y: The vector or list on which we want to find most common label.
        :return: The number of occurrences of the most common label
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


'''
    def Save_Model(self, file_name):
        """
        Saves the model in JSON format
        :param file_name: address and filename (without extension) to save the file.
        :return: None
        """

        model_data = {"model_param": str(self.root)}

        model_file = file_name + str(".json")

        try:
            with open(model_file, 'x') as modelfile:
                json.dump(model_data, modelfile, indent=4)
                print("model_saved sucessfully in file named : ", model_file)
        except:
            with open(model_file, 'w') as modelfile:
                json.dump(model_data, modelfile, indent=4)
                print("model_saved sucessfully in file named : ", model_file)
        return

    def Load_Model(self, file_name):
        """
        Loads the earlier saved model.
        :param file_name: Address with file name with extension (JSON file is supported)
        :return: reflects the json contents while model parameters are loaded.
        """

        with open(file_name, "r") as model_file:
            data = json.load(model_file)
            self.parameters = ast.literal_eval(data["model_param"])
            return data '''

