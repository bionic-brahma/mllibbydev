######################################################################
# K - nearest neighbor algorithm
# Created in collaboration for Risk Latte Americas Inc.
######################################################################

# Imports
import numpy as np
from collections import Counter


# End of imports

def euclidean_distance(x1, x2):
    """
    Calculates the euclidean_distance between two given points
    :param x1: Point one in n-coordinates
    :param x2: Second point in n-coordinates
    :return: Distance between the points in euclidean reference
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:

    def __init__(self, k=3):
        """
        Constructor for the KNN class.
        Creates an empty instance of the KNN.
        :param k: Number of the nearest data points to be considered while classifying the new data point.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fits the data (training dataset) in K- Nearest Neighbour
        :param X: Feature data matrix
        :param y: List of labels for data matrix
        :return: None
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the class (cluster for given data)
        :param X: Data to classify in clusters
        :return: Clusters recognised by most common label
        """

        y_pred = [self.hypothisize(x) for x in X]

        return np.array(y_pred)

    def hypothisize(self, x):
        """
        Takes the given data point and assigns the most common label after classifying it to a cluster.
        :param x: Dataset.
        :return: Label for the cluster assigned to data point.
        """
        # Compute distances 
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices 
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return common label
        most_common = Counter(k_neighbor_labels).most_common(1)

        return most_common[0][0]
