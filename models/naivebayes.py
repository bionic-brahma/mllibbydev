######################################################################
# This is the decision tree algorithm. Branch division is based on the
# Entropy method.
# Created in collaboration for Risk Latte Americas Inc.
######################################################################

# Imports
import numpy as np


# End of Imports

class NaiveBayes:
    def __init__(self):
        """
        Constructor for the NaiveBayes classifier model
        Creates an empty instance of the NaiveBayes
        """
        self._classes = None

    def fit(self, X, y):
        """
        Fits the given  data on the model.
        :param X: Data feature matrix
        :param y: List of labels
        :return: None
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Predicts the labels for the given record.
        :param X: record to predict the label for it.
        :return: Predicted label for given record of data.
        """
        y_pred = [self.hypothisize(x) for x in X]
        return np.array(y_pred)

    def hypothisize(self, x):
        """
        Test the given record with the contengency matrix to look for the
        classification label
        :param x: data record
        :return: The label for the given data record
        """
        posteriors = []

        # calculate  probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # return class with highest  probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        probability density function for conditional probability P(x|class_idx)
        :param class_idx: class indices
        :param x: given index
        :return: Probability density function
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]

        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
