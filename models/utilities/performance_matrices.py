import numpy as np


def accuracy(y_true, y_pred):
    """
    Calculates the accuracy
    :param y_true: The list of actual labels
    :param y_pred: The list of predicted labels
    :return: accuracy
    """
    if len(y_true) != 0:
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    else:
        return 0.0