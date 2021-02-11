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


def confusion_matrix(actual, predicted):
    """
    This function computes the confusion matrix for the given predicted
    and actual lists of labels
    :param actual: The list of actual labels
    :param predicted: The list of predicted labels
    :return: list of unique labels and the confusion matrix
    """
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[y][x] += 1

    return unique, matrix
