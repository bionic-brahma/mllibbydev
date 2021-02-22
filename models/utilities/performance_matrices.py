#######################################################################################
## Work by Devendra Kumar for Risk Latte AI Inc.                                      #
#######################################################################################

import numpy as np


def accuracy(y_true, y_pred, matric=None, independent_regressors=None):
    """
    Calculates the accuracy
    :param y_true: The list of actual labels
    :param y_pred: The list of predicted labels
    :param matric: accuracy calculation matric
    :param independent_regressors: Used in case of the adjusted R squared matric
    :return: accuracy
    """
    accuracy = 0.0
    pseudo_bias = 0.00001

    if len(y_true) != 0:

        if matric == "MAE":

            accuracy = 1 - (1 / len(y_true)) * np.sum(np.abs(y_true - y_pred))

        elif matric == "R-squared":

            accuracy = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        elif matric == "R-squared_adjusted":

            accuracy = 1 - (1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)) * (
                        len(y_true) - 1) / (len(y_true) - 1 - independent_regressors)

        elif matric == "MSE":

            accuracy = 1 - (1 / len(y_true)) * np.sum((y_true - y_pred) ** 2)

        else:
            sum_correct = 0
            for i in range(len(y_true)):
                #print(y_true)
                #print(y_pred)
                if y_true[i] == y_pred[i]:
                    sum_correct +=1
            accuracy = sum_correct / len(y_true)

        return accuracy*100

    else:
        return accuracy


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


def k_fold_validation_accuracy(list_actual_y, list_predicted, matric=None, independent_regressors=None,
                               return_tolerance=False):
    """
    The method to return the k fold validation accuracy with tolerance if return_tolerance is
    kept true.
    :param list_actual_y: list containing the lists of the actual labels from the dataset
    :param list_predicted: list containing the lists of the predicted labels
    :param matric: accuracy calculation matric
    :param independent_regressors: Used in case of the adjusted R squared matric
    :param return_tolerance: if kept true, returns the tolerance level in accuracy
    :return: returns the K- fold validation accuracy
    """

    k = len(list_actual_y)
    accuracies = []
    for i in range(k):
        accuracies.append(accuracy(list_actual_y[i], list_predicted[i], matric, independent_regressors))
    k_fold_validation_acc = np.mean(accuracies)
    tolerance = max(np.max(accuracies), np.min(accuracies)) - k_fold_validation_acc
    if return_tolerance:
        return k_fold_validation_acc, tolerance
    else:
        return k_fold_validation_acc
