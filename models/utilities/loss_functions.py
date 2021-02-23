import numpy as np
from models.utilities.performance_matrices import accuracy


class Loss:
    def loss(self, y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
        """
        return NotImplementedError()

    def gradient(self, y, y_pred):
        """

        :param y:
        :param y_pred:
        :return:
        """
        raise NotImplementedError()

    def acc(self, y, y_pred):
        """

        :param y:
        :param y_pred:
        :return:
        """
        return 0


class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        """

        :param y:
        :param p:
        :return:
        """
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        """

        :param y:
        :param p:
        :return:
        """
        return accuracy(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        """

        :param y:
        :param p:
        :return:
        """
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)


class SquareLoss(Loss):
    def __init__(self):
        pass

    def loss(self, y, y_pred):
        """

        :param y:
        :param y_pred:
        :return:
        """
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        """

        :param y:
        :param y_pred:
        :return:
        """
        return -(y - y_pred)
