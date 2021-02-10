import numpy as np


def linear_kernel(x1, x2):
    """ linear kernel to be used in the algorithm

    Parameters:
    (Type: numpy array) x1 : ----> vector of x1 data
    (Type: numpy array) x2 : ----> vector of x2 data

    Return:

    (Type: numpy array)

    """
    return np.dot(x1, x2)


def polynomial_kernel(x, y, C=1, d=3):
    """ polynomial kernel to be used in the algorithm

    Parameters:
    (Type: numpy array) x1 : ----> vector of x data
    (Type: numpy array) y : ----> vector of y data
    (Type: integer/float) c : ----> constant
    (Type: integer) d -----> degree of polynomial

    Return:

    (Type: numpy array)

    """

    return (np.dot(x, y) + C) ** d


def gaussian_kernel(x, y, sigma=5.0):
    """ gaussian kernel to be used in the algorithm

   Parameters:
   (Type: numpy array) x1 : ----> vector of x data
   (Type: numpy array) x2 : ----> vector of y data
   (Type: integer) sigma -----> gaussian sigma

   Return:

   (Type: numpy array)

   """
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


def kernel_fx(x1, x2, kernel="linear", p=3, sigma=5.0):
    """ kernels to be used in the algorithm

   Parameters:
   (Type: numpy array) x1 : ----> vector of x1 data
   (Type: numpy array) x2 : ----> vector of x2 data
   (Type: string) kernel : ----> type of kernel used  linear/polynomial/gaussian
   (Type: integer) d -----> degree of polynomial
   (Type: integer) sigma -----> gaussian sigma

   Return:

   (Type: numpy array)

   """

    if kernel == "linear":
        return np.dot(x1, x2)

    elif kernel == "polynomial":
        return (1 + np.dot(x1, x2)) ** p

    elif kernel == "gaussian":
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

    else:
        raise Exception("invalid kernel selected, available kernels : linear, polynomial, gaussian")
