from models.utilities.split import cross_validation_split
import numpy as np

a = np.array([[1, 2], [2, 3], [2, 5], [6, 3], [7, 2], [8, 9], [1, 7], [11, 5]])

b = np.array([1, 2, 3, 4, 5, 6, 7, 8])

xf, yf = cross_validation_split(a, b, 4)

print(xf[0], yf[0])
