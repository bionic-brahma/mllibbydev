from models.utilities.split import k_fold_split, k_cross_validation_split
import numpy as np

a = np.array([[1, 2], [2, 3], [2, 5], [6, 3], [7, 2], [8, 9], [1, 7], [11, 5]])

b = np.array([1, 2, 3, 4, 5, 6, 7, 8])

res = k_cross_validation_split(a, b, k_value=2)
print(res[0][0][0],"\n\n",res[0][0][1])


