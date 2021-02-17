from models.utilities.split import k_cross_validation_split
import numpy as np

a = np.array([[1, 2], [2, 3], [2, 5], [6, 3], [7, 2], [8, 9], [1, 7], [11, 5]])

b = np.array([1, 2, 3, 4, 5, 6, 7, 8])
k=8
res = k_cross_validation_split(a, b, k_value=k)
for i in range(k):
    print("set number: ", i)
    print(res[0][i], "\n\n", res[1][i])


