from models.utilities.split import k_cross_validation_split, train_test_split
from models.utilities.performance_matrices import accuracy, k_fold_validation_accuracy
import numpy as np
from models import LinearRegression

model = LinearRegression.regression(iteration=100, learning_rate=0.001, show_steps=False)
a = np.array([[1, 2], [2, 3], [2, 5], [6, 3], [7, 2], [8, 9], [1, 7], [11, 5]])
b = np.array([3, 5, 7, 9, 9, 17, 8, 16])
tr_x, tx, tr_y, ty = train_test_split(a, b)
model.fit(tr_x, tr_y)
print("test accuracy: ", accuracy(ty, model.predict(tx), matric="MAE"))
print("predicted value for [10,1]:", model.predict([10,1]))
k = 3
res = k_cross_validation_split(a, b, k_value=k)

pred_list = []
actual_list = []

for i in range(k):
    model = LinearRegression.regression(iteration=100, learning_rate=0.001, show_steps=False)
    model.fit(res[0][i][0], res[0][i][1])
    pred_list.append(model.predict(res[1][i][0]))
    actual_list.append(res[1][i][1])

print("K-fold_validation_accuracy (accuracy, tolerance): ", k_fold_validation_accuracy(actual_list, pred_list, matric="MAE", return_tolerance=True))
