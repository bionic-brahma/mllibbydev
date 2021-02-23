from models.utilities.split import k_cross_validation_split, train_test_split
from models.utilities.performance_matrices import accuracy, k_fold_validation_accuracy
from models.utilities.pre_processing import OVOdatamaker
import numpy as np
from models import svm

model = svm.SVM()
a = np.array([[1, 2], [2, 3], [2, 5], [6, 3], [7, 2], [8, 9], [1, 7], [11, 5], [1, 1], [5, 5], [90, 90]])
b = np.array([5, 5, 5, 10, 10, 5, 5, 10, 2, 2, 2])

#OVO testing
dataset = OVOdatamaker(a, b)
print(dataset[1][1])

tr_x, tx, tr_y, ty = train_test_split(a, b)
model.fit(tr_x, tr_y)
print("predicted value for [10,100]:", model.predict([[10, 100]]))
print("predicted value for [1000,100]:", model.predict([[1000, 100]]))
print("test accuracy: ", accuracy(ty, model.predict(tx)))
print("predicted value for [10,100]:", model.predict([10, 100]))
k = 3
res = k_cross_validation_split(a, b, k_value=k)

pred_list = []
actual_list = []

for i in range(k):
    model = svm.SVM()
    model.fit(res[0][i][0], res[0][i][1])
    pred_list.append(model.predict(res[1][i][0]))
    actual_list.append(res[1][i][1])

print("K-fold_validation_accuracy (accuracy, tolerance): ",
      k_fold_validation_accuracy(actual_list, pred_list, return_tolerance=True))
