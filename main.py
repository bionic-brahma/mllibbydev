from models.utilities.split import k_cross_validation_split, train_test_split
from models.utilities.performance_matrices import accuracy, k_fold_validation_accuracy
from models.utilities.split import OVOdatamaker, subsets_by_label
import numpy as np
from models import MultiClassSVM

model = MultiClassSVM.MultiClassSVM()
a = np.array([[1, 2], [2, 3], [2, 5], [6, 3], [7, 2], [8, 9], [1, 7], [11, 5], [1, 1], [5, 5], [90, 90]])
b = np.array([5, 5, 5, 10, 10, 5, 5, 10, 2, 2, 2])

#OVO testing
databylabel = subsets_by_label(a, b)

dataset = OVOdatamaker(a, b)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
tr_x, tx, tr_y, ty = train_test_split(a, b)
model.fit(a,b)
print("predicted value for [10, 1],[1,10]:", model.predict([[10, 1],[1,10]]))
print("test accuracy: ", accuracy(ty, model.predict(tx)))
#print("predicted value for [10,100]:", model.predict([10, 100]))

