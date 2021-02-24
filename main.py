from models.svm import SVM
from models.utilities.split import k_cross_validation_split, train_test_split, dataset_by_single_label
from models.utilities.performance_matrices import accuracy, k_fold_validation_accuracy
from models.utilities.split import OVOdatamaker, subsets_by_label
from models.utilities.pre_processing import smote
import numpy as np

transformer = smote()
data = transformer.generate_synthetic_points([[1, 2], [2, 3], [2, 5], [6, 3], [7, 2], [8, 9], [1, 7]],500,3)
print(data)
a = np.array([[1, 2], [2, 3], [2, 5], [6, 3], [7, 2], [8, 9], [1, 7], [11, 5], [1, 1], [5, 5], [90, 90]])
b = np.array([2, 2, 2, 10, 10, 2, 2, 10, 2, 2, 2])
datax, datay = dataset_by_single_label(a,b,10)
print(datax,datay)

#print("predicted value for [10,100]:", model.predict([10, 100]))

