from models import  naivebayes
from models.utilities.split import k_cross_validation_split, train_test_split, dataset_by_single_label, subsets_by_label
from models.utilities.performance_matrices import accuracy, k_fold_validation_accuracy, confusion_matrix
from models.utilities.split import OVOdatamaker, subsets_by_label
from models.utilities.pre_processing import oversampling, datetime_formator, auto_oversample
import pandas as pd
import numpy as np

dataframe = pd.read_csv("Trained_model/test/Thyroid.csv")
labels = dataframe['label']
dataframe.drop('label', axis=1, inplace=True)
xfeat = dataframe
# print(np.unique(labels))
xfeat = [[2,2,2],[1,1,1],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9],[0,0,0],[10,10,10],[1,2,3],[2,3,4],[5,6,7],[5,6,8]]
labels = ["dev","dev","dev","dev","dev","dev","dev","dev","dev","dev","dev","arjun","arjun","arjun","arjun"]
# oversampled_dataX, oversampled_dataY = auto_oversample(xfeat, labels)
# print(len(xfeat), len(labels))
trx, testx, tr_y, testy = train_test_split(np.array(xfeat), np.array(labels), shuffle=True)
print("train: ",tr_y)
model = naivebayes.NaiveBayes()
model.fit(trx, tr_y)
predicted = model.predict(testx)
print(predicted)
input()
print("Accuracy: ", accuracy(testy, predicted))
confusion_matrix(testy, predicted)

print("+++++++++++++++smote++++++++++++++++++++")

# print("\n==============++==smote=========================")
X, y = auto_oversample(np.array(xfeat), np.array(labels), kpoint=2, method='smote')
# print("----- ----- -----  new record size ---- ---- -------", len(y))
trx, testx, tr_y, testy = train_test_split(X, y, shuffle=True)
print("train: ",tr_y)
model = naivebayes.NaiveBayes()
model.fit(trx, tr_y)
predicted = model.predict(testx)
print(predicted)
input()
print("Accuracy: ", accuracy(testy, predicted))
confusion_matrix(testy, predicted)

print("\n==============++==adasyn=========================")
X, y = auto_oversample(np.array(xfeat), np.array(labels), kpoint=2, method='adasyn')
# print("----- ----- -----  new record size ---- ---- -------", len(y))
trx, testx, tr_y, testy = train_test_split(X, y, shuffle=True)
print("train: ",tr_y)
model = naivebayes.NaiveBayes()
model.fit(trx, tr_y)
predicted = model.predict(testx)
print(predicted)
input()
print("Accuracy: ", accuracy(testy, predicted))
confusion_matrix(testy, predicted)
