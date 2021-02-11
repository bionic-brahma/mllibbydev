import numpy as np
from models.svm import SVM
from models.utilities.split import train_test_split
from models.utilities.performance_matrices import accuracy


X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [6, 8], [2, 1], [5, 1], [8, 4], [9, 6]])
y = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
y = np.where(y == 0, -1, 1)

trainx, testx, trainy, testy = train_test_split(X, y)

clf = SVM()
print(trainy)
clf.fit(trainx, trainy)
predictions = clf.predict(testx)
print(predictions)
print(testy)

print(accuracy(testy, predictions))



