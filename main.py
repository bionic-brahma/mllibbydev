import numpy as np
from models.utilities.split import train_test_split
from models.knn import KNN
from models.utilities.performance_matrices import accuracy

X = np.array([[2,3], [-1,-8], [9,7], [8,4],[-9,-6], [-6,-11]])
y = np.array(['Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Negative'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X_train.shape)
print(X_train[0])

print(y_train.shape)
print(y_train)

k = 5
clf = KNN(k=k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(X_test)
print(predictions)
print("custom KNN classification accuracy", accuracy(y_test, predictions))
acc = accuracy(y_test, predictions)
print("Accuracy:", acc)
