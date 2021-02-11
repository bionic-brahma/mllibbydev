import numpy as np
from models.utilities import split, performance_matrices
from models.decision_tree import DecisionTree


X = np.array([[1,2,3], [5,2,3], [1,2,5], [1,6,3], [1,5,3], [5,2,6], [5,6,3]])
y = np.array([1,1,1,0,0,0,1])

X_train, X_test, y_train, y_test = split.train_test_split(X, y, test_size=0.2)
print(X_train)
print("*******************\n",y_train)

clf = DecisionTree(max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = performance_matrices.accuracy(y_test, y_pred)

print("Accuracy:", acc)