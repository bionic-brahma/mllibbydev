import numpy as np
from models.utilities.split import train_test_split
from models.LogisticRegression import LogisticRegression
from models.utilities.pre_processing import convlist

X = np.array([[2, 3], [-1, -8], [9, 7], [8, 4], [-9, -6], [-6, -11]])
y = np.array([1, 0, 1, 1, 0, 0])

model = LogisticRegression(show_steps=True)
model.fit(X, y)
print(model.predict([[-1,-5]]))


