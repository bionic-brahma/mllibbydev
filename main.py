from models.utilities.split import k_cross_validation_split
from models.utilities.performance_matrices import accuracy
import numpy as np
from models import LinearRegression

model = LinearRegression.regression(iteration=100, learning_rate=0.001, show_steps=False)
a = np.array([[1, 2], [2, 3], [2, 5], [6, 3], [7, 2], [8, 9], [1, 7], [11, 5]])
b = np.array([1, 2, 3, 4, 5, 6, 7, 8])
#a= [[1,2], [2,4], [4,2], [4,6], [1,7]]
b= np.array([3, 5, 7, 9, 9, 17, 8 , 16])
model.fit(a,b)
print(model.predict([2,9]),"\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")
k=3
res = k_cross_validation_split(a, b, k_value=k)

for i in range(k):
    model = LinearRegression.regression(iteration=100, learning_rate=0.001, show_steps=False)
    print("set number: ", i)
    print(res[0][i], "------>>", res[1][i])
    model.fit(res[0][i][0],res[0][i][1])
    print("predicted values:-----> ", model.predict(res[1][i][0]))
    print("\nAccuracy: ", accuracy(res[1][i][1],model.predict(res[1][i][0])))



