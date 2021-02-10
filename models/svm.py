import numpy as np
import json
import ast


class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def Save_Model(self, file_name):

        model_data = {"model_param": str([self.w.tolist(), self.b])}
        model_file = file_name + str(".json")
        try:
            with open(model_file, 'x') as modelfile:
                json.dump(model_data, modelfile, indent=4)
                print("model_saved sucessfully in file named : ", model_file)
        except:

            with open(model_file, 'w') as modelfile:
                json.dump(model_data, modelfile, indent=4)
                print("model_saved sucessfully in file named : ", model_file)
        return

    def Load_Model(self, file_name):

        try:

            with open(file_name, "r") as model_file:
                data = json.load(model_file)

                params = ast.literal_eval(data["model_param"])

                self.w, self.b = params[0], params[1]
                print("model loaded sucessfully")
                return True

        except:

            print("model loading failed")
            return False
