########################################################
# Support vector machine by collaboration
# for Risk Latte Americas Inc.
########################################################

import numpy as np
import json
import ast


class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        This the constructor for SVM class.

        :param learning_rate: step size for weights and bias to update.
        :param lambda_param: Lagrange multiplies' value
        :param n_iters:  Number of iterations for training
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.label_dict = dict()

    def fit(self, X, y):
        """
        Fits the tree as per the training data.
        :param X: Feature matrix of the data. format--> [[1st rec], [2nd record],...]
        :param y: Data labels for the corresponding feature matrix. format [1st label, 2nd label,...]
        :return: None
        """
        n_samples, n_features = np.array(X).shape

        unique = np.unique(y)

        if len(unique) > 2:
            print(
                "The number of lebels in the target variable is more than 2. please switch to LogisticMultiClassifier.")
            return "[X]. Data is not for binary classifier."

        for i in range(len(unique)):
            self.label_dict[i] = unique[i]

        #print(self.label_dict)
        y_ = y

        for i in range(len(y)):
            if y[i] == unique[0]:
                y[i] = 0
            else:
                y[i] = 1

        for i in range(len(y)):
            if y[i] <= 0:
                y_[i] = -1
            else:
                y_[i] = 1

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
        """
        Make the classification for the give input feature vector
        :param X: Feature vector to make classification on
        :return: Label for the feature vector
        """
        approx = np.dot(X, self.w) - self.b
        out = np.array(np.sign(approx))

        labels = list()

        if out.size == 1:
            if out == -1:
                labels.append(self.label_dict[0])
            else:
                labels.append(self.label_dict[1])
        else:
            for i in range(out.size):
                if out[i] == -1:
                    labels.append(self.label_dict[0])
                else:
                    labels.append(self.label_dict[1])
        return labels


'''
    def Save_Model(self, file_name):
        """
        This function saves the trained model in .json format.
        :param file_name: File name (Without extension) along with the location address to save the model.
        :return:
        

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

    def Load_Model(self, file_name):
        """
        This function loads the already saved trained model.
        :param file_name: location along with the file name(with extension)
        :return: Boolean value, true is model loaded successfully or else false.
        """

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
            '''''
