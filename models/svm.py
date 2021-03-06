########################################################
# Support vector machine by collaboration   #
# for Risk Latte Americas Inc.              #
########################################################

import numpy as np
from tqdm import tqdm
from models.utilities.split import OVOdatamaker, shuffle_data


class SVM_base:

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

    def get_model_params(self):
        """
        This method returns a dictionary of parameters
        :return: dictionary of parameters
        """
        model_para = dict()
        model_para["label"] = self.label_dict
        model_para["weights"] = self.w
        model_para["bias"] = self.b

        return model_para

    def load_model_para(self, model_para):
        """
        This method loads the weights and bias of the given model parameters
        and hence can do the transfer learning with compatible dictionary
        :param model_para: dictionary containing model parameters
        :return: None
        """
        self.w = model_para["weights"]
        self.b = model_para["bias"]
        self.label_dict = model_para["label"]

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
                "The number of lebels in the target variable is more than 2. please switch to SVMMultiClassifier.")
            return "[X]. Data is not for binary classifier."

        for i in range(len(unique)):
            self.label_dict[i] = unique[i]

        for i in range(len(y)):
            if y[i] == unique[0]:
                y[i] = 0
            else:
                y[i] = 1
        y = y.astype('int64')
        y_ = y
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
        self.models = list()

    def get_model_params(self):
        """
        This method returns a dictionary of parameters
        :return: dictionary of parameters
        """
        model_para = dict()
        model_para["model_params"] = [model.get_model_params() for model in self.models]

        return model_para

    def load_model_para(self, model_para):
        """
        This method loads the weights and bias of the given model parameters
        and hence can do the transfer learning with compatible dictionary
        :param model_para: dictionary containing model parameters
        :return: None
        """
        for model_params in model_para["model_params"]:
            model = SVM_base(learning_rate=self.lr, lambda_param=self.lambda_param, n_iters=self.n_iters)
            model.load_model_para(model_params)
            self.models.append(model)

    def fit(self, X, y):
        """
        Fits the tree as per the training data.
        :param X: Feature matrix of the data. format--> [[1st rec], [2nd record],...]
        :param y: Data labels for the corresponding feature matrix. format [1st label, 2nd label,...]
        :return: None
        """
        datasets_by_OVO = OVOdatamaker(X, y)
        self.models = list()
        for i in tqdm(range(len(datasets_by_OVO)), desc="Training..."):
            tempX = datasets_by_OVO[i][0]
            tempY = datasets_by_OVO[i][1]
            tempX, tempY = shuffle_data(tempX, tempY, len(datasets_by_OVO))
            model = SVM_base(learning_rate=self.lr, lambda_param=self.lambda_param, n_iters=self.n_iters)
            model.fit(tempX, tempY)
            self.models.append(model)

    def predict(self, X):
        """
        Make the classification for the give input feature vector
        :param X: Feature vector to make classification on
        :return: Label for the feature vector
        """
        predicted_labels = list()
        predicted_outputs = list()
        for i in range(len(self.models)):
            pred_y = self.models[i].predict(X)
            predicted_labels.append(pred_y)

        for j in range(len(X)):
            predicted_outputs.append(mode_in_labels(np.array(predicted_labels)[:, j]))

        return predicted_outputs


def mode_in_labels(arr):
    set_labels = set(arr)
    freq_dict = dict()
    for i in set_labels:
        freq_dict[i] = 0
    for i in arr:
        freq_dict[i] += 1
    maxi = 0
    keytoreturn = 0
    for key, value in freq_dict.items():
        if maxi <= value:
            keytoreturn = key
            maxi = value

    return keytoreturn
