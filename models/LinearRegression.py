#######################################################################################
## Linear regression to work on binary classification                                 #
## Work by Devendra Kumar for Risk Latte AI Inc.                                      #
#######################################################################################

import numpy as np


class regression:
    def __init__(self, iteration=10, learning_rate=0.001, show_steps=False):
        """
        The constructor for linear regression class
        :param iteration: number of iterations to be performed in optimizing cost function
        :param learning_rate: the step for update in the weights in optimization process
        :param show_steps: If true, it will show the intermediate steps.
        """
        self.iterat = iteration
        self.lr = learning_rate
        self.show_steps = show_steps
        self.weights = None
        self.bias = None

    def fit(self, input_X, output_Y):
        """
        Fits the linear regressor as per the training data.
        :param input_X: Feature matrix of the data. format--> [[1st rec], [2nd record],...]
        :param output_Y: Data labels for the corresponding feature matrix. format [1st label, 2nd label,...]
        :return: None
        """
        nrecords, nfeatures = np.array(input_X).shape
        # num_non_linear_features= self.combination(self.degree+nfeatures-1,nfeatures-1)  #m-1+nCm-1

        self.weights = np.zeros(nfeatures)
        self.bias = 0.0

        for it in range(self.iterat):
            model = np.dot(input_X, self.weights) + self.bias
            loss = (1 / 2) * np.sum((model - output_Y) ** 2)

            self.weights = self.weights - self.lr * np.dot(np.transpose(input_X), (model - output_Y))
            self.bias -= self.lr * np.sum(model - output_Y)

            if self.show_steps:
                print("-------> iteration number: ", it, "  Loss: ", loss)
                print("-------> weights: ", self.weights)

    def predict(self, input_X):
        """
        Make the prediction for the give input feature vector
        :param input_X: Feature vector to make classification on
        :return: pritected value for the feature vector
        """
        return np.dot(input_X, self.weights) + self.bias  # [:, 0]

    def get_model_params(self):
        """
        This method returns a dictionary of parameters
        :return: dictionary of parameters
        """
        model_para = dict()
        model_para["weights"] = self.weights
        model_para["bias"] = self.bias

        return model_para

    def load_model_para(self, model_para):
        """
        This method loads the weights and bias of the given model parameters
        and hence can do the transfer learning with compatible dictionary
        :param model_para: dictionary containing model parameters
        :return: None
        """
        self.weights = model_para["weights"]
        self.bias = model_para["bias"]

