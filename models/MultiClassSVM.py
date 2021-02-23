from statistics import mode
from models import svm
from models.utilities.split import OVOdatamaker, shuffle_data
import numpy as np

class MultiClassSVM:

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

    def fit(self, X, y):
        """
        Fits the tree as per the training data.
        :param X: Feature matrix of the data. format--> [[1st rec], [2nd record],...]
        :param y: Data labels for the corresponding feature matrix. format [1st label, 2nd label,...]
        :return: None
        """
        datasets_by_OVO = OVOdatamaker(X, y)
        self.models = list()
        for i in range(len(datasets_by_OVO)):
            tempX = datasets_by_OVO[i][0]
            tempY = datasets_by_OVO[i][1]
            tempX, tempY = shuffle_data(tempX, tempY, len(datasets_by_OVO))
            model = svm.SVM(learning_rate=self.lr, lambda_param=self.lambda_param, n_iters=self.n_iters)
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
            pred_y_list = list()

        for j in range(len(X)):

            predicted_outputs.append(mode(np.array(predicted_labels)[:,j]))

        #print(np.array(predicted_labels))
        return predicted_outputs

