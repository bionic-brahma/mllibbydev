#######################################################################################
## Logistic regression to work on binary classification                               #
## Work by Devendra Kumar for Risk Latte AI Inc.                                      #
#######################################################################################
import sys
import numpy as np
from models.utilities.Activations import activfunc
sys.path.insert(1, '../utilities')


class LogisticRegression:

    def __init__(self, iteration=1000, learning_rate=0.001, show_steps=False, func='Sigmoid'):
        """Constructor for the LogisticRegression class.

    Parameter:

    (Type: Int) iteration:---> number of iterations to be performed in optimizing cost function
    (Type: Float) learning_rate:---> the step for update in the weights in optimization process
    (Type: Boolean) show_stepa:---> If true, it will show the intermediate stpes.
    (Type: string) func:---> choose activation function to be used (ReLU/ LeakyReLu/ Sigmoid/Gaussian/Tanh/Sinc/Bentid/Sinusoid/Softmax/Swish/SoftPlus/Linear/Softsign)

    Return:

    None
    """
        self.iter = iteration
        self.lr = learning_rate
        self.weights = None
        self.bias = None
        self.label_dict = dict()
        self.show_steps = show_steps
        self.func = func

    def fit(self, X, Y):
        """Computes the model and fits a logistic function on given data.

    Parameter:

    (Type: pd.DataFrame) X:---> feature matrix
    (Type: pd.DataFrame) Y:---> row containing labels for classes

    Return:

    None
    """
        unique = np.unique(Y)

        if len(unique) > 2:
            print(
                "The number of lebels in the target variable is more than 2. please switch to LogisticMultiClassifier.")
            return "[X]. Data is not for binary classifier."

        for i in range(len(unique)):
            self.label_dict[i] = unique[i]

        Y = np.where(Y == self.label_dict[0], 0, 1)

        n_rec, n_feat = X.shape
        # print("records= ", n_rec, "  features= ", n_feat)
        self.weights = np.zeros(n_feat)
        self.bias = 0.0

        for it in range(self.iter):

            # model= self.sigmoid(np.dot(X, self.weights)+self.bias)
            model = activfunc(np.dot(X, self.weights) + self.bias, activation_type=self.func)

            # print(model)
            log_loss = (1 / n_rec) * np.sum(-(np.dot(np.transpose(Y), np.log(model)) + np.dot(np.transpose((1 - Y)),
                                                                                              np.log(
                                                                                                  1 - model))))  # -[ylog(predicted_y)+(1-y)log(1-predicted_y)]

            for idx, x in enumerate(X):

                if Y[idx] <= 0:

                    self.weights = self.weights - self.lr * model[idx] * x
                    self.bias = self.bias - self.lr * model[idx]

                else:

                    self.weights = self.weights - self.lr * (model[idx] - 1) * x
                    self.bias = self.bias - self.lr * (model[idx] - 1)

            if self.show_steps:
                print("iteration: ", it, " loss: ", log_loss)
        # print("weights: ", self.weights)

    def predict(self, X):
        """Returns the predicted class label

    Parameter:

    (Type: pd.DataFrame) X:---> feature matrix

    Return:

    Predicted Class label

    """
        label_name = list()
        # labels = self.sigmoid(np.dot(X, self.weights)+self.bias)
        labels = activfunc(np.dot(X, self.weights) + self.bias, activation_type=self.func)

        for i in range(len(labels)):

            if 1 - labels[i] >= labels[i]:

                label_name.append(self.label_dict[0])

            else:

                if len(self.label_dict) == 1:

                    label_name.append(self.label_dict[0])

                else:

                    label_name.append(self.label_dict[1])

        return label_name

####################################################
####################################################
