##########################################################
#  The neural network by Devendra in collaboration
#  for Risk Latte Americas Inc.
##########################################################

#
 #    Please do consider testing with many datasets and
#            provide feedback
 #

import sys
from tqdm import tqdm
import numpy as np
from models.utilities.Activations import activfunc


def cross_entropy(label, prediction):
    """
    Cross Entropy loss function
    :param label: list of actual labels
    :param prediction: list of predicted labels
    :return: cross entropy
    """

    product = np.multiply(prediction, label)
    temp = product[product != 0]
    entropy = -np.log(temp)
    _cross_entropy = np.mean(entropy)

    return _cross_entropy


class NeuralNet:

    def __init__(self, n_inputs=None, n_outputs=None, hidden_layers=None, activations_for_layers=None):

        """
        This is the constrictor of the NeuralNet class.
        all the parameters are optional.

        :param: n_inputs: The total number of records in the dataset
        :param: n_outputs: The total number of classes or labels
        :param: hidden_layers: list where length of list gives the number of hidden layers and ith entry in the list
                indicates the number of neurons present in the ith hidden layer.
        :param: activations_for_layers: The list containing the activation functions for hidden layers
                Ex: ['ReLu', 'Softmax', 'Sigmoid', 'SoftPlus', 'Swish', 'Linear']  this is for a six hidden layer
                network. Its just an example without relevance of applicability.
                If nothing is provided. The 'LeakyReLU' will be used by default.
        :variable: W: Weights assigned to the layers
        :variable: B: Bias assigned to the layers"""

        self.loss = {}
        self.Flattened_Input = {}
        self.Layer_Output = {}
        self.dLayer_Output = {}
        self.dFlattened_Input = {}
        self.dBias = {}
        self.dWeights = {}
        if hidden_layers is None:
            hidden_layers = [5, 5, 3]

        self.number_of_Inputs = n_inputs
        self.number_of_Outputs = n_outputs
        self.Number_Hidden_Layers = len(hidden_layers)
        self.sizes = [self.number_of_Inputs] + hidden_layers + [self.number_of_Outputs]
        self.Weights = {}
        self.Bias = {}

        for i in range(self.Number_Hidden_Layers + 1):

            # random weights initialization
            self.Weights[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])

            # bias initialization to 0
            self.Bias[i + 1] = np.zeros((1, self.sizes[i + 1]))

    def neural_architecture(self, x):
        """
        Creates the architecture for the neural net with the parameters given in the class
        constructor.
        :param: x: Number of inputs flattened
        :param: Layer_Output: It gives dot product of H(Inputs) and W(Weights) Which is added to vector B(Bias)
        """
        self.Flattened_Input[0] = x.reshape(1, -1)

        for i in range(self.Number_Hidden_Layers):

            self.Layer_Output[i + 1] = np.matmul(self.Flattened_Input[i], self.Weights[i + 1]) + self.Bias[i + 1]

            self.Flattened_Input[i + 1] = activfunc(self.Layer_Output[i + 1], 'LeakyReLU')

        self.Layer_Output[self.Number_Hidden_Layers + 1] = np.matmul(self.Flattened_Input[self.Number_Hidden_Layers], self.Weights[self.Number_Hidden_Layers + 1]) + self.Bias[self.Number_Hidden_Layers + 1]

        if self.number_of_Outputs > 1:

            self.Flattened_Input[self.Number_Hidden_Layers + 1] = activfunc(self.Layer_Output[self.Number_Hidden_Layers + 1], 'Softmax')

        else:

            self.Flattened_Input[self.Number_Hidden_Layers + 1] = activfunc(self.Layer_Output[self.Number_Hidden_Layers + 1], 'Sigmoid')

        return self.Flattened_Input[self.Number_Hidden_Layers + 1]

    def backpropagation(self, x, y):
        """
        Function performing the backpropagation to update weights

        :param: x: Input data matrix (all entries should be numerical)
        :param: y: Output data labels

        """
        # creating and initialising the neural net with weights and biases along with input.
        self.neural_architecture(x)

        L = self.Number_Hidden_Layers + 1

        # It contains the differentiation of the loss calculated at the last layer
        self.dLayer_Output[L] = (self.Flattened_Input[L] - y)

        for k in range(L, 0, -1):

            self.dWeights[k] = np.matmul(self.Flattened_Input[k - 1].T, self.dLayer_Output[k])
            self.dBias[k] = self.dLayer_Output[k]

            self.dFlattened_Input[k - 1] = np.matmul(self.dLayer_Output[k], self.Weights[k].T)
            if k ==  L-1:
                self.dLayer_Output[k - 1] = np.multiply(self.dFlattened_Input[k - 1], activfunc(self.Flattened_Input[k - 1],'Softmax', deri=True))
            else:
                self.dLayer_Output[k - 1] = np.multiply(self.dFlattened_Input[k - 1], activfunc(self.Flattened_Input[k - 1],'LeakyReLU', deri=True))

    def fit(self, X, Y, epochs=100, initialize_with_random_weights='True', learning_rate=0.001, show_loss=False):

        """
        Function for training neural network using the data given.

        :param: X: data feature matrix
        :param: Y: list of the lables for the feature matrix
        :param: epochs: Number of iteration to be performed by the network
        :param: learning_rate: The step value, it decides how much to move at every iteration
        :param: show_loss: It displays the loss in the model

        """
        self.number_of_Inputs = X.shape[0]

        if initialize_with_random_weights:

            for i in range(self.Number_Hidden_Layers + 1):

                self.Weights[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
                self.Bias[i + 1] = np.zeros((1, self.sizes[i + 1]))

        for epoch in tqdm(range(epochs), file=sys.stdout, unit="epoch", desc="Training"):

            dWeights = {}
            dBias = {}

            for i in range(self.Number_Hidden_Layers + 1):

                dWeights[i + 1] = np.zeros((self.sizes[i], self.sizes[i + 1]))
                dBias[i + 1] = np.zeros((1, self.sizes[i + 1]))

            for x, y in zip(X, Y):

                self.backpropagation(x, y)

                for i in range(self.Number_Hidden_Layers + 1):

                    dWeights[i + 1] += self.dWeights[i + 1]
                    dBias[i + 1] += self.dBias[i + 1]

            m = X.shape[1]

            for i in range(self.Number_Hidden_Layers + 1):

                self.Weights[i + 1] -= learning_rate * (dWeights[i + 1] / m)
                self.Bias[i + 1] -= learning_rate * (dBias[i + 1] / m)

            if show_loss:
                """ 
                Loss value is displayed which is cross entropy value
                """
                Y_pred = self.predict(X)
                self.loss[epoch] = cross_entropy(Y, Y_pred)
                print("Loss:", self.loss[epoch])

    def predict(self, X):
        """
        Function of the neural network used for predicting the test cases or validation cases
        :param: X: Input Column or variable( Input test data)
        :return: returns the labels for the given feature matrix
        """
        Y_pred = []

        for x in X:

            y_pred = self.neural_architecture(x)
            Y_pred.append(y_pred)

        return np.array(Y_pred).squeeze()

    # ************ Using this Neural Network *********
