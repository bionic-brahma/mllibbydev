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

'''
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
'''


def cross_entropy(p, q):
    return -sum([p[i] * np.log(q[i]) for i in range(len(p))])


class NeuralNet:

    def __init__(self, n_inputs, n_outputs, hidden_layers=None, activations_for_layers=None):

        """
        This is the constrictor of the NeuralNet class.
        all the parameters are optional.

        :param: n_inputs: The total number of features in one record in the dataset
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
        self.label_dict = dict()
        if hidden_layers is None:
            hidden_layers = [2, 2]
        self.actual_prob = list()
        self.number_of_Inputs = n_inputs
        self.number_of_Outputs = n_outputs
        self.Number_Hidden_Layers = len(hidden_layers)
        self.sizes = [self.number_of_Inputs] + hidden_layers + [self.number_of_Outputs]
        self.Weights = list()
        self.Bias = list()
        print(self.Number_Hidden_Layers)
        # self.Weights = np.zeros((self.Number_Hidden_Layers+2, self.Number_Hidden_Layers+2, self.Number_Hidden_Layers+2))
        self.Weights.append(0)
        self.Bias.append(0)
        for i in range(self.Number_Hidden_Layers + 1):
            # random weights initialization
            self.Weights.append(np.random.randn(self.sizes[i], self.sizes[i + 1]))

            # bias initialization to 0
            self.Bias.append(np.zeros((1, self.sizes[i + 1])))

    def neural_architecture(self, x):
        """
        Creates the architecture for the neural net with the parameters given in the class
        constructor.
        :param: x: Number of inputs flattened
        :param: Layer_Output: It gives dot product of H(Inputs) and W(Weights) Which is added to vector B(Bias)
        """
        self.Flattened_Input[0] = np.array(x).reshape(1, -1)

        for i in range(self.Number_Hidden_Layers):
            # print("\nfalttened_input[i]:", self.Flattened_Input[i])
            # print("weigghyt_input[i+1]:", self.Flattened_Input[i])
            # print("falttened_input[i+1]:", self.Flattened_Input[i+1])
            # print(self.Layer_Output)
            self.Layer_Output[i + 1] = np.matmul(self.Flattened_Input[i], np.transpose(self.Weights[i + 1])) + \
                                       self.Bias[i + 1]

            self.Flattened_Input[i + 1] = activfunc(self.Layer_Output[i + 1], 'LeakyReLU')

        self.Layer_Output[self.Number_Hidden_Layers + 1] = np.matmul(self.Flattened_Input[self.Number_Hidden_Layers],
                                                                     np.transpose(
                                                                         self.Weights[self.Number_Hidden_Layers + 1])) + \
                                                           self.Bias[self.Number_Hidden_Layers + 1]

        if self.number_of_Outputs > 1:

            self.Flattened_Input[self.Number_Hidden_Layers + 1] = activfunc(
                self.Layer_Output[self.Number_Hidden_Layers + 1], 'Softmax')

        else:

            self.Flattened_Input[self.Number_Hidden_Layers + 1] = activfunc(
                self.Layer_Output[self.Number_Hidden_Layers + 1], 'Sigmoid')

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
        # print("y ===", y)
        self.dLayer_Output[L] = (self.Flattened_Input[L] - y)

        for k in range(L, 0, -1):

            self.dWeights[k] = np.matmul(self.Flattened_Input[k - 1].T, self.dLayer_Output[k])
            self.dBias[k] = self.dLayer_Output[k]

            self.dFlattened_Input[k - 1] = np.matmul(self.dLayer_Output[k], self.Weights[k].T)
            if k == L - 1:
                self.dLayer_Output[k - 1] = np.multiply(self.dFlattened_Input[k - 1],
                                                        activfunc(self.Flattened_Input[k - 1], 'Softmax', deri=True))
            else:
                self.dLayer_Output[k - 1] = np.multiply(self.dFlattened_Input[k - 1],
                                                        activfunc(self.Flattened_Input[k - 1], 'LeakyReLU', deri=True))

    def fit(self, X, Y, epochs=100, initialize_with_random_weights='True', learning_rate=0.001, show_loss=False):

        """
        Function for training neural network using the data given.

        :param: X: data feature matrix
        :param: Y: list of the lables for the feature matrix
        :param: epochs: Number of iteration to be performed by the network
        :param: learning_rate: The step value, it decides how much to move at every iteration
        :param: show_loss: It displays the loss in the model

        """

        self.number_of_Inputs = np.array(X).shape[0]

        unique = np.unique(Y)
        for i in range(len(unique)):
            self.label_dict[i] = unique[i]

        sum = dict()
        temp_sum = 0
        for j in range(len(self.label_dict)):
            for i in range((len(Y))):
                if Y[i] == self.label_dict[j]:
                    Y[i] = j
                    temp_sum += 1
            sum[j] = temp_sum
            temp_sum = 0

        for i in sum.values():
            self.actual_prob.append(i / len(Y))

        # print(self.label_dict)

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
            # print(Y)
            for x, y in zip(X, Y):

                self.backpropagation(x, y)

                for i in range(self.Number_Hidden_Layers + 1):
                    dWeights[i + 1] += self.dWeights[i + 1]
                    dBias[i + 1] += self.dBias[i + 1]

            m = np.array(X).shape[1]

            for i in range(self.Number_Hidden_Layers + 1):
                self.Weights[i + 1] -= learning_rate * (dWeights[i + 1] / m)
                self.Bias[i + 1] -= learning_rate * (dBias[i + 1] / m)

            if show_loss:
                """ 
                Loss value is displayed which is cross entropy value
                """
                Y_pred = self.predict(X)
                self.loss[epoch] = cross_entropy(self.actual_prob, Y_pred)
                print("Loss:", self.loss[epoch])

    def predict(self, X):
        """
        Function of the neural network used for predicting the test cases or validation cases
        :param: X: Input Column or variable( Input test data)
        :return: returns the labels for the given feature matrix
        """
        Y_pred = []
        confidence_pred_list = []
        for x in X:
            confidence = dict()
            # print("x turns for prdeiction: ", x)
            y_pred = self.neural_architecture(x)
            id = 0
            for i in np.squeeze(y_pred):
                confidence[self.label_dict[id]] = i
                id += 1

            most_probable_pred = self.label_dict[np.argmax(y_pred)]
            Y_pred.append(most_probable_pred)
            confidence_pred_list.append(confidence)
        # print(confidence_pred_list)
        return Y_pred

    # ************ Using this Neural Network *********
