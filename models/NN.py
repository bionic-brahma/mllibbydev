import sys

import numpy as np
from utilities.Activations import activfunc

sys.path.insert(1, '../utilities')

np.random.seed(1)


class ANN:
    def __init__(self, x, y, learning_rate=0.001, layer_neurons=None):
        """
        This is a 4 layer neural network 1st layer is input layer, 2nd layer is hidden layer with 12  neurons
        3rd layer is hidden layer with  32 neurons, 4th layer is output layer with number of neurons equals
        to 1 if not multiclass, else neurons are equal to no. of different labels(if multiple labels).
        """
        if layer_neurons is None:
            layer_neurons = [12, 32, 1]

        self.input = x  # input data without output labels
        self.y = y  # input data of labels

        self.output = np.zeros(self.y.shape)
        self.learning_rate = learning_rate  # learning rate for neural network

        labels = np.unique(y)
        self.no_of_labels = len(labels)

        # for multiclass if labels are strings, then converting them into non string values.
        self.new_y = self.y
        for i in range(len(self.new_y)):
            a = list()
            self.new_y[i] = list(labels).index(self.new_y[i])

        # defining one hot labels if multiclass
        if self.no_of_labels > 2:
            self.one_hot_labels = np.zeros((x.shape[0], self.no_of_labels))
            for i in range(x.shape[0]):
                self.one_hot_labels[i, self.new_y[i][0]] = 1

        # Neural layers parameters
        self.weights1 = np.random.randn(x.shape[1], layer_neurons[0])  # weights for layer 1
        self.bais1 = np.random.randn(1, layer_neurons[0])  # biases for layer 1
        self.layer1_fx = 'Sigmoid'  # activation function used in layer 1

        self.weights2 = np.random.randn(layer_neurons[0], layer_neurons[1])  # weights for layer 2
        self.bais2 = np.random.randn(1, layer_neurons[1])  # # biases for layer 2
        self.layer2_fx = 'Sigmoid'  # activation function used in layer 2

        # output layer parameters if multiclass classification
        if self.no_of_labels > 2:
            self.weights3 = np.random.randn(layer_neurons[1], self.no_of_labels)  # weights for layer 3
            self.bais3 = np.random.randn(1, self.no_of_labels)  # biases for layer 3
            self.layer3_fx = 'Softmax'  # activation function used in layer 3
        # if not  multiclass classification
        else:
            self.weights3 = np.random.randn(layer_neurons[1], layer_neurons[2])
            self.bais3 = np.random.randn(1, layer_neurons[2])
            self.layer3_fx = 'Sigmoid'

    def feedforward(self):
        self.layer1 = activfunc(np.dot(self.input, self.weights1) + self.bais1, activation_type=self.layer1_fx)
        self.layer2 = activfunc(np.dot(self.layer1, self.weights2) + self.bais2, activation_type=self.layer2_fx)
        self.output = activfunc(np.dot(self.layer2, self.weights3) + self.bais3, activation_type=self.layer3_fx)

    def backprop(self):
        # delta3 = 2*(self.y - self.output) * sigmoid_derivative(self.output)

        # if multiclass
        if self.no_of_labels > 2:
            error = self.output - self.one_hot_labels
            delta3 = (error) * activfunc(self.output, activation_type='Softmax', deri=True)

            # if not multiclass
        else:
            error = self.y - self.output
            delta3 = (error) * activfunc(self.output, activation_type='Sigmoid', deri=True)

        d_weights3 = np.dot(self.layer2.T, delta3)

        delta2 = np.dot(delta3, self.weights3.T) * activfunc(self.layer2, activation_type='Sigmoid', deri=True)
        d_weights2 = np.dot(self.layer1.T, delta2)

        # delta1 = np.dot(delta2,self.weights2.T) * sigmoid_derivative(self.layer1)
        delta1 = np.dot(delta2, self.weights2.T) * activfunc(self.layer1, activation_type='Sigmoid', deri=True)
        d_weights1 = np.dot(self.input.T, delta1)

        self.weights1 += d_weights1 * self.learning_rate
        self.weights2 += d_weights2 * self.learning_rate
        self.weights3 += d_weights3 * self.learning_rate

    def train(self, epoch=5000):
        print("Training network {} times".format(epoch))
        for i in range(epoch):
            self.feedforward()
            self.backprop()
        print("Training completed")

    def predict(self, values):
        layer1 = activfunc(np.dot(values, self.weights1) + self.bais1, activation_type=self.layer1_fx)
        layer2 = activfunc(np.dot(layer1, self.weights2) + self.bais2, activation_type=self.layer2_fx)
        output = activfunc(np.dot(layer2, self.weights3) + self.bais3, activation_type=self.layer3_fx)
        return output

    '''
    ************ Using this Neural Network *********
    
    nn=ANN(X,labels)
    nn.train()
    print(nn.predict(values))
    
    '''
