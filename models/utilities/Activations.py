import numpy as np

e = 2.718


def activfunc(Z, activation_type='ReLU', deri=False):
    """
        Activates the output of the neurons in neural network
        Parameters:
        (Type: numpy array) Z:---> sum added weight times input of neuron
        (Type: String) type:---> type of activation function to be used in network, ReLU/ LeakyReLu/ Sigmoid/Gaussian/Tanh/Sinc/Bentid/Sinusoid/Softmax/Swish/SoftPlus/Linear/Softsign
        (Type: float) deri:---> is derivate true: use derivated activation fucntion .

        Return:
        (Type: numpy array)
    """

    if activation_type == 'ReLU':
        if deri:
            return np.array([1 if i > 0 else 0 for i in np.squeeze(Z)])
        else:
            return np.array([i if i > 0 else 0 for i in np.squeeze(Z)])

    if activation_type == 'LeakyReLU':
        if deri:
            return np.where(Z < 0, 0.01, 1)
        else:
            return np.where(Z < 0, Z * 0.01, Z)

    elif activation_type == 'Sigmoid':
        if deri:
            return 1 / (1 + np.exp(-Z)) * (1 - (1 / (1 + np.exp(-Z))))
        else:
            return 1 / (1 + np.exp(-Z))

    elif activation_type == 'Gaussian':
        if deri:
            return -2 * Z * e ** ((-Z) ** 2)
        else:
            return e ** ((-Z) ** 2)

    elif activation_type == 'Tanh':
        if deri:
            return 1 - (np.tanh(Z)) ** 2
        else:
            return np.tanh(Z)

    elif activation_type == 'Sinc':
        if deri:
            return 1 if Z == 0 else (np.sin(Z)) / Z
        else:
            return 0 if Z == 0 else ((np.cos(Z)) / Z) - ((np.sin(Z)) / Z ** 2)

    elif activation_type == 'Bentid':  # bent identity actiation function
        if deri:
            return (Z / (2 * (np.sqrt((Z ** 2) + 1)))) + 1
        else:
            return (np.sqrt(((Z ** 2) + 1) - 1) / 2) + Z

    elif activation_type == 'Sinusoid':
        if deri:
            return np.sin(Z)
        else:
            return np.cos(Z)

    elif activation_type == 'Arctan':
        if deri:
            return 1 / ((Z ** 2) + 1)
        else:
            return np.arctan(Z)

    elif activation_type == 'Softsign':
        if deri:
            Z = abs(Z)
            return 1 / (1 + Z) ** 2
        else:
            Z = abs(Z)
            return Z / (1 + Z)

    elif activation_type == 'Linear':
        if deri:
            return 1
        else:
            return Z

    elif activation_type == 'Swish':
        Y = Z * 1 / (1 + np.exp(-Z))  # Y= X * sigmoid(X)
        if deri:
            return Y + (1 / (1 + np.exp(-Z))) * (1 - Y)
        else:
            return Y

    elif activation_type == 'Softmax':
        if deri:
            return 1
        else:
            e_x = np.exp(Z - np.max(Z))
            return e_x / e_x.sum(axis=1, keepdims=True)

    elif activation_type == 'SoftPlus':
        if deri:
            return 1 / (1 + (e ** -Z))
        else:
            return np.log(1 + (e ** Z))

    else:
        raise Exception('Invalid activation_type of Activation F(x)!')
