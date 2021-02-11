

##########################################################################################

import pandas as pd
import numpy as np
from models.utilities.pre_processing import OneHotEncoding
from models.utilities.split import train_test_split
from models.NN import NeuralNet
from models.utilities.performance_matrices import accuracy

"""Label Encoding"""
data_frame = pd.read_csv("/home/devendra/Desktop/work/splice.data", header=None, names=["Type", "Name", "Sequence"])

labels = {'Type': {"EI": 0, "IE": 1, "N": 2}}
data_frame = data_frame.replace(labels)
y = data_frame['Type']
data_frame_one_hot_encoded = OneHotEncoding('Sequence', data_frame)  # one hot encoding the dataframe
Input_Column = data_frame_one_hot_encoded.drop(['Name', 'Type', 'Sequence'],
                                               axis=1)  # Removing unwanted rows from the dataframe
Input_Column = np.asarray(Input_Column)  # changing the dataframe into array
X_train, X_test, y_train, y_test = train_test_split(Input_Column, y, test_size=0.05)  # Splitting the data into test and train(75:25)
num_inputs = Input_Column.shape[0]  # extracting total number of inputs
input_size = Input_Column.shape[1]  # extracting total number of features
n_inputs = X_test.shape[1]  # Extracting number of input values in the test data
y_OH_train = pd.get_dummies(y_train)  # Making dummies for the categories inside the y variable
y_OH_val = pd.get_dummies(y_test)
y_OH_train = np.expand_dims(y_OH_train, 1)  # Expanding dimensions of the array
y_OH_val = np.expand_dims(y_OH_val, 1)
print(y_OH_train.shape, y_OH_val.shape)
n_outputs = y_OH_val.shape[2]  # Storing the expanded dimensions in the n_outputs variable
ffsn_multi = NeuralNet(n_inputs, n_outputs, [2, 3, 4, 4, 5, 6])  # fitting the data to the model
ffsn_multi.fit(X_train, y_OH_train, epochs=100, learning_rate=0.001, show_loss=True)
Y_pred_train = ffsn_multi.predict(X_train)
Y_pred_train = np.argmax(Y_pred_train, 1)  # taking the max probability value from the output after applying softmax
Y_pred_val = ffsn_multi.predict(X_test)
Y_pred_val = np.argmax(Y_pred_val, 1)
accuracy_train = accuracy(Y_pred_train, y_train)
accuracy_val = accuracy(Y_pred_val, y_test)
print("Training accuracy", round(accuracy_train, 2))
print("test accuracy", round(accuracy_val, 2))

print(Y_pred_val)
