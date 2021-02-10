import numpy as np


def labels_change(y):
    """
    this  function checks if a dataset contains a string values or not, and if string values are present it encodes it into integer .
    if multiple string labels are found it will encode the labels as 1, 2, 3, 4 .. ... , n  where n = no of labels

    Parameters:
    (Type  numpy_array) y -----> numpy array of label column

    Return:
    (Type numpy array)   Y ----> numpy array with encoded labels

    """

    labels = np.unique(y)  # np.unique(dataset_y_column)

    no_labels = len(labels)

    for it_label in labels:
        if isinstance(it_label, str):
            print("string labels are found in data_set")
            try:
                for i, j in enumerate(y):
                    if j == it_label:
                        y[i] = (labels.index(it_label) + 1)
                    elif j == labels[1]:
                        y[i] = 1
            except:
                print("couldn't properly encode the data ")
                pass
    return y
