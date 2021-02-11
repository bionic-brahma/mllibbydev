import random
import numpy as np


def train_test_split_df(df, test_size=0.2, output_seperated='yes', label_name='label'):
    """
    ************** PARAMETER ***************
    df = dataset
    test_size = ratio of train_data  to test_data
    label_name = column heading where output labels are stored
    output_seperated =  output labels seperated in return or not

    ************* YIELD ****************

    output_separated='no':  return test_df, train_df

                return type : panda dataframes

                train_df : data used to train model   (containes X and Y)
                test_df : data used to test trained model  (containes X and Y)
    output_separated='yes':     return X_test, X_train, Y_test, Y_train

            return type :  numpy arrays

            X_test : data used to test trained model (contains feaured only)
            X_train : data used to train  model (contains feaured only)

            Y_test :  output labels used to test train model
            Y_train : output Labels used to train model

    ************  EXAMPLE ****************

    train, test = train_test_split_df(df,0.7,'no')
    xtest, xtrain,ytest,ytrain= train_test_split_df(df,0.7,'yes','label')
    """
    try:
        df = df.dropna()  # dropping empty columns
    except:
        pass

    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    if output_seperated == 'yes':
        Y_test = test_df.iloc[:, -1]  # y is the last column of the dataset , it contains labels
        Y_train = train_df.iloc[:, -1]

        X_test = test_df.iloc[:, :-1]  # X is the feature data without last column of the dataset
        X_train = train_df.iloc[:, :-1]

        return np.array(X_test), np.array(X_train), np.array(Y_test), np.array(Y_train)
    else:
        return test_df, train_df


# K folds splitting
def cross_validation_split(df, folds=5):
    """
    *******************   PARAMETER  ************

    dataset =  data input
    folds = No of times cross validation to be used

    ***************** YIELD ***************

    train_data_batches : list of lists of data for training
    test_data_batches : list of lists of data for testing

    ************************ EXAMPLE *******************
    train_array,test_array = cross_validation_split([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],5)

    train_array = [ [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 17, 18, 19, 20],
                    [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19],
                    [1, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 20],
                    [2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 18, 19, 20] ]

    test_array = [  [13, 15, 2, 10],
                    [20, 8, 18, 6],
                    [12, 9, 19, 5],
                    [4, 7, 3, 1],
                    [16, 17, 11, 14] ]

    """

    try:
        df = df.dropna()  # dropping empty columns
    except:
        pass

    fold_size = round(len(df) / folds)

    test_fold = list()  # list to contains folds for testing
    train_fold = list()  # list to contains fold for training

    for i in range(folds):
        indices = df.index.tolist()
        test_indices = random.sample(population=indices, k=fold_size)

        test_df = df.loc[test_indices]
        train_df = df.drop(test_indices)

        test_fold.append(test_df)  # list containing list of data for test
        train_fold.append(train_df)  # list containing list of data for train

    return test_fold, train_fold


def shuffle_data(X, y, seed=None):
    """
    Function for shuffling the data
    :param X: Input feature matrix
    :param y: list of the labels for the input matrix
    :param seed: seed for random function
    :return: suffled input feature matrix and shuffled label list
    """

    if seed:
        np.random.seed(seed)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    return X[indices], y[indices]


def train_test_split(X, y, test_size=0.3, shuffle=True):
    """
    The method is to split the give feature matrix and the labels for that
    into training and testing datasets.
    :param shuffle: If true the dataset indices will be shuffled before the splitting
    :param X: Feature matrix
    :param y: Labels for the featur matrix
    :param test_size: The relative size of the test dataset
    :return: train_X, test_X, train_y,  test_y
    """
    if shuffle:
        X, y = shuffle_data(X, y, random.randint(1, 1000))

    number_of_train_records = len(y) - int(len(y) // (1 / test_size))
    train_X, test_X = X[:number_of_train_records], X[number_of_train_records:]
    train_y, test_y = y[:number_of_train_records], y[number_of_train_records:]

    return train_X, test_X, train_y, test_y
