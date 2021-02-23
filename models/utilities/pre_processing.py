###########################################################################
# The Program is a part of data processing module. 				      #
# Work by Devendra Kumar for Risk Latte AI Inc.                           #
###########################################################################

#######################################################################################
# Please disable the next block while amending this file so that proper
# test_out can happen.
#######################################################################################

# The block suppresses the warning on runtime
import warnings
import numpy as np

warnings.filterwarnings("ignore")


#######################################################################################

def subset_by_label():
    pass


# Function to split dataset for one vs res classification
def OVOdatamaker(X, y):
    """
    The methode to split dataset into the subsets, n*(n-1) ubstes to perform ovo using any
    binary classification method.
    here n is the number of unique classes present in the dataset.
    :param X: Dataset feature matrix containing features of the all records
    :param y: The dataset labels in list format
    :return: list containing data sets
    """
    subsets_to_return = list()
    unique = np.unique(y)
    for i in unique:
        X_sub_index = np.where(y == i, True, False)
        try:
            X_sub = X[X_sub_index]
            Y_sub = y[X_sub_index]
        except:
            X_sub = X.iloc[X_sub_index]
            Y_sub = y.iloc[X_sub_index]

        subsets_to_return.append((X_sub, Y_sub))
    return subsets_to_return







# Function to search key in dataset.
def SearchKeyIncolumns(dataset, key="NaN"):
    """Returns the frequency dictionary like histogram

    Parameters:
    (Type: pandas.DataFrame) dataset:---> Dataset where key will be searched.
    (Type: String) key:---> Key to be searched in dataset

    Return:
    (Type: Dictionary) column_search_freq_dictionary:---> The histogram like frequency dictionary
    """

    # Making a copy of data to maintain integrity of original data
    data = dataset.copy()

    # Taking columns name 
    cols = data.columns

    # This dictionary will store columns names as key and value as frequency of the occurence of "key"
    column_search_freq_dictionary = {}

    # Filling up the column_search_freq_dictionary
    for i in cols:

        count = 0

        for j in data[i]:

            if j == key:
                count = count + 1

        column_search_freq_dictionary[i] = count

    return column_search_freq_dictionary


# Function to filter the data by columns on key density basis. 
def RelevanceColumnFilter(dataset, key="NaN", filter_threshold=0.8):
    """Performs column deletion based of the density of key

    If the density of key in a any column will be greatee than threshold,
    then that column will be removed.

    Parameters:
    (Type: pandas.DataFrame) dataset:---> Dataset.
    (Type: String) key:---> Key to be searched in dataset.
    (Type: float) filter_threshold:---> Threshold for the density of key

    Return:
    (Type: pandas.Dataframe) data:---> Filtered dataset after removing columns
    """

    # Getting the number of occurrence of "key" column wise in Dictionary.
    dictionary = SearchKeyIncolumns(dataset, key)

    # Creating a copy of original data to maintain its integrity.
    data = dataset.copy()

    # This list will have column names that are not relevant for further processing. (based on threshold)
    columns_to_drop = []

    # Converting the threshold to another scale range [0, total number of records in data]
    filter_threshold = filter_threshold * len(data)

    # Filling columns_to_drop
    for i in dictionary.items():

        if i[1] >= filter_threshold:
            columns_to_drop.append(i[0])

    # Dropping the columns which have more keys than threshold.
    data.drop(columns=columns_to_drop, inplace=True)

    return data


# Function to handle missing values. 
def ReplaceMeanAndMode(dataset, key='NaN'):
    """Handles the missing values(NaN or any value given in key)

    It replaces the 'key' in columns with the mode or mean of the concerned
    column. Uses mode for categorical nominal attributes and uses mean for others

    Parameters:
    (Type: pandas.DataFrame) dataset:---> Dataset with missing values.
    (Type: String) key:---> Missing value key

    Return:
    (Type: pandas.DataFrame) data:---> Dataset with missing values handled.

    """

    # Creating a copy of original data to maintain its intigrity.   
    data = dataset.copy()

    # Taking columns name 
    cols = data.columns

    for i in cols:

        # Converting columns to object type.
        # Helps in maintain uniformity in calling associated functions
        data[i] = data[i].astype(object)

        try:

            # Check condition for the categorical feature. Runs only for numerical entry.
            # (Exceptional handling enabled in case data[i][0] does not have isalpha func)
            if not data[i][0].isalpha():

                # Calculating mean after casting entries to float.
                total = sum([float(x) for x in np.array(data[i]) if x != key])
                count = sum([1.0 for x in np.array(data[i]) if x != key])
                mean = total / count

                # Replacing key (missing value) with mean. 
                data[i].replace(key, mean, inplace=True)

            # This block runs for the categorical parameters
            else:

                data[i].replace(key, np.nan, inplace=True)

                # Replacing missing values, ie. keys with mode of column.
                data[i].replace(np.nan, data[i].mode(), inplace=True)

        except:

            print("Function in error: pre_processing.ReplaceMeanAndMode")
            print("[x]. Raw data file is not well defend. Reliability may get compromised")

    return data


# Function to find elements of one list that are not in other list. 
def Diff(list1, list2):
    """Returns the difference(List1-List is set notation) of two list.

    Parameters:
    (Type: list()) List1:---> First list.
    (Type: list()) List2:---> Second list.

    Return:
    (Type: List()) List1-List2

    """
    list_dif = [i for i in list1 + list2 if i not in list1 or i not in list2]

    return list_dif


# Function to divide parameters names in categorical and continuous types
def GetAttributesInCategory(dataset, target_attribute, include_target_attribute=False):
    """Aggregates the attributes of dataset in categorical and continuous type.

    Returns two lists of categorical and continuous attribute's names.

    Parameters: (Type: pandas.DataFrame) dataset:---> Dataset with mixed attributes. (Type: String)
    target_attribute:---> label attribute name or name of dependent variable. (Type: Boolean)
    include_target_attribute:---> If 'yes', then it takes target variable also in consideration while saggregating.

    Return:
    (Type: List(), List()) List with categorical attributes' names, List with continuous attributes names.

    """

    # Creating a copy of original data to maintain its integrity.
    data = dataset.copy()

    # Taking columns name 
    cols = data.columns

    # This list will have categorical attributes' names
    nominal_cols = []

    for i in cols:

        # Calculating mean after casting entries to float.
        data[i] = data[i].astype(object)

        try:

            # If true the dependent variable or target variable will also be classified as categorical or numerical
            if include_target_attribute:

                if i != target_attribute:

                    # Check condition for the categorical feature. Runs only for numerical entry.
                    # (Exceptional handling enabled in case data[i][0] does not have isalpha func)
                    if data[i][0].isalpha():
                        nominal_cols.append(i)

            # If target variable is exempted.
            else:

                # Check condition for the categorical feature. Runs only for numerical entry.
                # (Exceptional handling enabled in case data[i][0] does not have isalpha func)
                if data[i][0].isalpha():
                    nominal_cols.append(i)
        except:

            print("Function in error: pre_processing.GetAttributesInCategory")
            print("[x]. Raw data file is not well defend. Reliability may get compromised")

    templist = nominal_cols.copy()

    # Now templist will have the categorical column names and name of target variable
    templist.append(target_attribute)

    # The second return value is the name of the columns with numerical entries.
    return nominal_cols, Diff(list(data.columns.values), templist)


# Function to change the type of columns of the dataset
def ChangeType(dataset, cols):
    """Changes the type of columns in float64 dtype

    Parameters:
    (Type: pandas.DataFrame) dataset:---> Dataset where change of type is required.
    (Type: List()) cols:--->  list of the column names where type change is required.

    Return:
    (Type: pandas.DataFrame) data:---> Dataset with changed types.

    """

    # Creating a copy of original data to maintain its integrity.
    data = dataset.copy()

    for i in cols:
        # Changing type of each column into float64
        data[i] = data[i].astype(float)

    return data


# Function to standardize the independent variables
def standardization(inputx, record_param=True):
    """Performs the standardization on attributes in inplace manner

    Performs standardization and returns the dictionary with elements as tuples(mean, std. deviation)
    for each attributes.

    Parameters:
    (Type: pandas.DataFrame) x:---> Dataset(excluding target).
    (Type: Boolean) record_param:---> Indicates whether to return parameter dictionary.

    Return:
    (Type: dict={ Name of attribute : tuple(mean, std. deviation)) data:---> Dictionary of parameters of standardization

    """
    # Getting column names
    columns = inputx.columns

    # This dictionary will store the parameters used in standardization process
    standardization_parameters = {}

    for i in columns:

        p = np.array(inputx[i])
        mean_of_data = np.mean(p)
        std_devi_of_data = np.std(p)

        # performing standardization column wise
        inputx[i] = [(x - mean_of_data) / std_devi_of_data for x in p]

        # Removing columns(features) which have no deviation throughout.
        if std_devi_of_data == 0:

            inputx.drop(i, axis=1, inplace=True)

        else:

            # Writing on dictionary, the params for column
            standardization_parameters[i] = (mean_of_data, std_devi_of_data)

    # Only returns the standardization_parameters if record_param is true.
    if record_param:
        return standardization_parameters


def OneHotEncoding(column_name, dataset_in_df_form):
    """
    Converts are returns the one hot encoded columns for the given column.
    This is inplace operation.
    :param column_name: Name of the given column
    :param dataset_in_df_form: Dataset after taking it in dataframe form
    :return: dataframe with one hot encoded columns
    """

    unique_values = np.unique(dataset_in_df_form[column_name])

    for unique_value in unique_values:
        try:
            new_column = np.where(unique_value == dataset_in_df_form[column_name], 1, 0)
            dataset_in_df_form[column_name + "_" + str(unique_value)] = new_column
        except Exception as e:
            print("[X] Error in function: pre_processing.OneHotEncoding\n", e)
    return dataset_in_df_form


def convlist(a):
    """
    Converts the iterable element into a list.
    :param a: the iterable variable
    :return:  list having the elements of the iterable variable
    """
    if type(a) is not list():
        return [x for x in a]
    return a
################################# End of File ###################################
