###########################################################################
# The Program is a part of data processing module. 				      #
# Work by Devendra Kumar for Risk Latte AI Inc.                           #
###########################################################################

#######################################################################################
# Please disable the next block while amending this file so that proper
# test_out can happen.
#######################################################################################
import warnings
import numpy as np
import random
import math
from random import randint
import models.utilities.split as sp
import re

# The block suppresses the warning on runtime

warnings.filterwarnings("ignore")


#######################################################################################

def datetime_formator(datetime, format_given=None):
    """
    This method converts the given formate of date - time to the standard format.

    :param datetime: datetime which is needed to be converted.
    :param format_given: format of the given date time
                            MM - > months
                            YYYY -> year
                            DD -> day
                            hh -> hours
                            mm -> minutes
                            ss -> seconds
    :return: reformatted date-time in string form
    """
    date_time_detailes_dict = dict()
    keys_for_date_time = ["day", "month", "year", "hour", "minute", "second", "milliseconds"]
    datetime_copy = datetime
    datetime_splited = re.split(':|/| |', datetime_copy)

    if format_given is None:

        for i in range(len(keys_for_date_time)):
            try:
                date_time_detailes_dict[keys_for_date_time[i]] = datetime_splited[i]
            except:
                pass

    else:
        date_time_formating = re.split(":|/|\\|-| |.|,|  ", format_given)

        for i in range(len(datetime_splited)):
            try:
                if date_time_formating[i] == "MM":
                    date_time_detailes_dict["month"] = datetime_splited[i]
                if date_time_formating[i] == "DD":
                    date_time_detailes_dict["day"] = datetime_splited[i]
                if date_time_formating[i] == "YYYY":
                    date_time_detailes_dict["year"] = datetime_splited[i]
                if date_time_formating[i] == "hh":
                    date_time_detailes_dict["hour"] = datetime_splited[i]
                if date_time_formating[i] == "mm":
                    date_time_detailes_dict["minute"] = datetime_splited[i]
                if date_time_formating[i] == "ss":
                    date_time_detailes_dict["second"] = datetime_splited[i]
            except:
                pass
    output_date_time_formating = str(date_time_detailes_dict["day"]) + "-" + str(date_time_detailes_dict["month"]) + "-" \
                                 + str(date_time_detailes_dict["year"]) + " " + str(date_time_detailes_dict["hour"]) + \
                                 ":" + str(date_time_detailes_dict["minute"]) + ":" + str(
        date_time_detailes_dict["second"])

    return output_date_time_formating


class oversampling:

    def __init__(self, method=None):
        """
        Constructor for the smote class
        """
        self.method = method
        if self.method is None:
            self.method = "smote"
        self.index_new = 0
        self.synthetic_data = list()

    def generate_records(self, No_of_records, i, indices, minority_class, k_points):
        """
        This method generates the data for the output dataset.

        :param No_of_records: number of records to be generated
        :param i: the index of the record around which the new records needed to be generated
        :param indices: the record that is indexed
        :param minority_class: sample data, usually the minority class data feature matrix
        :param k_points: number of nearest points that are to be considered.
        :return: None
        """

        while No_of_records != 0:
            features = len(minority_class[0])
            arr = []
            nn = randint(0, k_points - 2)

            for attr in range(features):
                difference = minority_class[indices[nn]][attr] - minority_class[i][attr]
                gap = random.uniform(0, 1)

                if self.method == "adasyn":
                    randomness_in_features = random.gauss(difference / 2, 0.1 * difference)
                    new_feat = minority_class[i][attr] + gap * difference + randomness_in_features
                elif self.method == "smote":
                    new_feat = minority_class[i][attr] + gap * difference
                else:
                    print("[X]Error: Please take method for upsampling - smote or adasyn")
                    return
                arr.append(new_feat)

            self.synthetic_data.append(arr)
            self.index_new = self.index_new + 1
            No_of_records = No_of_records - 1

    def k_neighbors(self, euclid_distance, k):
        """
        calculates the distance between points and returns k closest points
        :param euclid_distance:
        :param k: number of nearest points that are to be considered.
        :return: k points that are closest
        """
        nearest_idx_npy = np.empty([euclid_distance.shape[0], euclid_distance.shape[0]], dtype=np.int64)

        for i in range(len(euclid_distance)):
            idx = np.argsort(euclid_distance[i])
            nearest_idx_npy[i] = idx
            idx = 0

        return nearest_idx_npy[:, 1:int(k)]

    def find_k(self, X, k):
        """
        Finds k nearest neighbors using euclidean distance
        :param X: the feature matrix
        :param k: number of nearest neighbors
        :return: The k nearest neighbor
        """

        euclid_distance = np.empty([np.array(X).shape[0], np.array(X).shape[0]], dtype=np.float32)

        for i in range(len(X)):
            dist_arr = []
            for j in range(len(X)):
                dist_arr.append(math.sqrt(sum((np.array(X)[j] - np.array(X)[i]) ** 2)))
            dist_arr = np.asarray(dist_arr, dtype=np.float32)
            euclid_distance[i] = dist_arr

        return self.k_neighbors(euclid_distance, k)

    def generate_synthetic_points(self, minority_data_sample, percentage_of_data_increased, k_points):
        """
        This method generates (percentage_of_data_returned/100) * minority_data_sample synthetic minority samples.
        :param minority_data_sample: sample data, usually the minority class data feature matrix
        :param percentage_of_data_increased: this tells how much synthetic data will be return in addition to
                                            number of original records.
        :param k_points: number of nearest points that are to be considered.
        :return: (percentage_of_data_returned/100) * minority_data_sample synthetic minority samples.
        """
        percentage_of_data_returned = 100 + percentage_of_data_increased
        if percentage_of_data_returned < 100:
            raise ValueError("increase in data cannot be less than 0%")

        if k_points > np.array(minority_data_sample).shape[0]:
            raise ValueError("Size of k_points cannot exceed the number of samples.")

        percentage_of_data_returned = int(percentage_of_data_returned / 100)
        T = np.array(minority_data_sample).shape[0]

        indices = self.find_k(minority_data_sample, k_points)

        for i in range(indices.shape[0]):
            self.generate_records(percentage_of_data_returned, i, indices[i], minority_data_sample, k_points)

        return np.asarray(self.synthetic_data)


def auto_oversample(x, y, class_ratio_threshold=0.3, increase_to_ratio=0.60, kpoint=5):
    """

    :param x:
    :param y:
    :param class_ratio_threshold:
    :param increase_to_ratio:
    :return:
    """
    increase_to_ratio = 100 * increase_to_ratio
    synth = oversampling()
    total_number_of_rec = len(y)
    subsets, labels_order = sp.subsets_by_label(x, y)
    # print(subsets)
    label_rec_counts = dict()
    classes_to_oversample = list()

    for label in labels_order:
        label_rec_counts[label] = 0

    returnX = None
    returny = None

    for class_recods_set, label in zip(subsets, labels_order):

        label_rec_counts[label] = len(class_recods_set[0])
        #print("Label:-->", label, "  Count:-->", label_rec_counts[label])
        tempx = class_recods_set[0]
        tempy = class_recods_set[1]
        if label_rec_counts[label] < class_ratio_threshold * total_number_of_rec:
            classes_to_oversample.append(label)
            #print(increase_to_ratio, total_number_of_rec, label_rec_counts[label])
            percent_increase = 100 * (increase_to_ratio * total_number_of_rec - 100 * label_rec_counts[label]) \
                               / (100 * label_rec_counts[label] - increase_to_ratio * label_rec_counts[label])
            #print("percent increace=", percent_increase)

            generated_syth_data = synth.generate_synthetic_points(class_recods_set[0],
                                                                  percentage_of_data_increased=percent_increase,
                                                                  k_points=kpoint)
            #print(generated_syth_data)
            tempx = generated_syth_data
            tempy = [label for _ in range(len(generated_syth_data))]

        if returny is None:
            returny = tempy
        else:
            returny = np.concatenate((tempy, returny), axis=0)
        if returnX is None:
            returnX = tempx
        else:
            returnX = np.concatenate((tempx, returnX), axis=0)

    return returnX, returny


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
