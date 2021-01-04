#######################################################################################
## The Program is an application specific to the thyroid classification problem.      #
##                                                                                    #
## Work by Devendra Kumar for Risk Latte AI Inc.                                      #
#######################################################################################

#########################################
# Please disable the next block while ammending this file so that proper
# test can happen.
#########################################

# The block suppresses the warning on runtime
import warnings
warnings.filterwarnings("ignore")
#########################################



import pandas as pd
import numpy as np
import seaborn as sb

# Function to search key in dataset. 
def SearchKeyInColumns(dataset, key="NaN"):
    
    '''Returns the frequency dictionary like histogram
    
    Parameters:
    (Type: pandas.DataFrame) dataset:---> Dataset where key will be searched.
    (Type: String) key:---> Key to be searched in dataset
    
    Return:
    (Type: Dictionary) column_search_freq_dictionary:---> The histogram like frequency dictionary
    '''
    
    # Making a copy of data to maintain intrigity of original data
    data= dataset.copy()

    # Taking columns name 
    cols=data.columns

    # This dictionary will store columns names as key and value as frequency of the occurence of "key"
    column_search_freq_dictionary={}
    
    # Filling up the column_search_freq_dictionary
    for i in cols:
        
        count=0
        
        for j in data[i]:
            
            if j==key:
                
                count=count+1
                
        column_search_freq_dictionary[i]= count
        
    return column_search_freq_dictionary


# Function to filter the data by columns on key density basis. 
def RelevanceColumnFilter(dataset, key="NaN", filter_threshold=0.8):
    
    '''Performs column deletion based of the density of key
    
    If the density of key in a any column will be greatee than threshold,
    then that column will be removed.
    
    Parameters:
    (Type: pandas.DataFrame) dataset:---> Dataset.
    (Type: String) key:---> Key to be searched in dataset.
    (Type: float) filter_threshold:---> Threshold for the density of key
    
    Return:
    (Type: pandas.Dataframe) data:---> Filtered dataset after removing columns
    '''
    
    # Getting the number of occurence of "key" column wise in Dictionary.
    dictionary= SearchKeyInColumns(dataset, key)

    # Creating a copy of original data to maintain its intigrity.
    data= dataset.copy()

    # This list will have column names that are not relevant for further processing. (based on threshold)
    columns_to_drop= []

    # Converting the threshold to another scale range [0, total number of records in data]
    filter_threshold= filter_threshold*len(data)
    
    # Filling columns_to_drop
    for i in dictionary.items():
        
        if i[1]>=filter_threshold:
            
            columns_to_drop.append(i[0])
    
    # Droping the columns which have more keys than threshold. 
    data.drop(columns=columns_to_drop, inplace=True)
    
    return data


# Function to handle missing values. 
def ReplaceMeanAndMode(dataset, key='NaN'):
    
    '''Handles the missing values(NaN or any value given in key)
    
    It replaces the 'key' in columns with the mode or mean of the concerned 
    column. Uses mode for categorical nominal attributes and uses mean for others
    
    Parameters:
    (Type: pandas.DataFrame) dataset:---> Dataset with missing values.
    (Type: String) key:---> Missing value key
    
    Return:
    (Type: pandas.DataFrame) data:---> Dataset with missing values handled.

    '''

    # Creating a copy of original data to maintain its intigrity.   
    data= dataset.copy()
    
    # Taking columns name 
    cols= data.columns
    
    for i in cols:    
        
        # Converting columns to object type.
        # Helps in maintaing uniformity in calling associated functions
        data[i]=data[i].astype(object)
        
        try:
            
            # Check condition for the categorical feature. Runs only for numerical entry.
            # (Exceptional handling anabled in case data[i][0] does not have isalpha func)
            if not data[i][0].isalpha():

                total=0
                count=0

                # Calculating mean after casting entries to float.
                total= sum([float(x) for x in np.array(data[i]) if x != key])
                count= sum([1.0 for x in np.array(data[i]) if x != key])
                mean=total/count

                # Replacing key (missing value) with mean. 
                data[i].replace(key, mean,inplace=True)

            # This block runs for the categorical parameters
            else:

                data[i].replace(key,np.nan,inplace=True)

                # Replacing missing values, ie. keys with mode of column.
                data[i].replace(np.nan, data[i].mode(), inplace=True)
                
        except:
            
            print("[X]. Raw data file is not well defiend. Reliability may get compromised")
    
    return data


# Function to find elements of one list that are not in other list. 
def Diff(list1, list2):
    
    '''Returns the difference(List1-List is set notation) of two list.
    
    Parameters:
    (Type: list()) List1:---> First list.
    (Type: list()) List2:---> Second list.
    
    Return:
    (Type: List()) List1-List2
    
    '''   
    list_dif = [i for i in list1 + list2 if i not in list1 or i not in list2]
        
    return list_dif

   
# Function to divide paramenters names in categorical and continuous types 
def GetAttributesInCategory(dataset, target_attribute, include_target_attribute=False):
    
    '''Saggregates the attributes of dataset in categorical and continuous type.
    
    Returns two lists of categorical and continuous attribute's names.
    
    Parameters:
    (Type: pandas.DataFrame) dataset:---> Dataset with mixed attributes.
    (Type: String) target_attribute:---> label attribute name or name of dependent variable.
    (Type: Boolean) include_target_attribute:---> If 'yes', then it takes target variable also in consideration while saggregating.
    
    Return:
    (Type: List(), List()) List with categorical attribes' names, List with continuous attributes names.
    
    '''

    # Creating a copy of original data to maintain its intigrity.   
    data= dataset.copy()
    
    # Taking columns name 
    cols= data.columns
    
    # This list will have categorical attribes' names
    Nominal_cols=[]
    
    for i in cols:

        # Calculating mean after casting entries to float.
        data[i]=data[i].astype(object)
        
        try:
            
            # If true the dependent variable or target variable will also be clossified as catogorical or numerical
            if include_target_attribute:

                if i != target_attribute:

                    # Check condition for the categorical feature. Runs only for numerical entry.
                    # (Exceptional handling anabled in case data[i][0] does not have isalpha func)
                    if data[i][0].isalpha():

                        Nominal_cols.append(i)

            # If target variable is excempted.
            else:

                # Check condition for the categorical feature. Runs only for numerical entry.
                # (Exceptional handling anabled in case data[i][0] does not have isalpha func)
                if data[i][0].isalpha():

                    Nominal_cols.append(i)
        except:
            
            print("[X]. Raw data file is not well defiend. Reliability may get compromised")        
    
    templist=Nominal_cols.copy()

    # Now templist will have the categorical column names and name of target variable
    templist.append(target_attribute)
    
    # The scond return value is the name of the columns with numerical entries.
    return Nominal_cols, Diff(list(data.columns.values),templist)


# Function to change the type of cloumns of the dataset
def ChangeType(dataset, cols):
    
    '''Changes the type of columns in float64 dtype
    
    Parameters:
    (Type: pandas.DataFrame) dataset:---> Dataset where change of type is required.
    (Type: List()) cols:--->  list of the column names where type change is required.
    
    Return:
    (Type: pandas.DataFrame) data:---> Dataset with changed types.
    
    '''
    
    # Creating a copy of original data to maintain its intigrity.
    data= dataset.copy()  
    
    for i in cols:
        
        # Changing type of each column into float64
        data[i]=data[i].astype(float)
    
    return data


# Function to standardize the independent variables
def standardization(X, record_param= True):
    
    '''Performs the standardization on attributes in inplace manner
    
    Performs standardization and returns the dictionary with elements as tuples(mean, std. deviation)
    for each attributes.
    
    Parameters:
    (Type: pandas.DataFrame) X:---> Dataset(excluding target).
    (Type: Boolean) record_param:---> Indicates whether to return parameter dictionary.
    
    Return:
    (Type: dict={ Name of attribute : tuple(mean, std. deviation)) data:---> Disctionary of parameters of standardization.
    
    '''
    # Getting column names
    Columns=X.columns

    # This dictionary will store the paramenters used in standardization process
    Standardization_parameters={}
    
    for i in Columns:
        
        p= np.array(X[i])
        mean_of_data= np.mean(p)
        std_devi_of_data= np.std(p)

        # performing standardization column wise
        X[i]= [(x - mean_of_data)/std_devi_of_data for x in p]
        
        # Removing columns(features) which have no deviation throughout.
        if std_devi_of_data == 0:
            
            X.drop(i,axis=1,inplace=True)
            
        else:
            
            # Writting on dictionary, the params for column
            Standardization_parameters[i]= (mean_of_data,std_devi_of_data)

    # Only returns the Standardization_parameters if record_param is true.        
    if record_param:
        
        return Standardization_parameters
    
    
##    
## Rest code is specific to thyroid dataset    
##
    
# Loading dataset. 
data= pd.read_csv("Data/allbp.data")

# Giving the column names
data.columns= ["age","sex","on_thyroxine","query_on_thyroxine","on_antithyroid_medication","sick","pregnant","thyroid_surgery","I131_treatment", "query_hypothyroid","query_hyperthyroid","lithium","goitre","tumor","hypopituitary","psych","TSH_measured","TSH","T3_measured","T3","TT4_measured","TT4","T4U_measured","T4U","FTI_measured","FTI","TBG_measured","TBG","Referral_source", "Class"]

# Parsing the class attribute to get class
data["Class"]= [x.split(".")[0] for x in data["Class"]]

# Applying columns filter
data= RelevanceColumnFilter(data, key="?", filter_threshold=0.9)

# Dividing the categorical and contimuous parameters
colNom, NumcolNom= GetAttributesInCategory(data, "Class", True)

# Handling the missing values (?)
data= ReplaceMeanAndMode(data, key='?')

# Applying dummy encoding for the categorical attributes
data= pd.get_dummies(data,columns=colNom)

# Changing the types to float64
data= ChangeType(data,NumcolNom)

# Taking target variable out
Y= data["Class"]

# Taking the input parameters 
X= data.drop("Class",axis=1)

# Performing standardization along with feature selection and taking parameters of standarization out
std_param_dics = standardization(X, record_param= True)
standardX= X.copy()

# Extending with the target variable
standardX['label']= Y

# Writting standardization parameter file
file_handle = open("Model/Standardization_params_for_Thyroid_diseases.dict", "w+")
data = str(std_param_dics)
file_handle.write(data)
file_handle.close()

# Writing the excel file with the new dataset, ready for modelling set
standardX.to_excel(excel_writer="DataforModels/Thyroid.xlsx",index=False)

################################# End of File ###################################
