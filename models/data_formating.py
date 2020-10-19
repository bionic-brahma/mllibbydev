import pandas as pd
import re
import copy

class DataAdjust():
    def __init__(self, df):
        """ Identify the index column

        creating a temp variable to store the index column

        temp
        -------------
        if there is an index the len will be more than 0
        if the index column exist will be considered while importing

        """
        temp = re.findall(r"(?i)unnamed" or r"(?i)sr.no" or r"(?i)index" or r"(?i)s.no", df.columns[0])
        
        if len(temp) > 0:
            self.df = df.drop(df.columns[0], axis=1)
        else:
            self.df = df

        self.is_categorical = {}        # Storing all the categorical value
        self.column_name = df.columns   # list conaining column name
        # self.column_name_index = _rearrangeLabele()


    def categorica_data_encoding(self, NumberLabel = None):
        """Identify categorical features. 

        Parameters
        ----------
        NumberLabel is the count of lable from the end of the table
        df: original df after missing operations 
        NumberLabel: count of the atribute from the end of data to be treated as label

        Returns
        -------
        modified df and the dict for the cateregorical data
        """
        col_type = self.df.dtypes
        cat_var_index = [i for i, x in enumerate(col_type) if x == 'object']
        col_names = list(self.df)

        if NumberLabel is None:
            y_temp = self.df[self.df.columns[-1]].to_frame()
            x_temp = self.df.drop(self.df.columns[-1], axis=1)
        elif isinstance(NumberLabel, int):
            y_temp = self.df[self.df.columns[-NumberLabel:]]
            x_temp = self.df.drop(self.df.columns[-NumberLabel:], axis=1)


        # Loop over all the categorical column
        for i in cat_var_index:

            try:
                # Converting all the categorical value to lower case     
                x_temp[col_names[i]] = x_temp[col_names[i]].str.lower()

                # Counting the unique value of the categorical value
                unique_col = pd.unique(x_temp[col_names[i]])


                # Condition if the unique value is 2 and replacing it with numerical
                if len(unique_col) == 2:
                    dict_unique_2 = {}

                    # Loop for replacing each categorical value
                    for j, k in enumerate(unique_col):
                        dict_unique_2[k] = j

                    self.is_categorical[i] = dict_unique_2
                    
                    x_temp[col_names[i]].replace(dict_unique_2, inplace=True)

                # Condition if the unique value is >2 and using one hot encoding
                elif len(unique_col) > 2:

                    dummies_array = pd.get_dummies(x_temp[col_names[i]])

                    self.is_categorical[i] = list(dummies_array)

                    x_temp = x_temp.drop([col_names[i]], axis = 1)

                    x_temp = x_temp.join(dummies_array)

            except:
                
                # Converting all the categorical value to lower case 
                y_temp[col_names[i]] = y_temp[col_names[i]].str.lower()
                
                # Counting the unique value of the categorical value
                unique_col = pd.unique(y_temp[col_names[i]])

                # Condition if the unique value is 2 and replacing it with numerical
                if len(unique_col) == 2:
                    dict_unique_2 = {}

                    # Loop for replacing each categorical value
                    for j, k in enumerate(unique_col):
                        dict_unique_2[k] = j

                    self.is_categorical[i] = dict_unique_2
                    
                    y_temp[col_names[i]].replace(dict_unique_2, inplace=True)

                # Condition if the unique value is >2 and using one hot encoding
                elif len(unique_col) > 2:

                    dummies_array = pd.get_dummies(y_temp[col_names[i]])

                    self.is_categorical[i] = list(dummies_array)

                    y_temp = y_temp.drop([col_names[i]], axis = 1)

                    y_temp = y_temp.join(dummies_array)

            self.df = x_temp.join(y_temp)

    '''def _rearrangeLabele(self, ColumnToLabel = None):

        """ This function is for rearanging the data
        suppose a user want the 3 column to be a label
        so the 3rd will be shifted to last as column

        parameters
        -----------
        ColumnToLabel : name or location of the column to be treated as label

        Returns
        -------
        ColumnToLabeldict : dictonary containing index and name of rearrange column
        """
        _templist = self.df.columns
        _tempvar = { i : _templist[i] for i in range(len(_templist))}

        if ColumnToLabel == None:
            ColumnToLabeldict = _tempvar
            return ColumnToLabeldict

        else:
            if isinstance(ColumnToLabel, int):
                _templist.append(_templist.pop(ColumnToLabel-1))
                ColumnToLabeldict = { i : _templist[i] for i in range(len(_templist))}
                self.df = self.df[_templist]
                return ColumnToLabeldict

            elif isinstance(ColumnToLabel, (st r, object)):
                if not ColumnToLabel in _tempvar.values():
                    AssertionError "The Column name or the location is incorrect, 'this field is case sensitive'"
                _templist.append(_templist.pop(_templist.index(ColumnToLabel)))
                ColumnToLabeldict = { i : _templist[i] for i in range(len(_templist))}
                self.df = self.df[_templist]
                return ColumnToLabeldict
                '''
         

    def FormatList(self, ListtoConvert):
        """ Converting the list based on the data  

        Parameters
        ----------
        ListtoConvert : list or numpy.ndarray to be converted based on the 

        Returns
        -------                       
        modified list or numpy.ndarray
        """
        keys_list = list(self.is_categorical.keys())
        temp = ListtoConvert.copy()
        pop_index = []

        if isinstance(ListtoConvert, list): # Checking if passed input is list

            for i,j in enumerate(temp):
                if i in keys_list:          # Condition if the entry is avaliable in the conversion dict
                    if  isinstance(self.is_categorical[i], list):   # Condition for if the value is no binary
                        pop_index.append(i)
                        for k in self.is_categorical[i]:
                            if j.lower() == k:
                                ListtoConvert.append(1)
                            else:
                                ListtoConvert.append(0)
                    
                    # For the binay category like male, female entries
                    if isinstance(self.is_categorical[i], dict):    
                        ListtoConvert[i] = self.is_categorical[i][j.lower()]

            # Created this loop to pop the converted entry and 't' is used as each pop shift the list index
            t = 0
            for i in pop_index:
                ListtoConvert.pop(i - t)
                t += 1

        return ListtoConvert

        



        
if __name__ == "__main__":
    pass
