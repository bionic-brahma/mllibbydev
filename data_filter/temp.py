import pandas as pd
import re

class Data():
    def __init__(self, df):
        """ Identify the index column

        creating a temp variable to store the index column

        temp
        -------------
        if there is an index the len will be more than 0
        if the index column exist will be considered while importing

        """
        temp = re.findall(r"(?i)un" or r"(?i)sr" or r"(?i)ind", df.columns[0])
        
        if len(temp) > 0:
            self.df = df.drop(df.columns[0], axis=1)
        else:
            self.df = df

        self.is_categorical = {}        # Storing all the categorical value
        self.column_name = df.columns   # list conaining column name


    def categorica_data_encoding(self, NumberLabel = None):
        """Identify categorical features. 

        Parameters
        ----------
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

    def FormatList(Self, ListtoConvert):
        """ Converting the list based on the data  

        Parameters
        ----------
        ListtoConvert : list to be converted based on the 

        Returns
        -------
        modified df and the dict for the cateregorical data
        """
        pass

        



        
if __name__ == "__main__":

    df = pd.read_excel("onehot.xlsx")
    # print(count())
    # a = Data(df)
    # a.categorica_data_encoding()
    # print(a.df)
    # print(a.is_categorical)
    # print(df.columns[-2:])
