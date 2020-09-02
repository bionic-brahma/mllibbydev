import pandas as pd
import re

class data_filter():
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

        self.is_categorical = {}


    def categorica_data_encoding(self, label_no = None):
        """Identify categorical features. 

        Parameters
        ----------
        df: original df after missing operations 
        label_no: count of the atribute from the end of data to be treated as label

        Returns
        -------
        create modified df and the dict for the cateregorical data
        """
        col_type = self.df.dtypes
        cat_var_index = [i for i, x in enumerate(col_type) if x == 'object']
        col_names = list(self.df)

        if label_no is None:
            y_temp = self.df[self.df.columns[-1]]
            x_temp = self.df.drop(self.df.columns[-1], axis=1)
        elif isinstance(label_no, int):
            y_temp = self.df[self.df.columns[-label_no:]]
            x_temp = self.df.drop(self.df.columns[-label_no:], axis=1)


        # Loop over all the categorical column
        for i in cat_var_index:

            # Converting all the categorical value to lower case 
            self.df[col_names[i]] = self.df[col_names[i]].str.lower()
            

            # Counting the unique value of the categorical value
            unique_col = pd.unique(self.df[col_names[i]])


            # Condition if the unique value is 2 and replacing it with numerical
            if len(unique_col) == 2:
                dict_unique_2 = {}

                # Loop for replacing each categorical value
                for j, k in enumerate(unique_col):
                    dict_unique_2[k] = j

                self.is_categorical[i] = dict_unique_2
                
                self.df[col_names[i]].replace(dict_unique_2, inplace=True)

            # Condition if the unique value is >2 and using one hot encoding
            elif len(unique_col) > 2:

                dummies_array = pd.get_dummies(self.df[col_names[i]])

                self.is_categorical[i] = list(dummies_array)

                self.df = self.df.drop([col_names[i]], axis = 1)

                self.df = self.df.join(dummies_array)

    # def format_dataframes(Self):
        



        


df = pd.read_excel("onehot.xlsx")
# print(count())
# a = data_filter(df)
# a.categorica_data_encoding()
# print(a.df)
# print(a.is_categorical)
# print(X.head())
# print(y.head())