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


    def categorica_var(self):
        """Identify categorical features. 

        Parameters
        ----------
        df: original df after missing operations 

        Returns
        -------
        cat_dict: summary df with col index as key which needs to be change
        """
        col_type = self.df.dtypes
        col_name = list(self.df)
        cat_var_index = [i for i, x in enumerate(col_type) if x == 'object']

        for i in cat_var_index:

            unique_col = pd.unique(self.df[self.df.columns[i]])

            if len(unique_col) == 2:
                dict_unique_2 = {}

                for j, k in enumerate(unique_col):
                    dict_unique_2[k] = j
                self.is_categorical[i] = dict_unique_2

            elif len(unique_col) == 1:
                self.is_categorical[i] = 1

            # else:



        return cat_var_index



        


df = pd.read_excel("onehot.xlsx")
a = data_filter(df)
a_1= a.categorica_var()
print(a.df.columns[a_1[0]])
# print(list(pd.get_dummies(a.df[a.df.columns[a_1[1]]])))
# print(pd.get_dummies(a1))
# x = re.findall(r"(?i)unnamed" or r"(?i)sr" or r"(?i)index", df.columns[0])
# print(len(x))
# print(len(x)>0)

# df.reset_index(drop=True, inplace=True)
# print(df)