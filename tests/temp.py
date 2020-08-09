import pandas as pd
import re

class test_train_split():
    def __init__(self, df):

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
        cat_var_index = [i for i, x in enumerate(col_type) if x == 'object']

        return cat_var_index



        


df = pd.read_excel("onehot.xlsx")
a = test_train_split(df)
print(a.categorica_var())
# x = re.findall(r"(?i)unnamed" or r"(?i)sr" or r"(?i)index", df.columns[0])
# print(len(x))
# print(len(x)>0)

# df.reset_index(drop=True, inplace=True)
# print(df)