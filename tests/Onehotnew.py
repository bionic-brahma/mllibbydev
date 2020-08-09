

import pandas as pd
from itertools import chain
class one_hot():
    def __init__(self, df):
        
        self.df = df
        self.is_categorical = {}
        self._who_hot(df)
    
    def cat_var(self, df): 
        """Identify categorical features. 

        Parameters
        ----------
        df: original df after missing operations 

        Returns
        -------
        cat_var_df: summary df with col index and col name for all categorical vars
        """
        col_type = df.dtypes
        col_names = list(df)
        cat_var_index = [i for i, x in enumerate(col_type) if x=='object']
        cat_var_name = [x for i, x in enumerate(col_names) if i in cat_var_index]
        cat_var_df = pd.DataFrame({'cat_ind': cat_var_index, 
                                   'cat_name': cat_var_name})
        return cat_var_df, cat_var_name

    def one_hot_dummy(self, df, cols):
        cat_var_df, cat_var_name=self.cat_var(df)
#         print(cat_var_df)
        
        label_col = df.pop('Label')
        for cat_name in cols:
#             print(cat_name)
            arr=df[cat_name].unique()
            lenarr=arr.size
            
            #aa is the dictionary where I saved the changed values
            if(lenarr==2):
                df=df.replace(arr[0],1)
                df=df.replace(arr[1],0)
                aa = {x:i for i,x in enumerate(arr)}
#                 print(aa)
                dummies = pd.get_dummies(df[cat_name], drop_first=True)
                df = df.join(dummies)
            else:
                aa = {x:i for i,x in enumerate(arr)}
#                 print(aa)
                dummies = pd.get_dummies(df[cat_name], drop_first=True)
                df = df.join(dummies)
        df=df.drop(columns=cols)
        length=len(df.columns)
        df.insert(length, 'Label', label_col)
        return df
    
    def _who_hot(self, df):
        
        from pandas.api.types import is_string_dtype
        
        def _get_unique_values(series):
            return list(series.unique())
        
        columns_list = list(df)
        self.columns_list = columns_list
        for col in columns_list:
            
            # IF not a string, it is not categorical
            is_string = is_string_dtype(df[col])
            if (not is_string):
                self.is_categorical[col] = False
                continue
            
            unique_vals = _get_unique_values(df[col])
            num_unique_vals = len(unique_vals)
            
            assert num_unique_vals >=2, f'column {col} has strings but only {num_unique_vals} unique values'
            
            self.is_categorical[col] = unique_vals

    def one_hot_list(self, row):
        
        arr = []
        for col in self.columns_list:
            
            is_cat = self.is_categorical[col]
            val = row[col]

            if (not is_cat):
                arr.append(val)
            
            else:
                idx = is_cat.index(val)
                if (len(is_cat) == 2):
                    arr.append(idx)
                
                else:
                    tmp_arr = [0] * len(is_cat)
                    tmp_arr[idx] = 1
                    arr += tmp_arr
            
        return arr
            
        


df = pd.read_excel("onehot.xlsx", index_col=0)


one =one_hot(df)
print(one.is_categorical)
print(one.columns_list)


cat_var_df,cat_var_name=one.cat_var(df)
one.one_hot_dummy(df,cat_var_name)
