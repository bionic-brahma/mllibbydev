import numpy as np
import pandas as pd
import os
import data_formating as daf

def remove_correlated(X):
    """ Remove coorelated features

    Parameter
    ------------------
    X: dataframe object without label

    Returns
    ------------------
    column name which need to be droped"""

    if isinstance(X, pd.DataFrame): # Condition for testing the type
    
        threshold_corr = 0.9        # Thresold for correlation only lower value column will be consider
        corr_matrix = X.corr()      # Correlation matrix
        drop_column = np.full(corr_matrix.shape[0], False, dtype=bool)

        # Loop over all the element of the correlation matrix
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[0]):
                if corr_matrix.iloc[i,j] >= threshold_corr:
                    drop_column[j] = True

        column_drop = X.columns[drop_column]
        X.drop(column_drop, axis=1, inplace=True)
        return column_drop


def remove_less_significant(X, Y):
     """ Remove less signigficent features

    Parameter
    ------------------
    X: dataframe object without label
    Y: label dataframe

    Returns
    ------------------
    column name which need to be droped"""

    significance_level = 0.05
    regression_ols = None
    column_drop = np.array([])
    for i in range(0, len(X.columns)):
        pass
    pass










if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    test_file = os.path.join(BASE_DIR, "data_test", "onehot.xlsx")
    df = pd.read_excel(test_file)
    df = daf.DataAdjust(df)
    print(df.df.head())
    df.categorica_data_encoding()
    print(df.df.head())
    print(remove_correlated(df.df))
    