import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import data_formating as daf
import copy

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


def remove_less_significant(data, Y):
    """ Remove less signigficent features

    Parameter
    ------------------
    X: dataframe object without label
    Y: label dataframe

    Returns
    ------------------
    column name which need to be droped"""
    if isinstance(data, pd.DataFrame): # Condition for testing the type
        X = copy.copy(data)            # Using copy so the actual data is not changed
        significance_level = 0.05      # Threshold for p-value is selcetion as 5%
        OrdinaryLeastSquare = None
        column_drop = np.array([])

        # number of column time the loop
        for i in range(0, len(X.columns)):
            # Using OLS model
            OrdinaryLeastSquare = sm.OLS(Y,X).fit()
            max_col = OrdinaryLeastSquare.pvalues.idxmax()
            max_val = OrdinaryLeastSquare.pvalues.max()

            # condition to extrect the usable column
            if max_val > significance_level:
                X.drop(max_col, axis='columns', inplace=True)
                column_drop = np.append(column_drop, [max_col])
            else:
                break
        # print(OrdinaryLeastSquare.summary())
        return list(column_drop)









if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    test_file = os.path.join(BASE_DIR, "data_test", "kNN_data.xlsx")
    df = pd.read_excel(test_file)
    df = daf.DataAdjust(df)
    print(df.df.head())
    df.categorica_data_encoding()
    # print(df.df.head())
    Y = df.df[['label']]
    df.df.drop('label', axis = 'columns', inplace=True)
    print(list(remove_less_significant(df.df, Y)))
    print(df.df.head())
    print(list(remove_correlated(df.df)))
    