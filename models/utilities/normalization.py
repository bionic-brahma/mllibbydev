# Normalizing the feature array
def normal_norm(df):
    """ normalization function to normalize data

    Parameters:
    (Type: pandas dataframe) df ----> panda dataframe to be normalized

    Return:

    (Type: pandas dataframe)

    """
    normal_arr = (df - df.mean(axis=0)) / df.std(axis=0)
    return normal_arr


# n_feature are used for files  which have file row  X1,X2,X3,x4 .... Xn, label
def normalize_norm(df, n_features):  # normalisation
    """ normalization function to normalize data having defined columns as x1,x2,x3... ,xn, label

    Parameters:
    (Type: pandas dataframe) df ----> panda dataframe to be normalized
    (Type: integrer)  n_features ----> no of features spaces in the dataframe

    Return:

    (Type: pandas dataframe)

    """
    temp_df = df.copy()
    for i in range(0, n_features):
        xmean = df['x' + str(1 + i)].mean()
        xstd = df['x' + str(1 + i)].std()
        temp_df['x' + str(1 + i)] = (df['x' + str(1 + i)] - xmean) / xstd
    return temp_df


def normalize_min_max(df, n_features):  # standardization
    """ normalization function min max normalize dataframe having defined columns as x1,x2,x3... ,xn,

    Parameters:
    (Type: pandas dataframe) df ----> panda dataframe to be normalized
    (Type: integrer)  n_features ----> no of features spaces in the dataframe

    Return:
    (Type: pandas dataframe)

    """

    temp_df = df.copy()
    for i in range(0, n_features):
        xmin = df['x' + str(1 + i)].min()
        xmax = df['x' + str(1 + i)].max()
        temp_df['x' + str(1 + i)] = (df['x' + str(i)] - xmin) / (xmax - xmin)
    return temp_df


def normalize_mean(df, n_features):  # mean centralisation/ data centralizarion
    """ normalization function to perform mean normalization dataframe having defined columns as x1,x2,x3... ,xn

    Parameters:
    (Type: pandas dataframe) df ----> panda dataframe to be normalized
    (Type: integrer)  n_features ----> no of features spaces in the dataframe

    Return:

    (Type: pandas dtaframe)

    """
    temp_df = df.copy()
    for i in range(0, n_features):
        xmean = df['x' + str(1 + i)].mean()
        temp_df['x' + str(1 + i)] = (df['x' + str(1 + i)] - xmean)
    return temp_df
