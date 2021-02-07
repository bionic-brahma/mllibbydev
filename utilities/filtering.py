import numpy as np
from itertools import count
from operator import methodcaller


def format_dataframes(df):
    """ format pandas dataframe

    Parameters:
    (Type: pandas dataframe) df ----> panda dataframe to be formatted

    Return:

    (Type: pandas dtaframe) formatted dataframe

    """

    # dropping 1st column
    df.drop(df.columns[0], axis=1, inplace=True)

    # changing header with first row
    new_header = df.iloc[0]
    if 'x1' in ",".join(new_header):
        # df = df[1:]
        df.columns = [str(c).lower() for c in new_header]

        # removing first 4 or 5 rows
        del_rows = []
        for i in count():
            check_row = list(map(methodcaller('lower'), map(str, df.iloc[i])))
            if any(x == check_row[0][0] for x in ['0', '1', 'y', 'n', 'm', 'f']):
                break
            del_rows.append(i)

        df.drop(del_rows, axis=0, inplace=True)

    else:
        col_len = len(df.columns)
        headings = []
        for i in range(1, col_len):
            headings.append(f'x{i}')
        headings.append('label')
        df.columns = headings

    df = df.reset_index()
    df.drop('index', axis=1, inplace=True)
    return df


def filter_df_data(df):
    """ filter pandas dataframe for erroneous values

    Parameters:
    (Type: pandas dataframe) df ----> panda dataframe to be filtered

    Return:

    (Type: pandas dataframe) formatted dataframe

    """
    try:
        df['x1']
    except KeyError:
        df = format_dataframes(df)

    for key in df.keys():
        length = len(df[key])

        for i in range(length):
            if str(df[key][i]).lower().strip().startswith('m'):
                df.loc[i, key] = np.int64(1)
            elif str(df[key][i]).lower().strip().startswith('b'):
                df.loc[i, key] = np.int64(0)
    return df
