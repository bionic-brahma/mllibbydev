import xlrd
import pandas as pd
from sklearn import preprocessing
import numpy as np

from itertools import count
from operator import methodcaller
def format_dataframes(df):
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
            if any(x == check_row[0][0] for x in ['0', '1', 'y', 'n']):
                break
            del_rows.append(i)

        df.drop(del_rows, axis=0, inplace=True)

        # del_rows = [0,1,2,3]
        # fourth_row = map(methodcaller('lower'), map(str, df.iloc[4]))

        # if not any(x in fourth_row for x in ['0', '1', 'y', 'n']):
        #     del_rows.append(4)
        # df.drop(del_rows, axis=0, inplace=True)

        # resetting index and deleting old index col
        # df = df.reset_index()
        # df.drop('index', axis=1, inplace=True)
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

# read data from excel file

def filter_df_data(df):
    try:
        df['x1']
    except KeyError:
        df = format_dataframes(df)

    for key in df.keys():
        length = len(df[key])

        for i in range(length):
            if str(df[key][i]).lower().strip().startswith('y'):
                df.loc[i, key] = np.int64(1)
            elif str(df[key][i]).lower().strip().startswith('n'):
                df.loc[i, key] = np.int64(0)
    return df

# read data from excel file
#data_df = pd.read_excel(r'dummy_data.xlsx')
data_df = pd.read_excel(r'dummy_data.xlsx')
data_df = filter_df_data(data_df)
data_df = data_df.dropna()

predict = 'label'    # can be changed based on the column to be predicted

x = np.array(data_df.drop([predict], 1))
y = np.array(data_df[predict])

n_sample, n_features = np.shape(x)
n_labels = np.size(np.unique(y))

# Normalizing the feature array
def normal_norm(arr):
    normal_arr = (arr - arr.mean(axis=0))/arr.std(axis=0)
    return normal_arr

# utility functions

def euclideanDistance(a,b):
    dist = np.linalg.norm(a-b)
    return dist

def Srt(distances):
    distances.sort()
    return distances

# functions  to get distances with all the data points from the test data point

def getNeighbours(testdata,x, n_sample):
    similarity = []
    for i in range(0,n_sample):
        similarity.append([euclideanDistance(x[i],testdata),y[i]])
    return Srt(similarity)

#kNN predict to find best label
def kNN_predict(testdata,k,x,n_train,n_labels):
    similarity = getNeighbours(testdata,x,n_train)
    out = list(np.zeros(n_labels))
    for i in range(0,k):
        out[similarity[i][1]]+=1
    mx_i=0
    for i in range(0,n_labels):
        if out[mx_i]<out[i]:
            mx_i=i
    return mx_i

x_norm = normal_norm(x)


# Runnig on Training Data

n_train = int(0.8*n_sample)

k = int(pow(n_train,0.5)) if int(pow(n_train,0.5)) % 2 !=0 else int(pow(n_train,0.5)) + 1
output=np.array([])

for i in range(0,n_train):
    pred = kNN_predict(x_norm[i],k,x_norm,n_train,n_labels)
    output = np.append(output,pred)

acc = np.sum(output == y[:n_train])/n_train


print('Train Data Accuracy : '+ str(100*acc))

# Running on Test Data

n_test = n_sample-n_train
output=np.array([])

for i in range(n_train,n_sample):
    pred = kNN_predict(x_norm[i],k,x_norm,n_train,n_labels)
    output = np.append(output,pred)

acc = np.sum(output == y[n_train:])/n_test

print('Test Data Accuracy : '+ str(100*acc))

# forecast fucntion

def forecast(x, arr):
    x = (x - arr.mean(axis=0))/arr.std(axis=0)
    return kNN_predict(x,k,x_norm,n_train,n_labels)

X = [4 ,2 ,1 ,1 ,2]
forecast(X,x)
