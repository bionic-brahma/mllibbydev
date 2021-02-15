####################################################################
# Add only tested functions with all possible corner cases
# Maintained and collaborated by Devendra Kumar for Risk Latte AI Inc.
####################################################################

import pandas as pd
import numpy as np


def covNcorr(mat):
    """
        This function is to calculate the covariance and the correlation matrix of the given matrix 'mat'
        Calculation has been done in vectorised form using numpy module of python

        Parameter:
        (Type: numpy.array) mat:---> Data matrix is required in the columns as attributes(features) and rows as records

        Return:
        (Type: numpy.array) covmat:---> Covariance matrix of data matrix 'mat'
        (Type: numpy.array) corrmat:---> Correlation matrix of the data mtrix 'mat'

    """

    # Converting matrix to float type
    mat = mat.astype(np.float)

    # Taking shape of matrix
    r, c = mat.shape

    # Making array to contain Std. Deviation of columns of the matrix
    colstd = np.zeros(c).astype(np.float)

    # Making data as mean centralised
    for i in range(c):
        try:
            mat.iloc[:, i] = (mat.iloc[:, i] - np.mean(mat.iloc[:, i]))
        except:
            mat[:, i] = (mat[:, i] - np.mean(mat[:, i]))

    # Finding the Std. Deviation of each column in vectorized operation
    for i in range(c):
        # This operation is only valid if the matrix is mean centralised.
        # Its valid in this case as the mean centralisation has been done in previous step.
        try:
            colstd[i] = 1 / (r - 1) * np.sqrt(np.dot(np.transpose(mat.iloc[:, i]), mat.iloc[:, i]))
        except:
            colstd[i] = 1 / (r - 1) * np.sqrt(np.dot(np.transpose(mat[:, i]), mat[:, i]))
    # Containers/Variables to hold covariance and correlation of mat
    covmat = np.zeros([c, c]).astype(np.float)
    corrmat = np.zeros([c, c]).astype(np.float)

    # Calculating covariance and correlation of data matrix 'mat'
    for i in range(c):

        for j in range(c):
            # Covariance
            try:
                covmat[i][j] = 1 / (r - 1) * np.dot(np.transpose(mat.iloc[:, i]), mat.iloc[:, j])
            except:
                covmat[i][j] = 1 / (r - 1) * np.dot(np.transpose(mat[:, i]), mat[:, j])
            # Correlation
            try:
                corrmat[i][j] = 1 / (r - 1) * np.dot(np.transpose(mat.iloc[:, i]), mat.iloc[:, j]) / (colstd[i] * colstd[j])
            except:
                corrmat[i][j] = 1 / (r - 1) * np.dot(np.transpose(mat[:, i]), mat[:, j]) / (colstd[i] * colstd[j] +1)
    return covmat, corrmat


def PCA(mat, ip=90, pca_vectors=0, print_msg=False):
    """
        This function is to find out the principal components of the given data.

        Parameters: (Type: numpy.array) mat:---> Data matrix is required in the columns as attributes(features) and
        rows as records (without output labels) (Type: float) ip:---> ip is the threshold of the information that is
        should be preserved at minimum (Information Preserved). If value of pca_vectors variable is non zero. then
        the ip value will get override. (Type: int) pca_vectors:---> Number of vectors to have in the transformation
        basis. If it is 0 then the vectors will be in such manner that the information preserved is atleast till ip
        level


        Return: (Type: numpy.array) ready_transform_basis:---> This is the transformation matrix. it can be used to
        calculate the projection of the data matrix. It consists of most significant vectors as column entry. (Type:
        numpy.array) projected_data:---> This contains the projected matrix. ie. dot product of mat and
        ready_transform_basis (Type: float) information_retained:---> This gives the floating point number
        indicating the information retained by the Ready_transform_basis

    """

    # function to calculate the covariance and the correlation of the matrix
    covmat, corrmat = covNcorr(mat)

    # finding the eigenvalues and eigenvectors of covariance matrix
    e, v = np.linalg.eig(covmat)

    # sorting using dataframes
    df = pd.DataFrame([e, v]).transpose()
    df.columns = (["EigenValue", "EigenVector"])
    dfsorted = df.sort_values(by="EigenValue", ascending=False)
    eigenvalues_sorted = np.array(dfsorted["EigenValue"])
    eigenvectors_sorted = np.array(dfsorted["EigenVector"])

    # calculating the information preserved and the principal components
    sum_eigenvalues = np.sum(eigenvalues_sorted)
    eigen_temp_sum = 0.0
    ip_calc = 0.0
    transform_matrix = np.transpose(eigenvectors_sorted[0]).copy()
    i = 0
    flag = 1
    while flag:
        eigen_temp_sum = eigen_temp_sum + eigenvalues_sorted[i]
        ip_calc = eigen_temp_sum / (sum_eigenvalues * 100)
        if print_msg:
            print("Adding Component No.", i + 1)
        if i != 0:
            transform_matrix = np.append(transform_matrix, np.transpose(eigenvectors_sorted[i]))
        i = i + 1
        if print_msg:
            print("Information Retained = ", ip_calc)
        if pca_vectors == 0:
            if ip <= ip_calc or ip_calc == 100:
                flag = 0
        else:
            if i >= pca_vectors or ip_calc == 100:
                flag = 0
    ready_transform_basis = np.transpose(transform_matrix.reshape(i, eigenvectors_sorted[0].shape[0])).copy()
    if print_msg:
        print("+++++++++++++++++++++++++++++++Transform_basis+++++++++++++++++++++++++")
        print(ready_transform_basis)
        print("Projected Data:")

    # calculating the projection matrix
    projected_data = np.dot(mat, ready_transform_basis)
    if print_msg:
        print(projected_data)

    # information retained
    information_retained = ip_calc

    return ready_transform_basis, projected_data, information_retained


def remove_correlated(input_x, threshold_corr=0.95):
    """ Removes coorelated features
    
    input_x should be all the numerical columns or else
    only numericals columns will be selected and that can lead to error

    Parameter:
   
    (Type: pd.DataFrame) input_x:---> dataframe object without label
    (Type: Float) threshold_corr:---> Columns with more than threshold will be removed.

    Returns
   
    column name which need to be dropped"""

    # Condition for testing the type
    if isinstance(input_x, pd.DataFrame):

        # Correlation matrix
        corr_matrix = input_x.corr()

        # This will contain the columns to be removed      
        drop_column = np.full(corr_matrix.shape[0], False, dtype=bool)
        print(drop_column.shape)

        # Loop over all the element of the correlation matrix
        for i in range(corr_matrix.shape[0]):

            for j in range(i, corr_matrix.shape[0]):

                if i == j:

                    pass

                else:

                    if corr_matrix.iloc[i, j] >= threshold_corr:
                        drop_column[j] = True

        # Checks if the shape is right or not
        if input_x.shape[1] != drop_column.shape[0]:

            print("[input_x]. Function expects the input_x to be all numerical columns.")

            return 0

        else:

            column_drop = input_x.columns[drop_column]
            input_x.drop(column_drop, axis=1, inplace=True)

        return column_drop
