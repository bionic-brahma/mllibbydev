#######################################################################################
## The class contains the methods related to data compression and dimension reduction.#
##                                                                                    #
## Work by Devendra Kumar for Risk Latte AI Inc.                                      #
#######################################################################################

# begining of imports                                               ###
                                                                    ###
import numpy as np                                                  ###
import pandas as pd                                                 ###
                                                                    ###
# End of imports                                                    ###

#######################################################################################
#######################################################################################

class DataCompression():
    '''
    if print_msg is false. Nothing will be printed in console
    '''

    def __init__(self, print_msg = False):
        self.print_msg = print_msg
        

# Begining of the covNcorr() function

    def covNcorr(self, Mat):
        '''
        This function is to calculate the covariance and the correlation matrix of the given matrix 'Mat'
        Calculation has been done in vectorised form using numpy module of python

        Parameter:
        (Type: numpy.array) Mat:---> Data matrix is required in the columns as attributes(features) and rows as records
        
        Return:
        (Type: numpy.array) CovMat:---> Covariance matrix of data matrix 'Mat'
        (Type: numpy.array) CorrMat:---> Correlation matrix of the data mtrix 'Mat'

        '''

        # Converting Matrix to float type
        Mat= Mat.astype(np.float)

        # Taking shape of matrix
        r,c= Mat.shape

        # Making array to contain Std. Deviation of columns of the matrix
        Colstd= np.zeros(c).astype(np.float)

        # Making data as mean centralised
        for i in range(c):

            Mat[:,i]= Mat[:,i]- np.mean(Mat[:,i])

        # Finding the Std. Deviation of each column in vectorized operation  
        for i in range(c):

            # This operation is only valid if the matrix is mean centralised.
            # Its valid in this case as the mean centralisation has been done in previous step.  
            Colstd[i]= 1/(r-1) * np.sqrt(np.dot(np.transpose(Mat[:,i]),Mat[:,i]))

        # Containers/Variables to hold covariance and correlation of Mat
        CovMat=np.zeros([c,c]).astype(np.float)
        CorrMat=np.zeros([c,c]).astype(np.float)

        # Calculating covariance and correlation of data matrix 'Mat'
        for i in range(c):
            
            for j in range(c):

                # Covariance
                CovMat[i][j]= 1/(r-1) * np.dot(np.transpose(Mat[:,i]),Mat[:,j])

                # Correlation
                CorrMat[i][j]= 1/(r-1) * np.dot(np.transpose(Mat[:,i]),Mat[:,j]) / (Colstd[i]*Colstd[j])
        
        return CovMat, CorrMat

    # End of the covNcorr() function




    # Begining of PCA() function

    def PCA(self, Mat, IP=90, PCA_vectors=0):
        '''
        This function is to find out the principal components of the given data.

        Parameters:
        (Type: numpy.array) Mat:---> Data matrix is required in the columns as attributes(features) and rows as records (without output labels)
        (Type: float) IP:---> IP is the threshold of the information that is should be preserved at minimum (Information Preserved). If value of PCA_vectors variable is non zero. then the IP value will get override.
        (Type: int) PCA_vectors:---> Number of vectors to have in the transformation basis. If it is 0 then the vectors will be in such manner that the information preserved is atleast till IP level
        

        Return:
        (Type: numpy.array) Ready_Transform_basis:---> This is the transformation matrix. it can be used to calculate the projection of the data matrix. It consists of most significant vectors as column entry.
        (Type: numpy.array) projected_data:---> This contains the projected matrix. ie. dot product of Mat and Ready_Transform_basis
        (Type: float) information_Retained:---> This gives the floating point number indicationg the information retained by the Ready_transforma_basis

        '''

        # function to calulate the covariance and the correlation of the matrix
        CovMat, CorrMat = self.covNcorr(Mat)

        # finding the eigenvalues and eigenvectors of covariance matrix
        e,v= np.linalg.eig(CovMat)

        # sorting using dataframes
        df= pd.DataFrame([e,v]).transpose()
        df.columns=(["EigenValue","EigenVector"])
        dfsorted=df.sort_values(by= "EigenValue",ascending=False)
        eigenvalues_sorted= np.array(dfsorted["EigenValue"])
        eigenvectors_sorted= np.array(dfsorted["EigenVector"])

        # calculating the information preserved and the principal components
        sum_eigenvalues= np.sum(eigenvalues_sorted)
        eigen_temp_sum=0.0
        IP_calc= 0.0
        Transform_matrix= np.transpose(eigenvectors_sorted[0]).copy()
        i=0
        flag=1
        while flag:
            eigen_temp_sum= eigen_temp_sum+ eigenvalues_sorted[i]
            IP_calc= eigen_temp_sum/sum_eigenvalues *100
            if self.print_msg:
                print("Adding Component No.", i+1)
            if i!=0:
                Transform_matrix = np.append(Transform_matrix,np.transpose(eigenvectors_sorted[i]))
            i=i+1
            if self.print_msg:
                print("Information Retained = ", IP_calc)
            if PCA_vectors==0:
                if IP <= IP_calc or IP_calc==100:
                    flag=0
            else:
                if i>=PCA_vectors or IP_calc==100:
                    flag=0
        Ready_Transform_basis= np.transpose(Transform_matrix.reshape(i,eigenvectors_sorted[0].shape[0])).copy()
        if self.print_msg:
            print("+++++++++++++++++++++++++++++++Transform_basis+++++++++++++++++++++++++")
            print(Ready_Transform_basis)
            print("Projected Data:")

        #calculating the projection matrix
        projected_data= np.dot(Mat,Ready_Transform_basis)
        if self.print_msg:
            print(projected_data)

        #information retained
        information_Retained= IP_calc
       
        return Ready_Transform_basis, projected_data, information_Retained

    # End of the PCA() function


