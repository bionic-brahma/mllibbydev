#!/usr/bin/env python
# coding: utf-8




#######################################################################################
## Polynomial regression is progress. remaining work:                                 #
## applying the degree parameter to the model.                                        #
##                                                                                    #
## Work by Devendra Kumar for Risk Latte AI Inc.                                      #
#######################################################################################


class regression:
    
    '''
    Work in progress
    '''
    
    
    def __init__(self, iteration=10, learning_rate=0.01, degree=1, print_msg = False):
        
        self.iterat=iteration
        self.lr=learning_rate
        self.degree=degree
        self.weights=None
        self.bias=None
        self.print_msg=print_msg

        
#################################################################
  ##  Working on the polynomial regression
#################################################################
    '''
    def factorial(self,n):
        if n<=1:
          return 1
        else:
          return n*self.factorial(n-1)

    def combination(self,n,r):
        return self.factorial(n)/(self.factorial(r)*self.factorial(n-r) )   
    '''
##################################################################
  
    def fit(self, input_X, output_Y):
        
        '''
        Work in progress

        '''
        nrecords, nfeatures = input_X.shape
        
        #num_non_linear_features= self.combination(self.degree+nfeatures-1,nfeatures-1)  #m-1+nCm-1

        if self.print_msg:
            print("records= ", nrecords, "  features= ", nfeatures)
        self.weights= np.zeros(nfeatures)

        self.bias=0.0
        
        if self.print_msg:
            fig= plt.figure()
            f= fig.add_subplot(111)
            f.scatter(input,output)

        for it in range(self.iterat):
            model= np.dot(input_X, self.weights) + self.bias
            loss= (1/nrecords)* np.sum((model- output_Y)**2)
            self.weights= self.weights-(1/nrecords)*self.lr*2*np.dot(np.transpose(input_X),(model-output_Y))
            self.bias-= (1/nrecords)*self.lr*2*np.sum(model-output_Y) 
            
            if self.print_msg:
                print("-------> iteration number: ",it,"  Loss: ",loss)
                print("-------> weights: ",self.weights)

            y=self.predict(input)
            
            if self.print_msg:
                if it%40==0 or it== self.iterat-1:
                    f.plot(input,y,color="red", alpha=self.iterat/(it+1))

    def predict(self, input_X):
        return (np.dot(input_X ,self.weights) + self.bias)[:,0]


########################## # END # ##############################

