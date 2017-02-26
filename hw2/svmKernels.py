"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np
from numpy import linalg as LA

_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    return (np.dot(X1, X2.T) + 1)**_polyDegree


def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    res = np.matrix(np.zeros([n1, n2]))
    for i in range(n1):
        diff = np.subtract(X2, X1[i]) 
        sum = np.sum(np.power(diff, 2), axis = 1)    # sum is 1 by n2
        res[i] = np.exp(-sum / (2 * _gaussSigma ** 2))    

    return res


