'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from numpy import linalg as LA

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters


    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        cost = -y.T.dot(np.log(self.sigmod(X.dot(theta)))) - (1 - y).T.dot(np.log(1 - self.sigmod(X.dot(theta)))) + 0.5 * regLambda * LA.norm(theta,2)
        return cost.item(0)


    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        n,d = X.shape
        gradient = np.zeros((d, 1))
        gradient[0] = np.sum(self.sigmod(X.dot(theta)) - y)
    
        for i in range(1, d):
            gradient[i] = np.sum(np.dot(X.T, self.sigmod(X.dot(theta)) - y) + regLambda * theta[i])

        return gradient


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n,d = X.shape
        X= np.c_[np.ones((n, 1)), X]
        n,d = X.shape
        theta = np.matrix(np.zeros((d,1)))

        for i in xrange(self.maxNumIters):
            oldtheta = theta
            theta = theta - self.alpha * self.computeGradient(theta, X, y, self.regLambda)
            print 'Gradient:', self.computeGradient(theta, X, y, self.regLambda)
            print 'theta', theta
            print 'Cost:', self.computeCost(theta, X, y, self.regLambda)
            print 'Changing:', LA.norm(theta - oldtheta)
            if LA.norm(theta - oldtheta) <= self.epsilon:
                self.theta = theta
                break
        
        self.theta = theta

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n,d = X.shape
        X= np.c_[np.ones((n, 1)), X]
        Y = np.array(self.sigmod(X * self.theta))
        for i in range(len(Y)):
            if (Y[i] > 0.5): 
                Y[i] = 1
            else: 
                Y[i] = 0
                
        return Y

    def sigmod(self, z):
        return 1 / (1 + np.exp(-z))
