'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        self.nIters = numBoostingIters
        self.maxDepth = maxTreeDepth


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        K = 3
        n, d = X.shape
        weight = np.ones(y.shape) * (1.0 / n)
        self.h_arr = [None] * self.nIters
        self.alpha_arr = [None] * self.nIters
        self.y = y
        for i in range(0, self.nIters):
            # fit and save classifier(i) using weight
            modelDT = DecisionTreeClassifier(max_depth=1)
            modelDT.fit(X, y, sample_weight=weight)
            self.h_arr[i] = modelDT
            # compute error(i)
            error = 0
            ypred = modelDT.predict(X)
            for j in range(0, n):
                error += (ypred[j] != y[j]) * weight[j]
            error /= float(np.sum(weight))
            # compute and save alpha(i)
            alpha = 0.5 * np.log((1 - error) / error) + np.log(K - 1)
            self.alpha_arr[i] = alpha
            # update weight(i)
            for k in range(0, n):
                weight[k] = weight[k] * np.exp(alpha * (ypred[k] != y[k]))


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n, d = X.shape
        k_set = list(set(self.y))
        dict = np.zeros(shape=(len(k_set), n))

        for i in range(0, self.nIters):
            ypred = self.h_arr[i].predict(X)
            for j in range(0, n):
                k = k_set.index(ypred[j])
                dict[k, j] += self.alpha_arr[i]

        max_index = np.argmax(dict, axis=0)
        result = [0] * n
        for i in range(0, n):
            result[i] = k_set[max_index[i]]

        return result