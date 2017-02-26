'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing

    def extract_counts(self, L):
        """
        helper function: Take a 1D numpy array as input and return a dict mapping values to counts
       """
        uniques = set(list(L))

        counts = dict((u, np.sum(L == u)) for u in uniques)
        return counts


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        m, n = X.shape
        self.unique_labels = list(set(y))
        self.K = len(self.unique_labels)
        self.feature_counts = [None] * self.K
        self.label_counts = [None] * self.K
        self.all_prior = [None] * self.K
        for k in range(0, self.K):
            self.feature_counts[k] = [self.extract_counts(L) for L in X[y == self.unique_labels[k]].T]
            self.label_counts[k] = float(sum(y == self.unique_labels[k]))
        total = np.sum(self.label_counts)
        for k in range(0, self.K):
            self.all_prior[k] = self.label_counts[k]/total

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        m, n = X.shape
        predictProbs = self.predictProbs(X)
        max_index = np.argmax(predictProbs, axis=1)
        result = [0] * m
        for i in range(0, m):
            result[i] = self.unique_labels[max_index[i]]
        return result
    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''

        m, d = X.shape
        result = np.zeros(shape=(m, self.K))
        for i, xi in enumerate(X):
            Pxi = np.zeros(shape=(self.K, d))
            for k in range(0, self.K):
                for j, v in enumerate(xi):
                    count = self.feature_counts[k]
                    nc = count[j].get(v, 0)
                    nn = len(count[j])
                    # Compute probabilities with laplace smoothing
                    Pxi[k, j] = (nc + 1) / (self.label_counts[k] + nn)
                # Compute the predicted class probabilities (for K classes)
                result[i, k] = np.log(self.all_prior[k]) + np.sum(np.log(Pxi[k]))

        return result
