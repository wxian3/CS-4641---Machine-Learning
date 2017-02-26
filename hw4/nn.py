'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from sklearn import preprocessing

class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate=1.8, numEpochs=100):
        '''
        Constructor
        Arguments:
            layers - a numpy array of L-2 integers (L is # layers in the network) 
            epsilon - one half the interval around zero for setting the initial weights
            learningRate - the learning rate for backpropagation
            numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''

        n,d = X.shape

        self.weights = {}
        # initialize weights of input layer 
        self.weights[1] = np.random.uniform(-self.epsilon, self.epsilon, (self.layers[0], d + 1))
        # initialize weights of hidden layers 
        for i in range(1, len(self.layers)):
            self.weights[i + 1] = np.random.uniform(-self.epsilon, self.epsilon, (self.layers[i], self.layers[i - 1] +1))
        # initialize weights of output layer
        self.weights[len(self.layers) + 1] = np.random.uniform(-self.epsilon, self.epsilon, (len(set(y)), self.layers[len(self.layers) - 1] +1))

        labelBinarizer = preprocessing.LabelBinarizer()
        labelBinarizer.fit(y)
        binary_labels = labelBinarizer.transform(y)

        #initialize gradients and error
        gradients = [0] * (len(self.weights) + 1)
        error = {}

        for e in range(self.numEpochs):
            self.forward_propogation(X,self.weights)

            error[len(self.layers) + 1] = self.activated[len(self.layers) + 1][:, 1:] - binary_labels
            for i in range(len(self.layers), 0, -1): 
                error[i] = error[i + 1].dot(self.weights[i + 1][:, 1:]) * self.activated[i][:, 1:] * (1- self.activated[i][:, 1:])
            
            for i in range(len(self.weights)):
                gradients[i + 1] = gradients[i + 1] + error[i + 1].T.dot(self.activated[i])
                reg_term = np.c_[np.zeros([self.weights[i + 1].shape[0], 1]), self.weights[i + 1][:, 1:]] * 0.0001
                gradients[i + 1] = (gradients[i + 1] / n) + reg_term 
                self.weights[i + 1] = self.weights[i + 1] - self.learningRate * gradients[i + 1]

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        self.forward_propogation(X, self.weights)
        output = self.activated[len(self.activated) - 1]
        predicted = np.argmax(output[:, 1:], axis=1)
        return predicted
    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        
    def forward_propogation(self, X, weights):        
        self.activated = {}

        self.activated[0] = np.c_[np.ones(X.shape[0]), X]
        for i in range(1, len(self.layers) + 2):
            self.activated[i] = self.sigmoid(self.activated[i-1].dot(weights[i].T))
            self.activated[i] = np.c_[np.ones(self.activated[i].shape[0]), self.activated[i]] 

    def sigmoid(self, zeta):
        return 1.0 / (1.0 + np.exp(-1.0 * zeta)) 
