'''
    AUTHOR Wenqi Xian
'''

from numpy import loadtxt
import numpy as np

from nn import NeuralNet

X_train = loadtxt('data/digitsX.dat', delimiter=',')
y_train = loadtxt('data/digitsY.dat', delimiter=',')
layers = np.array([25])

NN = NeuralNet(layers = layers, learningRate = 1.8, numEpochs = 700)
NN.fit(X_train,y_train)
predicted = NN.predict(X_train)
accuracy = 100.0 * (predicted == y_train).sum() / y_train.shape[0]
print accuracy