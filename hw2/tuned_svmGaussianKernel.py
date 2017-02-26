"""
=====================================
Test SVM with custom Gaussian kernels
=====================================

Author: Eric Eaton, 2014

Adapted from scikit_learn documentation.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from numpy import linalg as LA
import random

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
    # vectorization!!!
    res = np.matrix(np.zeros([n1, n2]))
    for i in range(n1):
        diff = np.subtract(X2, X1[i])
        sum = np.sum(np.power(diff, 2), axis = 1)   
        res[i] = np.exp(-sum / (2 * _sigma ** 2)) 

    return res

# load the data
#iris = datasets.load_iris()
#X = iris.data[:, :2]  # we only take the first two features
#Y = iris.target


# load the data
filename = "data/svmTuningData.dat"
allData = np.loadtxt(filename, delimiter=',')
np.random.shuffle(allData)

X = allData[:,:-1]
Y = allData[:,-1]


# Split the dataset in two parts
X_train = X[0 : len(X)/2] 
X_test = X[len(X)/2 : len(X)] 
y_train = Y[0 : len(X)/2] 
y_test = Y[len(X)/2 : len(X)] 

print "Training the SVMs..."

tuned_sigma = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100]
tuned_c = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100]

best_score = 0

for i in range(0, len(tuned_c)):
    for j in range(0, len(tuned_sigma)):
        _sigma = tuned_sigma[j]
        myModel = svm.SVC(C = tuned_c[i], kernel=myGaussianKernel)
        myModel.fit(X_train, y_train)
        score = myModel.score(X_test, y_test)
        # test
        print score
        if (score > best_score):
            best_score = score
            best_c = tuned_c[i]
            best_sigma = tuned_sigma[j]

_sigma = best_sigma
myModel = svm.SVC(C = best_c, kernel=myGaussianKernel)
myModel.fit(X, Y)

print "best sigma: ", best_sigma
print "best c: ", best_c
print "best score: ", best_score

print ""
print "Testing the SVMs..."

h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# get predictions for both my model and true model
myPredictions = myModel.predict(np.c_[xx.ravel(), yy.ravel()])
myPredictions = myPredictions.reshape(xx.shape)

# plot my results
plt.pcolormesh(xx, yy, myPredictions, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired) # Plot the training points
plt.axis('tight')

plt.show()

