"""
======================================================
Test the boostedDT against the standard decision tree
======================================================

Author: Eric Eaton, 2014

"""
print(__doc__)

import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from boostedDT import BoostedDT

# load the data set
filename = 'data/challengeTrainLabeled.dat'
train_data = np.loadtxt(filename, delimiter=',')
n,d = train_data.shape
Xtrain = train_data[:, 0:d-1]
ytrain = train_data[:, d-1]

filename = 'data/challengeTestUnlabeled.dat'
Xtest = np.loadtxt(filename, delimiter=',')

# train the decision tree
modelDT = DecisionTreeClassifier()
modelDT.fit(Xtrain,ytrain)

# train the boosted DT
modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=1)
modelBoostedDT.fit(Xtrain,ytrain)

# output predictions on the remaining data
ypred_DT = modelDT.predict(Xtest)
ypred_BoostedDT = modelBoostedDT.predict(Xtest)

print ypred_BoostedDT
# compute the training accuracy of the model
#accuracyDT = accuracy_score(ytest, ypred_DT)
#accuracyBoostedDT = accuracy_score(ytest, ypred_BoostedDT)

#print "Decision Tree Accuracy = "+str(accuracyDT)
#print "Boosted Decision Tree Accuracy = "+str(accuracyBoostedDT)