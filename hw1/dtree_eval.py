'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score



def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    accuracy_score_arr = []
    accuracy_score_arr_1 = []
    accuracy_score_arr_3 = []

    for trail in range(0,100):
        # Load Data
        filename = 'data/SPECTF.dat'
        data = np.loadtxt(filename, delimiter=',')
        X = data[:, 1:]
        y = np.array([data[:, 0]]).T
        n,d = X.shape

        # shuffle the data
        idx = np.arange(n)
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        for k in range(0, 10):
            # split the data
            Xtest = X[k * len(X)/10 : (k + 1) * len(X)/10,:]  # train on 1/10 instances
            Xtrain = np.concatenate((X[0 : k * len(X)/10, :], X[(k + 1) * len(X)/10: ,:]), axis=0)
            ytest = y[k * len(X)/10 : (k + 1) * len(X)/10,:]  # test on remaining instances
            ytrain = np.concatenate((y[0 : k * len(y)/10, :], y[(k + 1) * len(y)/10: ,:]), axis=0)

            # train the basic decision tree 
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(Xtrain,ytrain)
            # output predictions on the remaining data
            y_pred = clf.predict(Xtest)
            # compute the training accuracy of the model
            accuracy_score_arr.append(accuracy_score(ytest, y_pred))
            
            # train the 1-level decision tree 
            clf = tree.DecisionTreeClassifier(max_depth=1)
            clf = clf.fit(Xtrain,ytrain)
            # output predictions on the remaining data
            y_pred = clf.predict(Xtest)
            # compute the training accuracy of the model
            accuracy_score_arr_1.append(accuracy_score(ytest, y_pred))

            # train the 3-level decision tree 
            clf = tree.DecisionTreeClassifier(max_depth=3)
            clf = clf.fit(Xtrain,ytrain)
            # output predictions on the remaining data
            y_pred = clf.predict(Xtest)
            # compute the training accuracy of the model
            accuracy_score_arr_3.append(accuracy_score(ytest, y_pred))

    # update these statistics based on the results of your experiment
    meanDecisionTreeAccuracy = np.mean(accuracy_score_arr)
    stddevDecisionTreeAccuracy = np.std(accuracy_score_arr)
    meanDecisionStumpAccuracy = np.mean(accuracy_score_arr_1)
    stddevDecisionStumpAccuracy = np.std(accuracy_score_arr_1)
    meanDT3Accuracy = np.mean(accuracy_score_arr_3)
    stddevDT3Accuracy = np.std(accuracy_score_arr_3)

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print "Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")"
    print "Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")"
    print "3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")"
# ...to HERE.
