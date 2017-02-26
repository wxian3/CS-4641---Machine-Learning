import numpy as np

def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    n = len(x1)
    X1 = np.array(x1)
    X2 = np.array(x2)

    polyX1 = X1
    for i in range(0, 5): 
        polyX1 = np.c_[polyX1, X1 ** (i + 1)]

    polyX2 = X2
    for j in range(0, 5):
        polyX2 = np.c_[polyX2, X2 ** (j + 1)]
    
    mapFeature = np.array(np.ones([n, 28]))
    mapFeature[:, 0] = 1
    mapFeature[:, 1] = X1
    mapFeature[:, 2] = X2
    i = 3
    for m in range(0, 5):
        for n in range(0,5):
            mapFeature[:, i] = polyX1[:, m] * polyX2[:, n] 
            i = i + 1
        
    return mapFeature
        

