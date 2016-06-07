from numpy import matrix, array
from config import dtype

def as_matrix(x, data_type=dtype):
    return matrix(x, data_type)

def as_array(x, data_type=dtype):
    return array(x, data_type)

def uncentered_cov(X):
    '''Uncentered covariance matrix'''
    if X.shape[1] <= 1: return None
    return X.dot(X.T) / (X.shape[1] - 1)
