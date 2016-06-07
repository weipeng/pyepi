from numpy import sum, power, ones, mean, sqrt
try:
    from scipy import stats
except:
    print 'No scipy'
from linalg import as_array

def RSS(x, y, idx=0):
    return ((as_array(x)[idx:] - as_array(y)[idx:]) ** 2).sum()

def RWSE(x, y, idx=0):
    x, y = as_array(x), as_array(y)
    w = sum(x[idx:])
    return (((x[idx:] - y[idx:]) * x[idx:]/w) ** 2).sum()

def RMSE(x, y, idx=0):
    return sqrt(MSE(x, y, idx))

def MSE(X, Y, idx=0, axis=None):
    return power((as_array(X)[idx:] - as_array(Y)[idx:]), 2).mean(axis=axis)

def MSPE(x, y, idx=0):
    '''x is the base array for comparison
       s.t. 
            error = \sum_i ((x_i - y_i) / x_i ** 100) ^ 2        
    '''
    return mean((ones(len(x[idx:])) - as_array(y[idx:]) / as_array(x[idx:])) ** 2)

def confidence_interval(data, confidence=0.95):
    a = as_array(data)
    if len(a.shape) > 1:
        axis, n = 0, a.shape[1]
    else:
        axis, n = None, a.shape[0]

    m, se = mean(a, axis=axis), stats.sem(a, axis=axis)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m - h, m, m + h

