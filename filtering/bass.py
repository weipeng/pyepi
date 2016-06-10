import sys
sys.path.insert(0, '..')
from common.linalg import as_array, as_matrix
from numpy import std, where, zeros
from numpy.random import choice, multivariate_normal
from scipy.stats import norm


class Bass(object):
    
    @staticmethod
    def step(X, X_out, w, y, err, R, index=1):
        s_std = std(X[index].A1)  
        tmp_ws = as_array([norm.pdf(y, x[0, index], s_std) for x in X.T])
        tmp_ws *= w 
        tmp_ws /= tmp_ws.sum()

        s_idx = where(tmp_ws >= err)[0]
        l_idx = where(tmp_ws < err)[0]

        s_ws = tmp_ws[s_idx] / tmp_ws[s_idx].sum()
        if l_idx.shape[0] > 0:
            noise = multivariate_normal(zeros(X.shape[0]), R, l_idx.shape[0])
            idx = choice(s_idx, l_idx.shape[0], p=s_ws)
            X_out[:, l_idx] = X_out[:, idx] + as_matrix(noise).T
            tmp_ws[l_idx] = tmp_ws[idx]
        
        w = tmp_ws / tmp_ws.sum()

        return X_out, w, l_idx.shape[0]
            
            
