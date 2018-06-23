from scipy.stats import norm
from numpy import cov, tile, std, average
from numpy.random import choice, multivariate_normal
from common.linalg import as_array, init_weights


class ParticleFilter(object):

    def __init__(self, num_part, params={}):
        self.num_part = num_part
        self.weights = init_weights(num_part)
        
        self.x_prior = None
        
    def step(self, y, predict_P=False, index=1):
        X = self.x_prior
        states = X[:2, :]
        params = X[2:, :]
        
        s_std = std(X[index].A1) 
        
        tmp_ws = as_array([norm.pdf(y, x[0, index], s_std) for x in states.T])
        n_weights = self.weights * tmp_ws
        sum_weights = n_weights.sum()
    
        if sum_weights != 0:
            n_weights /= sum_weights
            neff = 1.0 / (n_weights ** 2).sum() 
        
        if sum_weights == 0 or neff < self.num_part/2.0:
            idx = choice(range(X.shape[1]), X.shape[1], p=self.weights)
            self.weights = tile(as_array(1.0 / self.num_part), self.num_part)
            
            self.x_post = X[:, idx]
        else:
            self.x_post = X
            self.weights = n_weights

        p_mean = average(params, axis=1, weights=self.weights).A1
        p_cov = cov(params, aweights=self.weights)
        self.x_post[2:, :] = multivariate_normal(p_mean, p_cov, X.shape[1]).T
 
        for i, x in enumerate(self.x_post[2:, :].T):
            if x.any() < 0:
                while True:
                    new = multivariate_normal(p_mean, p_cov, 1).T
                    if new.all() > 0 and new[0, 1] > new[0, 2]:
                        self.x_post[2:, i] = new
                        break
        
    def fit(self, x):
        self.x_prior = x 
