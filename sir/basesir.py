from common.config import dtype
from numpy import inf


class BaseSIR(object):
    '''The abstract base for the SIR model and its variant models'''
    def __init__(self, params):
        self.init_i = dtype(params.get('init_i', 0.0))
        self.set_init_i(self.init_i)
        
        self.epoch = 0
        self.epochs = params.get('epochs', 52)

        self.fit(params, False)

    def check_bounds(self, x, low_bnd=0, up_bnd=1):
        if x < low_bnd: 
            x = 0.0
        elif x > up_bnd: 
            x = 1.0
        return x

    def fit(self, params, refit=False):
        pass

    def set_init_i(self, i, s=inf):
        self.i = dtype(i)
        self.s = 1 - i if dtype(s) is inf else i 
        self.Is = [self.i]
        self.Ss = [self.s]
