import sys
sys.path.append('..')
from basesir import BaseSIR
from common.stats import RSS, RMSE, MSPE
from common.linalg import as_matrix, dtype as data_type
from common.utils import get_data
from numpy import eye, ones, corrcoef
from filtering.kalmanfilter import KalmanFilter 


class SIR(BaseSIR):
    '''The linear SIR model
    '''
    def __init__(self, params):
        super(SIR, self).__init__(params)
        self.CDC_obs = get_data(self.CDC)
        
    def fit(self, params, refit=False):
        if not refit:
            self.filter_type = params.get('filter_type', '')
            assert 'beta' in params, 'The paramter beta is missing.'
            self.beta = float(params['beta'])

            assert'alpha' in params, 'The paramter alpha is missing.'
            self.alpha = float(params['alpha'])

            assert'CDC' in params, 'The paramter CDC is missing.'
            self.CDC = params['CDC']

            self.filter = None
        if refit:
            self.filter_type = params.get('filter_type', '')
            self.beta = float(params.get('beta', 0)) or self.beta
            self.alpha = float(params.get('alpha', 0)) or self.alpha

            self.CDC = params.get('CDC') or self.CDC

            self.epochs = params.get('epochs', 52)
            self.epoch = params.get('epoch', 0)

        self.init_i = float(params.get('init_i', self.i))
        self.set_init_i(self.init_i)
        self.score = 0.0

        self.filtering = params.get('filtering', False)
        if self.filtering: 
            self._init_filter()

        return self

    def predict(self):
        while self.epoch < self.epochs - 1:
            self.update_states()    
            self.epoch += 1
        self.get_score()

    def update_states(self):
        self.s += self._delta_s()
        self.i += self._delta_i()

        self.s = self.check_bounds(self.s)
        self.i = self.check_bounds(self.i)

        self.Is.append(self.i)
        self.Ss.append(self.s)

    def predict_with_filter(self):
        if not self.filtering or not self.filter:
            raise Exception('The filtering flag must be set True, \
                             and the filter needs to be inialized')

        F = self.filter 
        while self.epoch < self.epochs - 1:
            x = as_matrix([self.s, self.i]).T
            F.fit(x)
            y = as_matrix([self.CDC_obs[self.epoch]]).T
            F.step(y, verbose=False)

            self.s = self.check_bounds(F.x_post[0, 0])
            self.i = self.check_bounds(F.x_post[1, 0])
            self.update_states()
            self.epoch += 1
            
        self.get_score()
        self.score 
                    
    def _delta_s(self):
        return - self.alpha * self.s * self.i 

    def _delta_i(self):
        return self.alpha * self.s * self.i - self.beta * self.i

    def get_score(self):
        self.outcome = [x for i, x in enumerate(self.Is)]
        self.scores = {}
        self.scores['SSE'] = RSS(self.outcome, self.CDC_obs, 1)
        self.scores['RMSE'] = RMSE(self.CDC_obs, self.outcome, 1)
        self.scores['MSPE'] = MSPE(self.CDC_obs, self.outcome, 1)
        self.scores['CORR'] = corrcoef(self.CDC_obs, self.outcome, 1)

    def _init_filter(self):
        num_states = 2
        num_obs = 1
        A = as_matrix([[1, -self.alpha], 
                       [0, 1 + self.alpha - self.beta]])

        B = as_matrix([0, 1])

        Cov = eye(num_states, dtype=data_type) * 0.0001

        V = Cov.copy()
        W = eye(num_obs, dtype=data_type) * 0.0001
        
        self.filter = KalmanFilter(num_states, num_obs, A, B, V, W, Cov)
    
