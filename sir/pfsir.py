from .sir import SIR
from common.config import data_type
from common.linalg import as_array, as_matrix, init_weights
from common.stats import RSS, MSPE, RMSE
from numpy.random import normal, uniform
from numpy import *
from filtering.particlefilter import ParticleFilter


class ParticleSIR(SIR):
    
    def __init__(self, num_enbs, params):
        self.num_enbs = num_enbs
        super(ParticleSIR, self).__init__(params)

        del self.alpha
        del self.beta
        
        self.current_Is = uniform(0, self.i * 2, num_enbs)
        self.current_Ss = ones(num_enbs) - self.current_Is
        self.alphas = uniform(0., 1, num_enbs)
        self.betas = uniform(0., 1, num_enbs)

        self.weights = [init_weights(num_enbs)] # matrix-like

        for i in range(num_enbs):
            if self.alphas[i] < self.betas[i]:
                self.alphas[i], self.betas[i] = self.betas[i], self.alphas[i]  

        self.Is = [self.current_Is.tolist()]
        self.Ss = [self.current_Ss.tolist()]

    def update_states(self):
        for j in range(self.num_enbs):
            s = self.current_Ss[j]
            i = self.current_Is[j]
            s += self._delta_s(self.current_Ss[j], self.current_Is[j], 
                               self.alphas[j])
            i += self._delta_i(self.current_Ss[j], self.current_Is[j], 
                               self.alphas[j], self.betas[j])

            s = self.check_bounds(s)
            i = self.check_bounds(i)

            self.current_Is[j] = i
            self.current_Ss[j] = s

        self.Is.append(self.current_Is.tolist())
        self.Ss.append(self.current_Ss.tolist())

    def _init_filter(self):
        num_states = 4
        num_obs = 1
        
        self.filter = ParticleFilter(self.num_enbs)

    def predict_with_filter(self):
        F = self.filter

        while self.epoch < self.epochs - 1:
            X = as_matrix([self.current_Ss, self.current_Is, 
                           self.alphas, self.betas])
        
            F.fit(X)
            y = self.CDC_obs[self.epoch]
            F.step(y, predict_P=False)
            self.weights.append(F.weights)

            x_post = F.x_post
            for j in range(self.num_enbs):
                self.current_Ss[j] = self.check_bounds(x_post[0, j])
                self.current_Is[j] = self.check_bounds(x_post[1, j])
                self.alphas[j] = self.check_bounds(x_post[2, j], inf)
                self.betas[j] = self.check_bounds(x_post[3, j], inf)

            self.update_states()
            self.epoch += 1

        self.get_score()

    def _delta_s(self, s, i, alpha):
        return - alpha * s * i

    def _delta_i(self, s, i, alpha, beta):
        return alpha * s * i - beta * i

    def check_par_bounds(self, par):
        if par < 0: par = 0
        return par

    def get_score(self):
        I_mat = as_array(self.Is)
        for i, w in enumerate(self.weights):
            I_mat[i] *= w 

        self.IS = sum(I_mat, axis=1)

        time_gap = self.epochs / 52
        idx = [x for x in range(self.epochs) if not x % time_gap]

        self.score = RSS(self.CDC_obs, self.IS[idx])
        self.scores = {}
        self.scores['SSE'] = self.score
        self.scores['RMSE'] = RMSE(self.CDC_obs, self.IS[idx])
        self.scores['MSPE'] = MSPE(self.CDC_obs, self.IS[idx])
        self.scores['CORR'] = corrcoef(self.CDC_obs, self.IS[idx])[0, 1]
        return self.score
