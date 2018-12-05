from numpy import zeros, eye, dot
from common.linalg import as_matrix


class KalmanFilter(object):

    def __init__(self, num_states, num_obs, A, B, V, W, P=None):
        self.num_states = num_states
        self.num_obs = num_obs

        self.A = None if A is None else as_matrix(A)
        self.B = None if B is None else as_matrix(B)
        self.V = as_matrix(V) 
        self.W = as_matrix(W) 
        
        self.x_prior = as_matrix(zeros(num_states)).T
        self.x_post = as_matrix(zeros(num_states)).T
        self.P_prior = P if P is not None else \
                       as_matrix(zeros((num_states, num_obs)))
        self.P_post = P if P is not None else \
                      as_matrix(zeros((num_states, num_obs)))

        self.K = as_matrix(zeros((num_states, num_obs)))
        self.I = as_matrix(eye(num_states))

    def step(self, obs, predict_x=False, predict_P=True):
        self.H = self.get_H()
        self.K = self.get_K()
        K = self.K.copy()
        z = obs - self.B @ self.x_prior
        
        self.cali = self.K @ z
        self.x_post = self.x_prior + self.cali
        self.K = K

        if predict_P:
            self.P_post = (self.I - self.K @ self.B) @ self.P_prior
            self.P_prior = self.A @ (self.P_post @ self.A.T) + self.V

        if predict_x:
            self.x_prior = self.A @ self.x_post


    def fit(self, x, P=None):
        self.x_prior = x.copy()
        self.x_post = x
        
        if P is not None:
            self.P_prior = P.copy()
            self.P_post = P

    def get_H(self):
        return (self.B @ (self.P_prior @ self.B.T) + self.W).I

    def get_K(self):
        return self.P_prior @ self.B.T @ self.H
