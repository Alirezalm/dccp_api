from numpy import exp, log, diagflat
from numpy.random import rand, randn



class LogRegProb(object):

    def __init__(self, local_dataset, local_response):
        self.local_dataset = local_dataset
        self.local_response = local_response

    def compute_obj_at(self, x):
        n = x.shape[0]
        x = x.reshape(n, 1)
        h = self._logistic_func(x)
        obj_val = -self.local_response.T @ log(h) - (1 - self.local_response).T @ log(1 - h)
        return obj_val.astype(float)

    def compute_grad_at(self, x):
        h = self._logistic_func(x)

        return self.local_dataset.T @ (h - self.local_response)

    def compute_hess_at(self, x):
        h = self._logistic_func(x)
        return self.local_dataset.T @ diagflat(h * (1 - h)) @ self.local_dataset

    def _logistic_func(self, x):
        x = x.reshape(x.shape[0], 1)
        z = self.local_dataset @ x
        h = 1 / (1 + exp(-z))
        h[h == 1] = 1 - 1e-8
        h[h == 0] = 1e-8
        return h

    def __str__(self):
        return "Sparse Logistic Regression Problem"


class QuadConsProb(object):
    def __init__(self, problem_data):
        self.obj_hess = problem_data['obj_hess']
        self.obj_vec = problem_data['obj_vec']
        self.obj_const = problem_data['obj_const']

    def compute_obj_at(self, x):
        return 0.5 * x.T @ self.obj_hess @ x + self.obj_vec.T @ x + self.obj_const

    def compute_grad_at(self, x):
        return self.obj_hess @ x + self.obj_vec

    def compute_hess_at(self, x):
        return self.obj_hess
