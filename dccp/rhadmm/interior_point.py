from numpy import log
from scipy.optimize import minimize


def interior_point(objective, const, init):

    t = 1e-3
    miu = 10
    def ip_obj(x_t):

        return t * objective(x_t) - log(-const(x_t))

    max_iter = 100
    eps = 1e-3
    for i in range(max_iter):

        x = minimize(ip_obj, x0 = init, method = 'cg')

        if 1 / t < eps:

            return x
        t *= miu

    return 0
