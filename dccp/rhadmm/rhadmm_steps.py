from numpy import zeros, inf
from numpy.linalg import norm
from numpy.random import randn
from scipy.optimize import minimize


def update_primary_vars(rhadmm_obj, rhadmm_grad, n_vars, constr = None):
    initial_condition = zeros((n_vars,))
    method = 'CG'
    if constr is not None:
        quad_constr = []
        # for const in


        method = 'trust-constr'

    solver = minimize(rhadmm_obj, jac = rhadmm_grad, x0 = initial_condition, method = method, options = {},
                      constraints = constr)
    if solver.success:
        return solver.x.reshape(n_vars, 1)
    else:
        raise ValueError("Inner Solver inside RHADMM failed.")
