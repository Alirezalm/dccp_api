from numpy import zeros, inf
from numpy.linalg import norm
from numpy.random import randn
from scipy.optimize import minimize


def update_primary_vars(rhadmm_obj, rhadmm_grad, n_vars, sfp, kappa, bound, x):
    initial_condition = zeros((n_vars,))
    constr = ()
    method = 'cg'
    # if sfp:
    #     method = 'SLSQP'
    #     constr = (
    #         {
    #             'type': 'ineq', 'fun': lambda x_var: norm(x_var, 1) - kappa * bound
    #         }
    #     )
    x = minimize(rhadmm_obj, jac = rhadmm_grad, x0 = initial_condition, method = method, options = {},
                 constraints = constr)['x']
    # print(f"{method}")
    return x.reshape(n_vars, 1)
