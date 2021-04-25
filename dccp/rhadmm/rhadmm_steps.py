from numpy import zeros, inf
from numpy.linalg import norm
from numpy.random import randn
from scipy.optimize import minimize


def update_primary_vars(rhadmm_obj, rhadmm_grad, n_vars, sfp, kappa, bound, x):
    initial_condition = zeros((n_vars,))
    constr = ()
    method = 'CG'
    # if sfp:
    #     method = 'SLSQP'
    #     constr = (
    #         {
    #             'type': 'ineq', 'fun': lambda x_var: norm(x_var, 1) - kappa * bound
    #         }
    #     )
    solver = minimize(rhadmm_obj, jac = rhadmm_grad, x0 = initial_condition, method = method, options = {})
    if solver.success:
        return solver.x.reshape(n_vars, 1)
    else:
        raise ValueError("Inner Solver inside RHADMM failed.")
