from numpy import zeros, inf
from numpy.linalg import norm
from numpy.random import randn
from scipy.optimize import minimize, NonlinearConstraint


def update_primary_vars(rhadmm_obj, rhadmm_grad, n_vars, constrs = None):
    initial_condition = zeros((n_vars,))
    method = 'CG'
    quad_constr = []
    if constrs is not None:
        method = 'trust-constr'
        quad_constr = []
        for constr in constrs:
            quad_constr.append(NonlinearConstraint(
                lambda x: x.T @ constr['hessian_mat'] @ x + constr['grad_vec'].T @ x + constr['const'],
                -inf, 0
            ))

    solver = minimize(rhadmm_obj, jac = rhadmm_grad, x0 = initial_condition, method = method, options = {},
                      constraints = quad_constr)
    if solver.success:

        return solver.x.reshape(n_vars, 1)
    else:
        raise ValueError("Inner Solver inside RHADMM failed.")
