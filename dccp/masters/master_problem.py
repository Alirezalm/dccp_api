import gurobipy as gp
from gurobipy import GRB
from numpy import ones


def solve_master(problem_instance, cut_manager):
    model = gp.Model('master')

    n = problem_instance.nVars
    N = problem_instance.nNodes
    M = problem_instance.bound
    kappa = problem_instance.nZeros
    total_cuts = len(cut_manager.cut_storage)

    # defining variables
    alpha = model.addMVar(shape = N, lb = -GRB.INFINITY)
    x = model.addMVar(shape = n, lb = -GRB.INFINITY)
    delta = model.addMVar(shape = n, vtype = GRB.BINARY)

    obj = ones((problem_instance.nNodes, 1)).T @ alpha
    model.setObjective(obj, GRB.MINIMIZE)

    # introducing linear cuts
    i = 0

    for cut in cut_manager.cut_storage:

        if i == N - 1:
            i = 0
        else:
            i += 1

        if problem_instance.soc:
            model.addConstr(alpha[i] >= cut['fx'] + cut['gx'].T @ x - cut['gx'].T @ cut['x'] + cut['eig'] / 2 * (
                    x @ x - 2 * cut['x'].T @ x + cut['x'].T @ cut['x']), name = f"cut['cut_id']")
        else:
            model.addConstr(alpha[i] >= cut['fx'] + cut['gx'].T @ x - cut['gx'].T @ cut['x'], name = f"cut['cut_id']")

    if problem_instance.problem_instance.constr is not None:
        if problem_instance.soc:
            for const_cut in cut_manager.const_cut_storage:
                model.addConstr(
                    0 >= const_cut['gx'] + const_cut['ggx'].T @ x - const_cut['ggx'].T @ const_cut['x'] + const_cut[
                        'eig_g'] / 2 * (
                            x @ x - 2 * const_cut['x'].T @ x + const_cut['x'].T @ const_cut['x']), name = 'gcut')
        else:
            for const_cut in cut_manager.const_cut_storage:
                model.addConstr(0 >= const_cut['gx'] + const_cut['ggx'].T @ x - const_cut['ggx'].T @ const_cut['x'],
                                name = 'gcut')

    total_cuts = len(cut_manager.cut_storage) + len(cut_manager.const_cut_storage)
    print(f'Total Optimality Cuts: {total_cuts}\n'
          f'Objective cuts: {len(cut_manager.cut_storage)}\n'
          f'Constraints Cuts: {len(cut_manager.const_cut_storage)}')

    for i in range(n):
        model.addConstr(x[i] <= M * delta[i], name = f'b1{i}')
        model.addConstr(-M * delta[i] <= x[i], name = f'b2{i}')

    model.addConstr(delta.sum() <= kappa, name = 'd')
    model.setParam('OutputFlag', 0)
    model.setParam('MIPGap', 1e-4)
    model.optimize()
    # model.printStats()
    return model.objval, delta.x.reshape(n, 1), x.x.reshape(n, 1)
