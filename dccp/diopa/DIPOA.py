"""
Main loop of the DIPOA Algorithm
"""
from numpy import zeros
from numpy.linalg import eig, norm
from numpy.ma import ones

from dccp.diopa.cut_store_gen import CutStoreGen
from dccp.diopa.heuristics import sfp
from dccp.masters.master_problem import solve_master
from dccp.rhadmm.rhadmm import rhadmm


def dipoa(problem_instance, comm, mpi_class):
    rank = comm.Get_rank()
    size = comm.Get_size()
    max_iter = 10
    n = problem_instance.nVars
    binvar = zeros((problem_instance.nVars, 1))  # initial binary
    if problem_instance.sfp:
        binvar = sfp(problem_instance, rank, comm, mpi_class)
    cut_manager = CutStoreGen()
    rcv_x = None  # related to MPI gather
    rcv_gx = None
    rcv_eig = None
    upper_bound = 1e8
    lower_bound = -upper_bound
    eps = 0.05  # 5%
    data_memory = {
        'x': None,
        'lb': [],
        'ub': [],
        'iter': []
    }
    x = zeros((n, 1))
    min_eig = 0
    problem_instance.sfp = False
    min_const_eig = 0

    if (rank == 0) & problem_instance.soc & (problem_instance.problem_instance.constr is not None):
        print('COMPUTING EIGENVALUES ...')
        min_const_eig = min(eig(problem_instance.problem_instance.compute_const_hess_at(index = 0))[0])

    if problem_instance.name == 'dsqcqp':
        min_eig = min(eig(problem_instance.problem_instance.compute_hess_at())[0])

    if rank == 0:
        print("DIPOA STARTS...\n")
    for k in range(max_iter):
        x, fx, gx = rhadmm(problem_instance, bin_var = binvar, comm = comm,
                           mpi_class = mpi_class)  # solves primal problem

        if problem_instance.soc & (problem_instance.name == 'dslr'):
            print('COMPUTING EIGENVALUES ...')
            min_eig = min(eig(problem_instance.problem_instance.compute_hess_at(x))[0])

        ub = comm.reduce(fx, op = mpi_class.SUM, root = 0)

        if rank == 0:
            upper_bound = min(ub, upper_bound)
            # upper_bound = ub
            rcv_x = zeros((size, n))
            rcv_gx = zeros((size, n))

        rcv_fx = comm.gather(fx, root = 0)
        if problem_instance.soc:
            rcv_eig = comm.gather(min_eig, root = 0)
        comm.Gather([x, mpi_class.DOUBLE], rcv_x, root = 0)
        comm.Gather([gx, mpi_class.DOUBLE], rcv_gx, root = 0)

        if rank == 0:
            for node in range(size):
                if problem_instance.soc:
                    cut_manager.store_cut(k, node, rcv_x[node, :].reshape(n, 1), rcv_fx[node],
                                          rcv_gx[node, :].reshape(n, 1), rcv_eig[node])
                else:
                    cut_manager.store_cut(k, node, rcv_x[node, :].reshape(n, 1), rcv_fx[node],
                                          rcv_gx[node, :].reshape(n, 1))
            if problem_instance.problem_instance.constr is not None:
                gx = problem_instance.problem_instance.compute_const_at(0, x)
                ggx = problem_instance.problem_instance.compute_const_grad_at(0, x)
                if problem_instance.soc:
                    cut_manager.store_const_cut(x, gx, ggx, min_const_eig)
                else:
                    cut_manager.store_const_cut(x, gx, ggx)

            lower_bound, binvar = solve_master(problem_instance, cut_manager)
            data_memory['ub'].append(upper_bound)
            data_memory['lb'].append(lower_bound)
            data_memory['iter'].append(k)

        rel_gap = comm.bcast((upper_bound - lower_bound) / abs(upper_bound + 1e-8), root = 0)
        if rank == 0:
            print(f"k: {k} lb: {lower_bound}, ub:{upper_bound} gap: {rel_gap}")
        if rel_gap <= eps:
            break
    if rank == 0:
        print("CONVERGED. STORING THE DATA AND PACKING SOLUTION\n")
    data_memory['x'] = [item[0] for item in x]
    data_memory['obj'] = lower_bound
    data_memory['gap'] = (upper_bound - lower_bound) / abs(upper_bound + 1e-8)

    return data_memory
