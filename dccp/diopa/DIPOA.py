"""
Main loop of the DIPOA Algorithm
"""
from numpy import zeros
from numpy.linalg import eig

from dccp.diopa.cut_store_gen import CutStoreGen
from dccp.diopa.heuristics import sfp
from dccp.masters.master_problem import solve_master
from dccp.rhadmm.rhadmm import rhadmm


def dipoa(problem_instance, comm, mpi_class):
    rank = comm.Get_rank()
    size = comm.Get_size()
    max_iter = 100
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
    for k in range(max_iter):
        x, fx, gx = rhadmm(problem_instance, bin_var = binvar, comm = comm,
                           mpi_class = mpi_class)  # solves primal problem
        if problem_instance.soc:
            min_eig = min(eig(problem_instance.problem_instance.compute_hess_at(x))[0])

        ub = comm.reduce(fx, op = mpi_class.SUM, root = 0)

        if rank == 0:
            upper_bound = min(ub, upper_bound)
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

            lower_bound, binvar = solve_master(problem_instance, cut_manager)
            data_memory['ub'].append(upper_bound)
            data_memory['lb'].append(lower_bound)
            data_memory['iter'].append(k)

        rel_gap = comm.bcast((upper_bound - lower_bound) / upper_bound, root = 0)
        if rank == 0:
            print(f"lb: {lower_bound}, ub:{upper_bound} gap: {rel_gap}")
        if rel_gap <= eps:
            break

    data_memory['x'] = [item[0] for item in x]
    data_memory['obj'] = lower_bound
    data_memory['gap'] = (upper_bound - lower_bound) / upper_bound

    return data_memory
