from numpy import ones

from dccp.rhadmm.rhadmm import rhadmm


def sfp(problem_instance, rank, comm, mpi_class):
    if rank == 0:
        print("Performing SFP step...\n")
    n = problem_instance.nVars
    binvar = ones((n, 1))
    x, fx, gx = rhadmm(problem_instance, bin_var = binvar, comm = comm, mpi_class = mpi_class)
    if rank == 0:
        print("SFP done!\n\n")
    return x


