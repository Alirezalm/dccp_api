from numpy import ones, sort, where

from dccp.rhadmm.rhadmm import rhadmm


def sfp(problem_instance, rank, comm, mpi_class):
    if rank == 0:
        print("Performing SFP step...\n")
    n = problem_instance.nVars
    binvar = ones((n, 1))
    x, fx, gx = rhadmm(problem_instance, bin_var = binvar, comm = comm, mpi_class = mpi_class)
    largest_nonzeros = sort(x, axis = 0)[::-1][0:problem_instance.nZeros]
    for index, item in enumerate(x):
        if item in largest_nonzeros:
            x[index] = 1
        else:
            x[index] = 0

    if rank == 0:
        print("SFP done!\n\n")
    return x
