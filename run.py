from mpi4py import MPI

from dccp.problem.problem import Problem



def run(data):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    problem = Problem(problem_data = data).create_random_problem_instance(0.16)
    solution_data = problem.solve(comm, MPI)




