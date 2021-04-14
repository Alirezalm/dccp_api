import json
import sys

from flask import json
from mpi4py import MPI

from dccp.problem.problem import Problem

with open('config.json') as jsonfile:
    my_data = json.load(jsonfile)


def run(data):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    problem = Problem(problem_data = data).create_random_problem_instance(1)
    solution_data = problem.solve(comm, MPI)

    if rank == 0:
        with open('solution.json', 'w') as json_ans:
            json.dump(solution_data, json_ans)
        return solution_data


if __name__ == '__main__':
    run(my_data)
