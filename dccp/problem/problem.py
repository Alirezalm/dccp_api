from time import time

from numpy import zeros
from numpy.random import randn, rand
from sklearn import preprocessing

from dccp.diopa.DIPOA import dipoa
from dccp.problem.problem_classes import LogRegProb
from dccp.rhadmm.rhadmm import rhadmm

PROBLEM_CLASS = {
    'dslr': 'distributed sparse logistic regression',
    'dspo': 'distributed sparse portfolio optimization',
}


class Problem(object):
    def __init__(self, problem_data: dict):
        self.name = problem_data['name']
        self.nVars = int(problem_data['nVars'])
        self.nSamples = int(problem_data['nSamples'])
        self.nZeros = int(problem_data['nZeros'])
        self.compareTo = problem_data['compareTo']
        self.nNodes = int(problem_data['nNodes'])
        self.sfp = problem_data['sfp']
        self.problem_instance = None
        self.bound = None

    # should be run before solve
    def create_random_problem_instance(self, bound):
        if self.name == PROBLEM_CLASS['dslr']:
            dataset = preprocessing.normalize(randn(self.nSamples, self.nVars), norm = 'l2')
            response = randn(self.nSamples, 1)
            response[response >= 0.5] = 1
            response[response < 0.5] = 0
            self.problem_instance = LogRegProb(local_dataset=dataset, local_response=response)
            self.bound = bound
            return self

    def solve(self, comm, mpi_class):  # starts dipoa algorithm
        start_time = time()
        solution_data = dipoa(self, comm, mpi_class)
        elapsed_time = time() - start_time
        solution_data['elapsed_time'] = elapsed_time
        return solution_data
