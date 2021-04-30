from time import time

from numpy import zeros, eye
from numpy.linalg import eig
from numpy.random import randn, rand
from sklearn import preprocessing
from numpy.random import seed
from dccp.diopa.DIPOA import dipoa
from dccp.problem.problem_classes import LogRegProb, QuadConsProb
from dccp.rhadmm.rhadmm import rhadmm

PROBLEM_CLASS = {
    'distributedSparseLogisticRegression': 'dslr',
    'distributedSparseQCQP': 'dsqcqp'
}


class Problem(object):
    def __init__(self, problem_data: dict):
        self.name = problem_data['name']
        self.nVars = int(problem_data['nVars'])
        if self.name == "dslr":
            self.nSamples = int(problem_data['nSamples'])
        self.nZeros = int(problem_data['nZeros'])
        self.compareTo = problem_data['compareTo']
        self.nNodes = int(problem_data['nNodes'])
        self.sfp = bool(problem_data['sfp'])
        self.soc = bool(problem_data['soc'])
        self.problem_instance = None
        self.bound = None

    # should be run before solve
    def create_random_problem_instance(self, bound):
        # seed(0)  # just for test/
        if self.name == PROBLEM_CLASS['distributedSparseLogisticRegression']:
            dataset = preprocessing.normalize(randn(self.nSamples, self.nVars), norm = 'l2')
            response = randn(self.nSamples, 1)
            response[response >= 0.5] = 1
            response[response < 0.5] = 0
            self.problem_instance = LogRegProb(local_dataset = dataset, local_response = response)
            self.bound = bound
            return self
        elif self.name == PROBLEM_CLASS['distributedSparseQCQP']:

            self.problem_instance = QuadConsProb(problem_data = problem_data)
            self.bound = bound
            return self
        else:
            raise ValueError(f'problem class {self.name} is not supported yet')

    def solve(self, comm, mpi_class):  # starts dipoa algorithm
        start_time = time()
        solution_data = dipoa(self, comm, mpi_class)
        elapsed_time = time() - start_time
        solution_data['elapsed_time'] = elapsed_time
        return solution_data
