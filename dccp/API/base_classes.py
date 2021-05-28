from abc import ABC, abstractmethod

from numpy import ndarray


class BaseObjectiveFunction(ABC):

    @abstractmethod
    def fx(self, x: ndarray) -> float:
        """
        fx computes local objective values
        :param x: vector of variables
        :return: f(x)
        """

    @abstractmethod
    def gfx(self, x: ndarray) -> ndarray:
        """
        gfx computes local objectives gradient
        :param x: vector of variables
        :return: gradient of fx
        """
        pass

    @abstractmethod
    def hfx(self, x: ndarray) -> ndarray:
        """
        gfx computes local objectives Hessian
        :param x: vector of variables
        :return: Hessian of fx
        """
        pass


class NLPConstr(BaseObjectiveFunction):
    pass


class Constraint(ABC):
    pass


class Problem(ABC):
    pass
