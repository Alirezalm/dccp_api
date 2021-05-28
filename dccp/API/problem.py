from numpy import ndarray

from dccp.API.base_classes import BaseObjectiveFunction


class ObjectiveFunction(BaseObjectiveFunction):

    def __init__(self, obj_data: dict, constr: dict = None):
        self.obj_data = obj_data
        self.constr = constr

    def fx(self, x: ndarray) -> float:
        pass

    def gfx(self, x: ndarray) -> ndarray:
        pass

    def hfx(self, x: ndarray) -> ndarray:
        pass
