import click
from numpy import ndarray

from dccp.API.base_classes import BaseObjectiveFunction
from dccp.exceptions.dccp_errors import NotEnoughInfo

NUM_FUNC = 3


class CardModel(object):
    def __init__(self, obj_list: list, constr_list: list = None):
        self.obj_list = obj_list
        self.constr_list = constr_list
        self.validated = False
        self._validation()

    def _validation(self):
        if len(self.obj_list) <= NUM_FUNC:
            raise NotEnoughInfo('make sure that fx, gfx, and hfx are provided.')
        elif self.constr_list:
            for index, constr in enumerate(self.constr_list):
                if len(constr) <= NUM_FUNC:
                    raise NotEnoughInfo(f'make sure that fx, gfx, and hfx for constraint {index} are provided.')
        else:
            click.echo(click.style('problem data validated', fg = 'green'))
            self.validated = True

    class ObjectiveFunction(BaseObjectiveFunction):

        def __init__(self, obj_data: dict, constr: dict = None):
            self.obj_data = obj_data
            self.constr = constr

        def fx(self, x: ndarray) -> float:
            return self.obj_data['fx'](x)

        def gfx(self, x: ndarray) -> ndarray:
            return self.obj_data['gfx'](x)

        def hfx(self, x: ndarray) -> ndarray:
            return self.obj_data['hfx'](x)
