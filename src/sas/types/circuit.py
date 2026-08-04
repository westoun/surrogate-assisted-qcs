from copy import deepcopy
from typing import List, Union

from .gates import Gate


class Circuit():

    qubit_num: int
    gates: List[Gate]
    fitness: Union[float, None]

    def __init__(self, qubit_num: int):
        self.qubit_num = qubit_num
        self.gates = []
        self.fitness = None

    def copy(self) -> "Circuit":
        my_copy = deepcopy(self)
        my_copy.fitness = None
        return my_copy

    def __repr__(self) -> str:
        return "[" + ", ".join([str(g) for g in self.gates]) + "]"
