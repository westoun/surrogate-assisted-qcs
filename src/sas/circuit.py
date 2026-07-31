from copy import deepcopy
from typing import List, Union


class Circuit():

    qubit_num: int
    gates: List[str]
    fitness: Union[float, None]

    def __init__(self, qubit_num: int):
        self.qubit_num = qubit_num
        self.gates = []
        self.fitness = None

    def copy(self) -> "Circuit":
        my_copy = deepcopy(self)
        my_copy.fitness = None
        return my_copy
