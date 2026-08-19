from copy import deepcopy
from typing import List, Union

from .gates import Gate


class Circuit():

    qubit_num: int
    gates: List[Gate]
    true_fitness: Union[float, None]
    surrogate_fitness: Union[float, None]

    def __init__(self, qubit_num: int):
        self.qubit_num = qubit_num
        self.gates = []
        self.true_fitness = None
        self.surrogate_fitness = None

    def copy(self) -> "Circuit":
        my_copy = deepcopy(self)
        my_copy.true_fitness = None
        my_copy.surrogate_fitness = None
        return my_copy

    def __repr__(self) -> str:
        return "[" + ", ".join([str(g) for g in self.gates]) + "]"

    def __hash__(self):
        gates = ";".join([str(gate) for gate in self.gates])
        return hash(gates)

    def __eq__(self, value: "Circuit") -> bool:
        return self.__hash__() == value.__hash__()
