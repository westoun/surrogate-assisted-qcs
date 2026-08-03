from dataclasses import dataclass
from typing import List

H_GATE = "H"
S_GATE = "S"
T_GATE = "T"
CX_GATE = "CX"


class Gate():
    name: str
    target: int

    def __init__(self, target: int):
        self.target = target

    def __repr__(self):
        return self.name + "(" + ", ".join([str(q) for q in self.qubits]) + ")"

    @property
    def qubits(self) -> List[int]:
        return [self.target]


class H(Gate):
    name = H_GATE


class S(Gate):
    name = S_GATE


class T(Gate):
    name = T_GATE


class CX(Gate):
    name: str
    control: int
    target: int

    def __init__(self, control: int, target: int):
        self.name = CX_GATE
        self.control = control
        self.target = target

    @property
    def qubits(self) -> List[int]:
        return [self.control, self.target]
