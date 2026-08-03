from random import choice, randint, sample, choices
from typing import Any

from .circuit import Circuit
from .gates import Gate, H, S, T, CX


def random_gate(qubit_num: int, uniform_configuration_choice: bool = True) -> Gate:
    if uniform_configuration_choice:
        return _random_gate_uniform_by_config(qubit_num)
    else:
        return _random_gate_uniform_by_gatetype(qubit_num)


def _random_gate_uniform_by_config(qubit_num: int) -> Gate:
    GateTypes = [
        H, S, T, CX
    ]
    weights = [
        qubit_num, qubit_num, qubit_num, qubit_num * (qubit_num - 1)
    ]

    GateType = choices(population=GateTypes, weights=weights, k=1)[0]

    if GateType == CX:
        control, target = sample(range(0, qubit_num), 2)
        return CX(control=control, target=target)
    else:
        target = randint(0, qubit_num - 1)
        return GateType(target)


def _random_gate_uniform_by_gatetype(qubit_num: int) -> Gate:
    GateTypes = [
        H, S, T, CX
    ]

    GateType = choice(GateTypes)

    if GateType == CX:
        control, target = sample(range(0, qubit_num), 2)
        return CX(control=control, target=target)
    else:
        target = randint(0, qubit_num - 1)
        return GateType(target)


def random_circuit(qubit_num: int, gate_count: int) -> Circuit:
    circuit = Circuit(qubit_num)

    for _ in range(gate_count):
        circuit.gates.append(
            random_gate(qubit_num)
        )

    return circuit
