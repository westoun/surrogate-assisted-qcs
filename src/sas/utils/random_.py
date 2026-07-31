from typing import Any
from src.sas.circuit import Circuit


def random_gate(qubit_num: int, uniform_configuration_choice: bool = True) -> Any:
    if uniform_configuration_choice:
        return _random_gate_uniform_by_config(qubit_num)
    else:
        return _random_gate_uniform_by_gatetype(qubit_num)


def _random_gate_uniform_by_config(qubit_num: int) -> Any:
    # TODO: Adjust circuit typing once decided.
    raise NotImplementedError()


def _random_gate_uniform_by_gatetype(qubit_num: int) -> Any:
    # TODO: Adjust circuit typing once decided.
    raise NotImplementedError()


def random_circuit(qubit_num: int, gate_count: int) -> Circuit:
    circuit = Circuit(qubit_num)

    for _ in range(gate_count):
        circuit.gates.append(
            random_gate(qubit_num)
        )

    return circuit
