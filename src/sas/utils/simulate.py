import numpy as np
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator, Statevector

from src.sas.types import Circuit, H, S, T, CX


def simulate_unitary(circuit: Circuit) -> np.ndarray:
    qiskit_circuit = circuit_to_qiskit(circuit)
    unitary = Operator(qiskit_circuit).data
    return unitary


def circuit_to_qiskit(circuit: Circuit, add_measurement: bool = False) -> QuantumCircuit:
    qiskit_circuit = QuantumCircuit(circuit.qubit_num)

    for gate in circuit.gates:
        if type(gate) == H:
            qiskit_circuit.h(gate.target)
        elif type(gate) == S:
            qiskit_circuit.s(gate.target)
        elif type(gate) == T:
            qiskit_circuit.t(gate.target)
        elif type(gate) == CX:
            qiskit_circuit.cx(gate.control, gate.target)
        else:
            raise NotImplementedError(
                f"No mapping found for gate type '{type(gate)}'")

    if add_measurement:
        qiskit_circuit.measure_all()

    return qiskit_circuit


def simulate_state_vector(circuit: Circuit) -> np.ndarray:
    qiskit_circuit = circuit_to_qiskit(circuit)
    vector = Statevector.from_circuit(qiskit_circuit)
    return vector
