import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from .circuit import Circuit
from .gates import H, S, T, CX

def simulate_unitary(circuit: Circuit) -> np.ndarray:
    qiskit_circuit = circuit_to_qiskit(circuit)
    unitary = Operator(qiskit_circuit).data
    return unitary

def circuit_to_qiskit(circuit: Circuit) -> QuantumCircuit:
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
            raise NotImplementedError(f"No mapping found for gate type '{type(gate)}'")

    return qiskit_circuit
     