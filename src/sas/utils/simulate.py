import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator, Statevector
from typing import List

from src.sas.types import Circuit, H, S, T, CX

simulator: AerSimulator = None


def simulate_unitary(circuit: Circuit) -> np.ndarray:
    qiskit_circuit = circuit_to_qiskit(circuit)
    unitary = Operator(qiskit_circuit).data
    return unitary


def circuit_to_qiskit(circuit: Circuit, add_measurement: bool = False, base_state: int = None) -> QuantumCircuit:
    qiskit_circuit = QuantumCircuit(circuit.qubit_num)

    if base_state is not None:
        assert base_state < 2 ** circuit.qubit_num

        bit_string = bin(base_state)[2:].zfill(circuit.qubit_num)
        for bit_i, bit_value in enumerate(bit_string):

            if bit_value == "1":
                qiskit_circuit.x(bit_i)

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


def simulate_state_vector(circuit: Circuit, base_state: int = None) -> np.ndarray:
    qiskit_circuit = circuit_to_qiskit(circuit, base_state=base_state)
    vector = Statevector.from_circuit(qiskit_circuit)
    return vector


def get_shot_distribution(circuit: Circuit, shots: int = 10_000, seed: int = 0) -> List[float]:
    global simulator
    if simulator is None:
        simulator = AerSimulator(seed_simulator=seed)

    qiskit_circuit = circuit_to_qiskit(circuit, add_measurement=True)

    job = simulator.run(qiskit_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(qiskit_circuit)

    distribution = np.zeros(
        shape=(2 ** circuit.qubit_num, ), dtype=float
    ).tolist()

    for solution, count in counts.items():
        # Reverse measurement bit order as qiskit seems to index bottom up.
        solution = solution[::-1]

        base_state = int(solution, 2)

        distribution[base_state] = count / shots

    return distribution
