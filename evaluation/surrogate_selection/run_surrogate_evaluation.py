
import click
from datetime import datetime
import numpy as np
import os
import pickle
import random
import sys
sys.path.append(os.path.abspath('../..'))  # nopep8
from typing import List, Tuple
from uuid import uuid4

from src.sas.surrogates.interface import ISurrogate
from src.sas.utils.random_ import random_circuit
from sas.types.circuit import Circuit
from src.sas.utils.graph import circuit_to_dag, graph_to_hash
from src.sas.utils.fitness import unitary_distance
from src.sas.utils.simulate import simulate_unitary

from logging_ import log_dataset_details
from src.sas.utils.time import TimeRecorder


def remove_duplicates(circuits: List[Circuit]) -> List[Circuit]:
    unique_circuits = []

    encountered_hashes = set()

    for circuit in circuits:
        dag = circuit_to_dag(circuit)
        dag_hash = graph_to_hash(dag)

        if dag_hash not in encountered_hashes:
            unique_circuits.append(circuit)
            encountered_hashes.add(dag_hash)

    return unique_circuits


def load_or_generate_data(qubit_num: int, gate_count: int, circuit_count: int, seed: int) -> Tuple[np.ndarray, List[Circuit]]:
    data_path = f"circuits_{qubit_num}qn{gate_count}gc{circuit_count}cc{seed}s.pkl"

    # Keep loading and storing in same function to ensure format compatibility (OCP).
    if os.path.exists(data_path):
        with open(data_path, "rb") as data_file:
            data = pickle.load(data_file)
            return data["target"], data["circuits"]

    else:

        recorder = TimeRecorder()
        with recorder:

            circuits = [random_circuit(
                qubit_num=qubit_num, gate_count=gate_count
            ) for _ in range(circuit_count)]

            circuits = remove_duplicates(circuits)

            target_circuit = circuits[0]
            target = simulate_unitary(target_circuit)
            target_circuit.fitness = 0.0

            for circuit in circuits[1:]:
                unitary = simulate_unitary(circuit)
                fitness = unitary_distance(unitary, target)
                circuit.fitness = fitness

        log_dataset_details(
            circuits=circuits,
            qubit_num=qubit_num,
            gate_count=gate_count,
            circuit_count=circuit_count,
            seed=seed,
            duration=recorder.duration,
            data_path=data_path
        )

        with open(data_path, "wb") as data_file:
            pickle.dump({
                "target": target,
                "circuits": circuits
            }, data_file)

        return target, circuits


@click.command()
@click.option(
    "--model",
    "-m",
    type=click.STRING,
    help="The surrogate model to be used."  # TODO: Add available values.
)
@click.option(
    "--qubit-num",
    "-qn",
    type=click.INT,
    default=6,
    help="The number of qubits per circuit. Default is 6.",
)
@click.option(
    "--gate-count",
    "-gc",
    type=click.INT,
    default=20,
    help="The number of gates per circuit. Default is 20.",
)
@click.option(
    "--circuit-count",
    "-cc",
    type=click.INT,
    default=10_000,
    help="The number of gates per circuit. Default is 10_000.",
)
@click.option(
    "--seed",
    "-s",
    type=click.INT,
    default=0,
    help="The seed value used for pythons random module.",
)
@click.option(
    "--tag",
    "-t",
    type=click.STRING,
    default=None,
    help="An optional tag that is logged alongside the experiment config for later identification.",
)
def run_experiment(model: str, qubit_num: int, gate_count: int, circuit_count: int, seed: int, tag: str):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    target, circuits = load_or_generate_data(
        qubit_num, gate_count, circuit_count, seed)

    # train model


if __name__ == "__main__":
    run_experiment()
