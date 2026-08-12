
import math
import numpy as np
import os
import pickle
import random
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, \
    mean_squared_error
import sys
sys.path.append(os.path.abspath('../..'))  # nopep8
import torch
from tqdm import tqdm
from typing import List

from src.sas.utils import random_circuit, simulate_unitary, unitary_distance, \
    MultiMemoryRecorder, MultiTimeRecorder

from src.sas.types import Circuit
from src.sas.surrogates import CircuitFeatureSurrogate, CIRCUIT_FEATURE_SURROGATE, \
    GNNSurrogate, GNN_SURROGATE
from logging_ import log_model_performance


def generate_circuits(qubit_num: int, gate_count: int, count: int, max_tries: int = 100_000) -> List[Circuit]:
    encountered_circuits = set()

    circuits = []

    for _ in range(max_tries):
        circuit = random_circuit(
            qubit_num, gate_count)

        if circuit not in encountered_circuits:
            circuits.append(circuit)
            encountered_circuits.add(circuit)

        if len(circuits) == count:
            break
    else:
        print(
            f"Could not create {count} unique circuits within {max_tries} tries.")
        print(f"Returning {len(circuits)} circuits instead.")

    return circuits


def generate_data(qubit_num: int, gate_count: int, count: int) -> List[Circuit]:
    circuits = generate_circuits(qubit_num, gate_count, count)

    target_circuit = circuits[0]
    target = simulate_unitary(target_circuit)
    target_circuit.fitness = 0.0

    for circuit in circuits[1:]:
        unitary = simulate_unitary(circuit)
        fitness = unitary_distance(unitary, target)
        circuit.fitness = fitness

    return circuits


if __name__ == "__main__":
    seed_num: int = 15
    seed_offset: int = 50 + 15

    circuit_count = 1000

    qcs_setups = [
        (4, 20),
        (6, 15),
        (6, 20)
    ]

    layer_counts = [1, 2, 3, 4, 5]

    time_recorder = MultiTimeRecorder()
    memory_recorder = MultiMemoryRecorder()

    for seed_i in tqdm(range(seed_num)):
        seed = seed_i + seed_offset

        for qubit_num, gate_count in qcs_setups:

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            circuits = generate_data(
                qubit_num, gate_count, count=circuit_count
            )
            fitness_scores = [
                circuit.fitness for circuit in circuits
            ]

            X_train, X_test = circuits[:math.floor(
                0.7 * len(circuits))], circuits[math.floor(0.7 * len(circuits)):]
            y_train, y_test = fitness_scores[:math.floor(
                0.7 * len(fitness_scores))], fitness_scores[math.floor(0.7 * len(fitness_scores)):]

            for model in [CIRCUIT_FEATURE_SURROGATE, GNN_SURROGATE]:

                if model == GNN_SURROGATE:
                    element_counts = [8, 16, 32, 64, 128, 256]
                elif model == CIRCUIT_FEATURE_SURROGATE:
                    element_counts = [16, 32, 64, 128, 256, 512, 1024]
                else:
                    raise NotImplementedError(
                        f"No implementation provided for model type '{model}'")

                for layer_count in layer_counts:
                    for element_count in element_counts:

                        if model == CIRCUIT_FEATURE_SURROGATE:
                            surrogate = CircuitFeatureSurrogate(
                                qubit_num=qubit_num,
                                neuron_counts=[
                                    element_count for _ in range(layer_count)
                                ],
                                max_epochs=10_000
                            )
                        elif model == GNN_SURROGATE:
                            surrogate = GNNSurrogate(
                                channel_counts=[
                                    element_count for _ in range(layer_count)
                                ],
                                max_epochs=10_000
                            )
                        else:
                            raise NotImplementedError(
                                f"No implementation provided for model type '{model}'")

                        with memory_recorder["train"]:
                            with time_recorder["train"]:
                                surrogate.train(X_train)

                        with memory_recorder["inference"]:
                            with time_recorder["inference"]:
                                y_pred = surrogate.predict(X_test)

                        mse = mean_squared_error(y_pred, y_test)
                        rank_correlation = spearmanr(y_pred, y_test).statistic

                        log_model_performance(
                            qubit_num=qubit_num,
                            gate_count=gate_count,
                            model=surrogate.type,
                            layer_count=layer_count,
                            elements_per_layer=element_count,
                            seed=seed,
                            train_time=time_recorder["train"].duration,
                            inference_time=time_recorder["inference"].duration,
                            train_memory=memory_recorder["train"].peak,
                            inference_memory=memory_recorder["inference"].peak,
                            mse_score=mse,
                            rank_correlation=rank_correlation
                        )
