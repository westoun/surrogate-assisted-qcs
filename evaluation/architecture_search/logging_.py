
import os
from typing import List

from sas.types.circuit import Circuit
from src.sas.utils.os import save_to_json
from src.sas.utils.time import get_timestamp


def log_model_performance(
        qubit_num: int,
        gate_count: int,
        layer_count: int,
        neuron_count: int,
        seed: int,
        train_time: float,
        inference_time: float,
        train_memory: float,
        inference_memory: float,
        mse_score: float,
        rank_correlation: float
) -> None:
    target_path = "results/architecture_search.csv"

    add_header = not os.path.exists(target_path)

    with open(target_path, "a") as target_file:

        if add_header:
            header = "qubit_num; gate_count; layer_count; neuron_count; seed; "
            header += "train_time; inference_time; train_memory; inference_memory; mse_score; rank_correlation"
            target_file.write(header + "\n")

        line = f"{qubit_num}; {gate_count}; {layer_count}; {neuron_count}; {seed}; "
        line += f"{train_time}; {inference_time}; {train_memory}; {inference_memory}; {mse_score}; {rank_correlation}"
        target_file.write(line + "\n")
