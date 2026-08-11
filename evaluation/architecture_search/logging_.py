
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
        mse_score: float,
        rank_correlation: float
) -> None:
    target_path = "results.csv"

    add_header = not os.path.exists(target_path)

    with open(target_path, "a") as target_file:

        if add_header:
            header = "qubit_num; gate_count; layer_count; neuron_count; seed; mse_score; rank_correlation"
            target_file.write(header + "\n")

        line = f"{qubit_num}; {gate_count}; {layer_count}; {neuron_count}; {seed}; {mse_score}; {rank_correlation}"
        target_file.write(line + "\n")
