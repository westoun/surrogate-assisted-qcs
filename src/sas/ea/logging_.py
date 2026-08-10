
import os
from statistics import mean, median, stdev
from typing import List

from src.sas.types import Circuit
from .params import EAParams


def log_epoch_results(generation: int, population: List[Circuit], ea_duration: float,
                      eval_duration: float, train_duration: float, pred_duration: float,
                      fitness_mse: float, rank_correlation: float,
                      params: EAParams) -> None:
    target_path = params.logging_prefix + "_results.csv"

    add_header = not os.path.exists(target_path)

    with open(target_path, "a") as target_file:

        if add_header:
            header = "generation; fitness_best; fitness_median; fitness_mean; fitness_stdev; "
            header += "ea_duration; eval_duration; train_duration; pred_duration; "
            header += "fitness_mse; rank_correlation"
            target_file.write(header + "\n")

        fitness_scores = [
            circuit.fitness for circuit in population
        ]

        line = f"{generation}; {min(fitness_scores)}; {median(fitness_scores)}; {mean(fitness_scores)}; {stdev(fitness_scores)}; "
        line += f"{ea_duration}; {eval_duration}; {train_duration}; {pred_duration}; "
        line += f"{fitness_mse}; {rank_correlation}"
        target_file.write(line + "\n")
