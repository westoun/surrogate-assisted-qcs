
import os
from typing import List

from src.sas.types import Circuit
from .params import EAParams


def log_epoch_results(generation: int,
                      fitness_min: float,
                      ea_duration: float, eval_duration: float, train_duration: float, pred_duration: float,
                      ea_memory: float, eval_memory: float, train_memory: float, pred_memory: float,
                      fitness_mse: float, rank_correlation: float, selection_overlap: float,
                      survival_rate: float,
                      population_diversity: float,
                      explicit_evaluations: int,
                      params: EAParams) -> None:
    target_path = params.logging_prefix + "_results.csv"

    add_header = not os.path.exists(target_path)

    with open(target_path, "a") as target_file:

        if add_header:
            header = "generation; fitness_best; "
            header += "ea_duration; eval_duration; train_duration; pred_duration; total_duration; "
            header += "ea_memory; eval_memory; train_memory; pred_memory; max_memory; "
            header += "fitness_mse; rank_correlation; survival_rate; selection_overlap; "
            header += "population_diversity; "
            header += "explicit_evaluations"
            target_file.write(header + "\n")

        total_duration = ea_duration + eval_duration + train_duration + pred_duration

        max_memory = 0.0
        if ea_memory is not None and ea_memory > max_memory:
            max_memory = ea_memory
        if eval_memory is not None and eval_memory > max_memory:
            max_memory = eval_memory
        if train_memory is not None and train_memory > max_memory:
            max_memory = train_memory
        if pred_memory is not None and pred_memory > max_memory:
            max_memory = pred_memory

        line = f"{generation}; {fitness_min}; "
        line += f"{ea_duration}; {eval_duration}; {train_duration}; {pred_duration}; {total_duration}; "
        line += f"{ea_memory}; {eval_memory}; {train_memory}; {pred_memory}; {max_memory}; "
        line += f"{fitness_mse}; {rank_correlation}; {survival_rate}; {selection_overlap}; "
        line += f"{population_diversity}; "
        line += f"{explicit_evaluations}"
        target_file.write(line + "\n")
