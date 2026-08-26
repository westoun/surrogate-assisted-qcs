
from dataclasses import dataclass
import numpy as np

from src.sas.surrogates import ISurrogate


@dataclass
class EAParams:
    target: np.ndarray
    parent_count: int
    offspring_count: int
    max_evaluations: int
    evaluate_every: int
    qubit_num: int
    gate_count: int
    logging_prefix: str
    seed_population_size: int = None
