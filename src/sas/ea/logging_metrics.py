
from dataclasses import dataclass
import numpy as np
from random import sample, randint, choices, choice
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, \
    mean_squared_error
from statistics import mean, median, stdev
from typing import Any, List, Tuple
import warnings

from sas.types.circuit import Circuit
from src.sas.utils import simulate_unitary, unitary_distance, \
    random_circuit, random_gate, TimeRecorder, MultiTimeRecorder, \
    MemoryRecorder, MultiMemoryRecorder
from src.sas.surrogates import ISurrogate

from .params import EAParams
from .logging_ import log_epoch_results


def evaluate_surrogate(predicted_fitness_scores: List[float], actual_fitness_scores: List[float]) -> Tuple[float, float]:
    """Returns fitness mse and rank correlation"""

    # Case: no surrogate was used, so no fitness scores have
    # been computed prior to explicit evaluation.
    if None in predicted_fitness_scores:
        return 0.0, 1.0

    fitness_mse = mean_squared_error(
        predicted_fitness_scores, actual_fitness_scores)

    # Spearmanr is not defined if one of the input arrays has
    # a stdev of 0. In that case, returns nan.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        rank_correlation = spearmanr(
            predicted_fitness_scores, actual_fitness_scores).statistic

    return fitness_mse, rank_correlation


def extract_fitness_statistics(population: List[Circuit]) -> Tuple[float, float, float, float]:
    """Returns the minimum, median, mean, and std of the 
    fitness scores in the population."""

    fitness_scores = [
        circuit.fitness for circuit in population
    ]

    return min(fitness_scores), median(fitness_scores), mean(fitness_scores), stdev(fitness_scores)
