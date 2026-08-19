
from dataclasses import dataclass
import numpy as np
from random import sample, randint, choices, choice
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, \
    mean_squared_error
from statistics import mean, median, stdev
from typing import Any, List, Tuple, Dict
import warnings

from sas.types.circuit import Circuit
from src.sas.utils import simulate_unitary, unitary_distance, \
    random_circuit, random_gate, TimeRecorder, MultiTimeRecorder, \
    MemoryRecorder, MultiMemoryRecorder
from src.sas.surrogates import ISurrogate

from .params import EAParams
from .logging_ import log_epoch_results


def compute_survival_rate(new_parents: List[Circuit], prev_offspring: List[Circuit]) -> float:
    if prev_offspring is None:
        return None

    surviving_offspring = [
        circuit for circuit in prev_offspring if circuit in new_parents
    ]
    return len(surviving_offspring) / len(prev_offspring)


def compute_population_diversity(circuits: List[Circuit]) -> float:
    """Returns the normed all-possible-pairs diversity as described in
    'The Underlying Similarity of Diversity Measures Used in Evolutionary Computation'"""

    if len(circuits) < 2:
        return 0.0

    gene_count = len(circuits[0].gates)

    qubit_num = circuits[0].qubit_num
    unique_gate_configs = qubit_num ** 2 + 2 * qubit_num

    gene_value_frequencies = _build_gene_value_frequency_dict(circuits)

    diversity = 0.0
    for gate_i in range(gene_count):
        for gene in gene_value_frequencies[gate_i]:
            diversity += gene_value_frequencies[gate_i][gene] * \
                (1 - gene_value_frequencies[gate_i][gene])

    l = gene_count
    a = unique_gate_configs
    n = len(circuits)
    r = n % a
    diversity *= a / (l * ((a - 1) - r * (a - r) / n ** 2))

    return diversity


def _build_gene_value_frequency_dict(circuits: List[Circuit]) -> Dict:
    gene_count = len(circuits[0].gates)

    gene_value_frequencies = {}

    for gene_i in range(gene_count):
        gene_value_frequencies[gene_i] = {}

    for circuit in circuits:

        for gene_i, gate in enumerate(circuit.gates):
            gene = str(gate)

            if gene in gene_value_frequencies[gene_i]:
                gene_value_frequencies[gene_i][gene] += 1 / len(circuits)
            else:
                gene_value_frequencies[gene_i][gene] = 1 / len(circuits)

    return gene_value_frequencies


def evaluate_surrogate(true_fitness_scores: List[float], surrogate_fitness_scores: List[float]) -> Tuple[float, float]:
    """Returns fitness mse and rank correlation"""

    # Case: no surrogate was used, so no fitness scores have
    # been computed prior to explicit evaluation.
    if None in surrogate_fitness_scores:
        return 0.0, 1.0

    fitness_mse = mean_squared_error(
        true_fitness_scores, surrogate_fitness_scores)

    # Spearmanr is not defined if one of the input arrays has
    # a stdev of 0. In that case, returns nan.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        rank_correlation = spearmanr(
            true_fitness_scores, surrogate_fitness_scores).statistic

    return fitness_mse, rank_correlation
