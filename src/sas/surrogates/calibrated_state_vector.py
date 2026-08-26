
import numpy as np
from random import shuffle
from scipy.stats import spearmanr
from typing import List, Tuple, Dict
import warnings

from src.sas.types import Circuit, H, S, T, CX
from .interface import ISurrogate
from src.sas.utils import simulate_state_vector, vector_distance, unitary_distance, \
    simulate_unitary

CALIBRATED_STATE_VECTOR_SURROGATE = "calibrated_state_vector"


class CalibratedStateVectorSurrogate(ISurrogate):
    type = CALIBRATED_STATE_VECTOR_SURROGATE

    target: np.ndarray
    base_state: int
    base_state_correlation: float

    def __init__(self, target: np.ndarray):
        self.target = target
        self.base_state = None
        self.base_state_correlation = -1

    def train(self, circuits: List[Circuit]) -> None:
        if self.base_state is not None:
            return

        assert len(circuits) > 0

        unitaries = [
            simulate_unitary(circuit) for circuit in circuits
        ]

        fitness_scores = [
            unitary_distance(unitary, self.target) for unitary in unitaries
        ]

        for base_state in range(2 ** circuits[0].qubit_num):
            base_state_fitness_scores = [
                vector_distance(unitary.T[base_state],
                                self.target.T[base_state])
                for unitary in unitaries
            ]

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                base_state_correlation = spearmanr(
                    base_state_fitness_scores, fitness_scores).statistic

            if base_state_correlation > self.base_state_correlation:
                self.base_state_correlation = base_state_correlation
                self.base_state = base_state

    def predict(self, circuits: List[Circuit]) -> List[float]:

        fitness_scores = []

        for circuit in circuits:
            state_vector = simulate_state_vector(
                circuit, base_state=self.base_state)
            distance = vector_distance(
                state_vector, self.target.T[self.base_state])
            fitness_scores.append(distance)

        return fitness_scores

    @property
    def params(self) -> Dict:
        return {
            "type": self.type,
            "base_state": self.base_state,
            "base_state_correlation": self.base_state_correlation
        }
