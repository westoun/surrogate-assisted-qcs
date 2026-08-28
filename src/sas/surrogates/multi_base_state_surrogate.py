
import numpy as np
from random import shuffle, randint, sample
from statistics import mean
from typing import List, Tuple, Dict

from src.sas.types import Circuit, H, S, T, CX
from .interface import ISurrogate
from src.sas.utils import simulate_state_vector, vector_distance

MULTI_BASE_STATE_SURROGATE = "multi_base_state_vector"


class MultiBaseStateSurrogate(ISurrogate):
    type = MULTI_BASE_STATE_SURROGATE

    target: np.ndarray
    n: int

    def __init__(self, target: np.ndarray, n: int = 2):
        self.target = target
        self.n = n

    def train(self, circuits: List[Circuit]) -> None:
        pass  # Do nothing.

    def predict(self, circuits: List[Circuit]) -> List[float]:
        if len(circuits) == 0:
            return []

        base_states = sample(range(2 ** circuits[0].qubit_num), k=self.n)

        fitness_scores = []

        for circuit in circuits:
            distances = []

            for base_state in base_states:

                target_vector = self.target.T[base_state]

                state_vector = simulate_state_vector(
                    circuit, base_state=base_state)

                distance = vector_distance(state_vector, target_vector)
                distances.append(distance)

            fitness_scores.append(mean(distances))

        return fitness_scores

    @property
    def params(self) -> Dict:
        return {
            "type": self.type,
            "n": self.n
        }
