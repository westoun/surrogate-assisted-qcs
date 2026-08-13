
import numpy as np
from random import shuffle, randint
from typing import List, Tuple

from src.sas.types import Circuit, H, S, T, CX
from .interface import ISurrogate
from src.sas.utils import simulate_state_vector, vector_distance

RANDOM_BASE_STATE_SURROGATE = "random_base_state_vector"


class RandomBaseStateSurrogate(ISurrogate):
    type = RANDOM_BASE_STATE_SURROGATE

    target: np.ndarray

    def __init__(self, target: np.ndarray):
        self.target = target

    def train(self, circuits: List[Circuit]) -> None:
        pass  # Do nothing.

    def predict(self, circuits: List[Circuit]) -> List[float]:
        if len(circuits) == 0:
            return []

        base_state = randint(0, 2 ** circuits[0].qubit_num - 1)
        target_vector = self.target.T[base_state]

        fitness_scores = []

        for circuit in circuits:
            state_vector = simulate_state_vector(
                circuit, base_state=base_state)

            distance = vector_distance(state_vector, target_vector)
            fitness_scores.append(distance)

        return fitness_scores
