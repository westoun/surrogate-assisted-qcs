
import numpy as np
from random import shuffle
from typing import List, Tuple

from src.sas.types import Circuit, H, S, T, CX
from .interface import ISurrogate
from src.sas.utils import simulate_state_vector, vector_distance

STATE_VECTOR_SURROGATE = "state_vector"


class StateVectorSurrogate(ISurrogate):
    target_vector: np.ndarray

    def __init__(self, target: np.ndarray):
        self.target_vector = target.T[0]

    def train(self, circuits: List[Circuit], epochs: int = 200) -> None:
        pass  # Do nothing.

    def predict(self, circuits: List[Circuit]) -> List[float]:

        fitness_scores = []

        for circuit in circuits:
            state_vector = simulate_state_vector(circuit)
            distance = vector_distance(state_vector, self.target_vector)
            fitness_scores.append(distance)

        return fitness_scores
