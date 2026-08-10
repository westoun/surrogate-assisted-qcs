
import numpy as np
from random import shuffle, random
from typing import List, Tuple

from src.sas.types import Circuit, H, S, T, CX
from .interface import ISurrogate
from src.sas.utils import simulate_state_vector, vector_distance

RANDOM_FITNESS_SURROGATE = "random_fitness"


class RandomFitnessSurrogate(ISurrogate):
    type = RANDOM_FITNESS_SURROGATE

    def train(self, circuits: List[Circuit]) -> None:
        pass  # Do nothing.

    def predict(self, circuits: List[Circuit]) -> List[float]:

        fitness_scores = []

        for circuit in circuits:
            # Assign random fitness between 0.0 and 1.0
            fitness = random()
            fitness_scores.append(fitness)

        return fitness_scores
