
import numpy as np
from scipy.stats import wasserstein_distance
from typing import List, Tuple, Dict

from src.sas.types import Circuit, H, S, T, CX
from .interface import ISurrogate
from src.sas.utils import get_shot_distribution

SHOT_DISTANCE_SURROGATE = "shot_distance"


def state_to_distribution(state: np.ndarray) -> List:
    target_distribution = []
    for cell in state:
        target_distribution.append(abs(cell) ** 2)
    return target_distribution


class ShotDistanceSurrogate(ISurrogate):
    type = SHOT_DISTANCE_SURROGATE

    target_distribution: List
    shots: int

    def __init__(self, target: np.ndarray, shots: int = 10_000):
        target_state = target.T[0]
        self.target_distribution = state_to_distribution(target_state)

        self.shots = shots

    def train(self, circuits: List[Circuit]) -> None:
        pass  # Do nothing.

    def predict(self, circuits: List[Circuit]) -> List[float]:
        fitness_scores = []

        for circuit in circuits:
            shot_distribution = get_shot_distribution(
                circuit, shots=self.shots)
            distance = wasserstein_distance(
                shot_distribution, self.target_distribution)
            fitness_scores.append(distance)

        return fitness_scores

    @property
    def params(self) -> Dict:
        return {
            "type": self.type,
            "shots": self.shots,
        }
