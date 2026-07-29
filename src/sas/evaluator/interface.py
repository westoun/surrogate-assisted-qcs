from abc import ABC, abstractmethod
import numpy as np


class IEvaluator(ABC):
    target: np.ndarray

    def __init__(self, target: np.ndarray):
        self.target = target

    @abstractmethod
    def evaluate(self, circuit) -> float:
        ...
