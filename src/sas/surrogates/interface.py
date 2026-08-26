
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from sas.types.circuit import Circuit


class ISurrogate(ABC):

    @abstractmethod
    def train(self, circuits: List[Circuit]) -> None:
        ...

    @abstractmethod
    def predict(self, circuits: List[Circuit]) -> List[float]:
        ...

    def evaluate(self, circuits: List[Circuit]) -> None:
        circuits_to_evaluate = [
            circuit for circuit in circuits if circuit.surrogate_fitness is None
        ]

        if len(circuits_to_evaluate) == 0:
            return

        fitness_scores = self.predict(circuits_to_evaluate)

        for circuit, fitness in zip(circuits_to_evaluate, fitness_scores):
            circuit.surrogate_fitness = fitness

    @property
    @abstractmethod
    def type(self) -> str:
        ...

    @property
    def params(self) -> Dict:
        return {
            "type": self.type
        }
