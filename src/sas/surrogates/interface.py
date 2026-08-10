
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
        fitness_scores = self.predict(circuits)

        for circuit, fitness in zip(circuits, fitness_scores):
            circuit.fitness = fitness

    @property
    @abstractmethod
    def type(self) -> str:
        ...

    @property
    def params(self) -> Dict:
        return {
            "type": self.type
        }
