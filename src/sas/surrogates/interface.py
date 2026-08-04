
from abc import ABC, abstractmethod
from typing import List, Tuple

from sas.types.circuit import Circuit


class ISurrogate(ABC):

    @abstractmethod
    def train(self, circuits: List[Circuit]) -> None:
        ...

    @abstractmethod
    def predict(self, circuit: Circuit) -> Tuple[float, float]:
        ...
