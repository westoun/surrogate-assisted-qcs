
from dataclasses import dataclass
import numpy as np
from typing import Any, List

from .evaluator import IEvaluator
from src.sas.utils.random_ import random_circuit


@dataclass
class EAParams:
    target: np.ndarray
    evaluator: IEvaluator
    parent_count: int
    offspring_count: int
    max_generations: int
    qubit_num: int
    gate_count: int


class EvolutionaryAlgorithm():

    params: EAParams

    def __init__(self, params: EAParams):
        self.params = params

    def run(self) -> None:
        population = self.init_population(
            count=self.params.parent_count + self.params.offspring_count)

        for generation in range(self.params.max_generations):

            fitness_scores = self.evaluate(population)

            # TODO: Add logging.

            # check stopping criterion

            parents = self.select(population, fitness_scores,
                                  count=self.params.parent_count)

            offspring = self.mutate(parents, count=self.params.offspring_count)

            population = parents + offspring

    def init_population(self, count: int) -> List[Any]:
        population = [
            random_circuit(self.params.qubit_num, self.params.gate_count)
            for _ in range(count)
        ]
        return population

    def evaluate(self, circuits: List[Any]) -> List[float]:
        # TODO: Adjust circuit typing once decided.
        fitness_scores = [
            self.params.evaluator.evaluate(circuit) for circuit in circuits
        ]
        return fitness_scores

    def select(self, circuits: List[Any], fitness_scores: List[float], count: int) -> List[Any]:
        # TODO: Adjust circuit typing once decided.

        annotated_circuits: List = zip(circuits, fitness_scores)
        annotated_circuits.sort(key=lambda item: item[1])

        selected_annotated_circuits = annotated_circuits[:count]

        selected_circuits = [
            item[0] for item in selected_annotated_circuits
        ]

        return selected_circuits

    def mutate(self, circuits: List[Any], count: int) -> List[Any]:
        # TODO: Adjust circuit typing once decided.
        raise NotImplementedError()
