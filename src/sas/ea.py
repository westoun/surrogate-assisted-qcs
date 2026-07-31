
from dataclasses import dataclass
import numpy as np
from random import sample, randint
from typing import Any, List

from .evaluator import IEvaluator
from src.sas.utils.random_ import random_circuit, random_gate
from src.sas.circuit import Circuit


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

            self.evaluate(population)

            # TODO: Add logging.

            # check stopping criterion

            parents = self.select(population, count=self.params.parent_count)

            offspring = self.mutate(parents, count=self.params.offspring_count)

            population = parents + offspring

    def init_population(self, count: int) -> List[Circuit]:
        population = [
            random_circuit(self.params.qubit_num, self.params.gate_count)
            for _ in range(count)
        ]
        return population

    def evaluate(self, circuits: List[Circuit]) -> None:
        for circuit in circuits:
            circuit.fitness = self.params.evaluator.evaluate(circuit)

    def select(self, circuits: List[Circuit], count: int) -> List[Circuit]:
        sorted_circuits = sorted(circuits, key=lambda circuit: circuit.fitness)
        selected_circuits = sorted_circuits[:count]
        return selected_circuits

    def mutate(self, circuits: List[Circuit], count: int) -> List[Circuit]:
        selected_parents = sample(circuits, k=count)

        offspring = []
        for circuit in selected_parents:
            child = circuit.copy()

            gate_i = randint(0, len(child.gates) - 1)

            child.gates[gate_i] = random_gate(child.qubit_num)

        return offspring
