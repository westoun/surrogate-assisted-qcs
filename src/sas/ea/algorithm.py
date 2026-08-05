
from dataclasses import dataclass
import numpy as np
from random import sample, randint, choices
from typing import Any, List

from sas.types.circuit import Circuit
from src.sas.utils import simulate_unitary, unitary_distance, \
    random_circuit, random_gate, TimeRecorder
from src.sas.surrogates import ISurrogate

from .params import EAParams
from .logging_ import log_epoch_results


class EvolutionaryAlgorithm():

    params: EAParams
    surrogate: ISurrogate

    def __init__(self, params: EAParams, surrogate: ISurrogate = None):
        self.params = params
        self.surrogate = surrogate

    def run(self) -> None:
        timer = TimeRecorder()

        with timer:
            population = self.init_population(
                count=self.params.parent_count + self.params.offspring_count)

            self.evaluate(population)

            if self.surrogate is not None:
                self.surrogate.train(population, epochs=500)

        log_epoch_results(
            generation=0, population=population, duration=timer.duration, params=self.params)

        for generation in range(1, self.params.max_generations + 1):

            with timer:
                parents = self.select(
                    population, count=self.params.parent_count)

                if self.surrogate is None:
                    offspring = self.mutate(
                        parents, count=self.params.offspring_count)
                else:
                    offspring = self.mutate(
                        parents, count=self.params.offspring_count * 3)
                    self.surrogate.evaluate(offspring)
                    offspring = self.select(
                        offspring, count=self.params.offspring_count)

                population = parents + offspring

                self.evaluate(population)
                if self.surrogate is not None:
                    self.surrogate.train(population)

            log_epoch_results(
                generation=generation, population=population, duration=timer.duration, params=self.params)

    def init_population(self, count: int) -> List[Circuit]:
        population = [
            random_circuit(self.params.qubit_num, self.params.gate_count)
            for _ in range(count)
        ]
        return population

    def evaluate(self, circuits: List[Circuit]) -> None:
        for circuit in circuits:
            if circuit.simulated:
                continue

            unitary = simulate_unitary(circuit)
            distance = unitary_distance(unitary, self.params.target)
            circuit.fitness = distance
            circuit.simulated = True

    def select(self, circuits: List[Circuit], count: int) -> List[Circuit]:
        sorted_circuits = sorted(circuits, key=lambda circuit: circuit.fitness)
        selected_circuits = sorted_circuits[:count]
        return selected_circuits

    def mutate(self, circuits: List[Circuit], count: int) -> List[Circuit]:
        selected_parents = choices(circuits, k=count)

        offspring = []
        for circuit in selected_parents:
            child = circuit.copy()

            gate_i = randint(0, len(child.gates) - 1)
            child.gates[gate_i] = random_gate(child.qubit_num)

            offspring.append(child)

        return offspring
