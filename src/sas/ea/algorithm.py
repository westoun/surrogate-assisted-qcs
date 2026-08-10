
from dataclasses import dataclass
import numpy as np
from random import sample, randint, choices
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, \
    mean_squared_error
from typing import Any, List, Tuple
import warnings

from sas.types.circuit import Circuit
from src.sas.utils import simulate_unitary, unitary_distance, \
    random_circuit, random_gate, TimeRecorder, MultiTimeRecorder
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
        recorder = MultiTimeRecorder()

        with recorder["ea"]:
            population = self.init_population(
                count=self.params.parent_count + self.params.offspring_count)

        with recorder["eval"]:
            self.evaluate(population)

        with recorder["train"]:
            if self.surrogate is not None:
                self.surrogate.train(population, epochs=500)

        log_epoch_results(
            generation=0, population=population,
            ea_duration=recorder["ea"].duration,
            eval_duration=recorder["eval"].duration,
            train_duration=recorder["train"].duration,
            pred_duration=recorder["pred"].duration,
            fitness_mse=None, rank_correlation=None,
            params=self.params)

        for generation in range(1, self.params.max_generations + 1):

            if self.surrogate is None:
                with recorder["ea"]:
                    parents = self.select(
                        population, count=self.params.parent_count)
                    offspring = self.mutate(
                        parents, count=self.params.offspring_count)

            else:

                with recorder["ea"]:
                    parents = self.select(
                        population, count=self.params.parent_count)
                    offspring = self.mutate(
                        parents, count=self.params.offspring_count * 3)

                with recorder["pred"]:
                    self.surrogate.evaluate(offspring)

                with recorder["ea"]:
                    offspring = self.select(
                        offspring, count=self.params.offspring_count)

            predicted_fitness_scores = [
                circuit.fitness for circuit in offspring
            ]

            with recorder["eval"]:
                self.evaluate(offspring)

            actual_fitness_scores = [
                circuit.fitness for circuit in offspring
            ]

            fitness_mse = mean_squared_error(
                predicted_fitness_scores, actual_fitness_scores)

            # Spearmanr is not defined if one of the input arrays has
            # a stdev of 0. In that case, returns nan.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                rank_correlation = spearmanr(
                    predicted_fitness_scores, actual_fitness_scores).statistic

            population = parents + offspring

            with recorder["train"]:
                if self.surrogate is not None:
                    self.surrogate.train(population, epochs=100)

            log_epoch_results(
                generation=generation, population=population,
                ea_duration=recorder["ea"].duration,
                eval_duration=recorder["eval"].duration,
                train_duration=recorder["train"].duration,
                pred_duration=recorder["pred"].duration,
                fitness_mse=fitness_mse,
                rank_correlation=rank_correlation,
                params=self.params)

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
