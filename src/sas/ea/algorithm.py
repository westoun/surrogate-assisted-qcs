
from dataclasses import dataclass
import numpy as np
from random import sample, randint, choices, choice
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, \
    mean_squared_error
from typing import Any, List, Tuple
import warnings

from sas.types.circuit import Circuit
from src.sas.utils import simulate_unitary, unitary_distance, \
    random_circuit, random_gate, TimeRecorder, MultiTimeRecorder, \
    MemoryRecorder, MultiMemoryRecorder
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
        time_recorder = MultiTimeRecorder()
        memory_recorder = MultiMemoryRecorder()

        with memory_recorder["ea"]:
            with time_recorder["ea"]:
                population = self.init_population(
                    count=self.params.parent_count + self.params.offspring_count)

        with memory_recorder["eval"]:
            with time_recorder["eval"]:
                self.evaluate(population)

        with memory_recorder["train"]:
            with time_recorder["train"]:
                if self.surrogate is not None:
                    self.surrogate.train(population)

        log_epoch_results(
            generation=0, population=population,
            ea_duration=time_recorder["ea"].duration,
            eval_duration=time_recorder["eval"].duration,
            train_duration=time_recorder["train"].duration,
            pred_duration=time_recorder["pred"].duration,
            ea_memory=memory_recorder["ea"].peak,
            eval_memory=memory_recorder["eval"].peak,
            train_memory=memory_recorder["train"].peak,
            pred_memory=memory_recorder["pred"].peak,
            fitness_mse=None, rank_correlation=None,
            params=self.params)

        for generation in range(1, self.params.max_generations + 1):

            if self.surrogate is None:

                with memory_recorder["ea"]:
                    with time_recorder["ea"]:
                        parents = self.select(
                            population, count=self.params.parent_count)
                        offspring = self.mutate(
                            parents, count=self.params.offspring_count)

            else:

                with memory_recorder["ea"]:
                    with time_recorder["ea"]:
                        parents = self.select(
                            population, count=self.params.parent_count)
                        offspring = self.mutate(
                            parents, count=self.params.offspring_count * 3)

                with memory_recorder["pred"]:
                    with time_recorder["pred"]:
                        self.surrogate.evaluate(offspring)

                with memory_recorder["ea"]:
                    with time_recorder["ea"]:
                        offspring = self.select(
                            offspring, count=self.params.offspring_count)

            predicted_fitness_scores = [
                circuit.fitness for circuit in offspring
            ]

            with memory_recorder["eval"]:
                with time_recorder["eval"]:
                    self.evaluate(offspring)

            actual_fitness_scores = [
                circuit.fitness for circuit in offspring
            ]

            fitness_mse, rank_correlation = self.evaluate_surrogate(
                predicted_fitness_scores, actual_fitness_scores)

            population = parents + offspring

            with memory_recorder["train"]:
                with time_recorder["train"]:
                    if self.surrogate is not None:
                        self.surrogate.train(population)

            log_epoch_results(
                generation=generation, population=population,
                ea_duration=time_recorder["ea"].duration,
                eval_duration=time_recorder["eval"].duration,
                train_duration=time_recorder["train"].duration,
                pred_duration=time_recorder["pred"].duration,
                ea_memory=memory_recorder["ea"].peak,
                eval_memory=memory_recorder["eval"].peak,
                train_memory=memory_recorder["train"].peak,
                pred_memory=memory_recorder["pred"].peak,
                fitness_mse=fitness_mse,
                rank_correlation=rank_correlation,
                params=self.params)

    def evaluate_surrogate(self, predicted_fitness_scores: List[float], actual_fitness_scores: List[float]) -> Tuple[float, float]:
        # Case: no surrogate was used, so no fitness scores have
        # been computed prior to explicit evaluation.
        if None in predicted_fitness_scores:
            return 0.0, 1.0

        fitness_mse = mean_squared_error(
            predicted_fitness_scores, actual_fitness_scores)

        # Spearmanr is not defined if one of the input arrays has
        # a stdev of 0. In that case, returns nan.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            rank_correlation = spearmanr(
                predicted_fitness_scores, actual_fitness_scores).statistic

        return fitness_mse, rank_correlation

    def init_population(self, count: int, max_tries: int = 100_000) -> List[Circuit]:
        encountered_circuits = set()

        population = []

        for _ in range(max_tries):
            circuit = random_circuit(
                self.params.qubit_num, self.params.gate_count)

            if circuit not in encountered_circuits:
                population.append(circuit)
                encountered_circuits.add(circuit)

            if len(population) == count:
                break
        else:
            print(
                f"Could not create {count} unique circuits within {max_tries} tries.")
            print(f"Returning {len(population)} circuits instead.")

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

    def mutate(self, circuits: List[Circuit], count: int, max_tries: int = 100_000) -> List[Circuit]:
        encountered_circuits = set()
        for circuit in circuits:
            encountered_circuits.add(circuit)

        offspring = []

        for _ in range(max_tries):
            parent = choice(circuits)
            child = parent.copy()

            gate_i = randint(0, len(child.gates) - 1)
            child.gates[gate_i] = random_gate(child.qubit_num)

            if child not in encountered_circuits:
                offspring.append(child)
                encountered_circuits.add(child)

            if len(offspring) == count:
                break

        else:
            print(
                f"Could not create {count} unique circuits within {max_tries} tries.")
            print(f"Returning {len(offspring)} children instead.")

        return offspring
