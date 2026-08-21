
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
from .logging_metrics import evaluate_surrogate, \
    compute_survival_rate, compute_population_diversity, \
    compute_selection_overlap


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

        true_fitness_scores = [
            circuit.true_fitness for circuit in population
        ]

        fitness_min = min(true_fitness_scores)

        with memory_recorder["train"]:
            with time_recorder["train"]:
                if self.surrogate is not None:
                    self.surrogate.train(population)

        population_diversity = compute_population_diversity(population)

        log_epoch_results(
            generation=0,
            fitness_min=fitness_min,
            ea_duration=time_recorder["ea"].duration,
            eval_duration=time_recorder["eval"].duration,
            train_duration=time_recorder["train"].duration,
            pred_duration=time_recorder["pred"].duration,
            ea_memory=memory_recorder["ea"].peak,
            eval_memory=memory_recorder["eval"].peak,
            train_memory=memory_recorder["train"].peak,
            pred_memory=memory_recorder["pred"].peak,
            fitness_mse=None, rank_correlation=None,
            survival_rate=None, selection_overlap=None,
            population_diversity=population_diversity,
            explicit_evaluations=len(population),
            params=self.params)

        # Init variable outside of loop to ensure it is available
        # for survival rate computation.
        offspring = None
        explicit_evaluations = len(population)

        for generation in range(1, 20_000):  # High number for max generations

            with memory_recorder["ea"]:
                with time_recorder["ea"]:

                    if self.surrogate is None or (generation - 1) % self.params.evaluate_every == 0:
                        parents = self.select(
                            population, count=self.params.parent_count, use_surrogate=False)
                    else:
                        parents = self.select(
                            population, count=self.params.parent_count, use_surrogate=True)

                    survival_rate = compute_survival_rate(
                        new_parents=parents, prev_offspring=offspring)

                    offspring = self.mutate(
                        parents, count=self.params.offspring_count)

            population = parents + offspring

            fitness_mse, rank_correlation, selection_overlap = None, None, None

            if self.surrogate is None:

                explicit_evaluations += len(offspring)

                with memory_recorder["eval"]:
                    with time_recorder["eval"]:
                        self.evaluate(offspring)

                fitness_min = min(min([
                    circuit.true_fitness for circuit in population
                ]), fitness_min)

            elif generation % self.params.evaluate_every == 0:

                circuits_to_evaluate = [
                    circuit for circuit in population if circuit.true_fitness is None
                ]

                explicit_evaluations += len(circuits_to_evaluate)

                with memory_recorder["eval"]:
                    with time_recorder["eval"]:
                        self.evaluate(population)

                # Do not time log here, as predictions are only used for surrogate
                # evaluation.
                surrogate_fitness_scores_of_new_circuits = self.surrogate.predict(
                    circuits_to_evaluate)
                true_fitness_scores_of_new_circuits = [
                    circuit.true_fitness for circuit in circuits_to_evaluate
                ]

                fitness_mse, rank_correlation = evaluate_surrogate(
                    true_fitness_scores=true_fitness_scores_of_new_circuits,
                    surrogate_fitness_scores=surrogate_fitness_scores_of_new_circuits
                )

                true_fitness_scores = [
                    circuit.true_fitness for circuit in population
                ]

                # Don't log time here, as this step is only carried out to evaluate
                # the surrogate and does not affect the search itself.
                surrogate_fitness_scores = self.surrogate.predict(population)

                selection_overlap = compute_selection_overlap(
                    true_fitness_scores, surrogate_fitness_scores, count=self.params.parent_count)

                fitness_min = min(min(true_fitness_scores), fitness_min)

                with memory_recorder["train"]:
                    with time_recorder["train"]:
                        self.surrogate.train(population)

            else:

                with memory_recorder["pred"]:
                    with time_recorder["pred"]:
                        self.surrogate.evaluate(population)

            population_diversity = compute_population_diversity(population)

            log_epoch_results(
                generation=generation,
                fitness_min=fitness_min,
                ea_duration=time_recorder["ea"].duration,
                eval_duration=time_recorder["eval"].duration,
                train_duration=time_recorder["train"].duration,
                pred_duration=time_recorder["pred"].duration,
                ea_memory=memory_recorder["ea"].peak,
                eval_memory=memory_recorder["eval"].peak,
                train_memory=memory_recorder["train"].peak,
                pred_memory=memory_recorder["pred"].peak,
                fitness_mse=fitness_mse, rank_correlation=rank_correlation,
                selection_overlap=selection_overlap,
                survival_rate=survival_rate,
                population_diversity=population_diversity,
                explicit_evaluations=explicit_evaluations,
                params=self.params)

            if explicit_evaluations >= self.params.max_evaluations:
                break

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
            if circuit.true_fitness is not None:
                continue

            unitary = simulate_unitary(circuit)
            distance = unitary_distance(unitary, self.params.target)
            circuit.true_fitness = distance

    def select(self, circuits: List[Circuit], count: int, use_surrogate=False) -> List[Circuit]:
        if use_surrogate:
            sorted_circuits = sorted(
                circuits, key=lambda circuit: circuit.surrogate_fitness)
        else:
            sorted_circuits = sorted(
                circuits, key=lambda circuit: circuit.true_fitness)

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
