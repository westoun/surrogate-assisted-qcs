
import click
import numpy as np
import os
import random
import sys
sys.path.append(os.path.abspath('../..'))  # nopep8
import torch
from uuid import uuid4

from src.sas.ea import EvolutionaryAlgorithm, EAParams
from src.sas.utils import random_circuit, simulate_unitary
from src.sas.surrogates import ISurrogate, CircuitFeatureSurrogate, \
    CIRCUIT_FEATURE_SURROGATE, StateVectorSurrogate, STATE_VECTOR_SURROGATE, \
    ShotDistanceSurrogate, SHOT_DISTANCE_SURROGATE, RandomFitnessSurrogate, \
    RANDOM_FITNESS_SURROGATE, GNNSurrogate, GNN_SURROGATE, \
    RANDOM_BASE_STATE_SURROGATE, RandomBaseStateSurrogate

from logging_ import log_experiment_details, log_end_timestamp


@click.command()
@click.option(
    "--model",
    "-m",
    type=click.STRING,
    default=None,
    help=("The surrogate model to be used. Default is None. "
          f"Allowed: None, 'None', '{CIRCUIT_FEATURE_SURROGATE}', '{STATE_VECTOR_SURROGATE}', "
          f"'{SHOT_DISTANCE_SURROGATE}', '{RANDOM_FITNESS_SURROGATE}', '{GNN_SURROGATE}', "
          f"'{RANDOM_BASE_STATE_SURROGATE}'."
          )
)
@click.option(
    "--evaluate_every",
    "-ee",
    type=click.INT,
    default=5,
    help="The interval at which explicit evaluation shall take place. Default is 5.",
)
@click.option(
    "--qubit-num",
    "-qn",
    type=click.INT,
    default=6,
    help="The number of qubits per circuit. Default is 6.",
)
@click.option(
    "--gate-count",
    "-gc",
    type=click.INT,
    default=20,
    help="The number of gates per circuit. Default is 20.",
)
@click.option(
    "--seed",
    "-s",
    type=click.INT,
    default=0,
    help="The seed value used for pythons random module.",
)
@click.option(
    "--tag",
    "-t",
    type=click.STRING,
    default=None,
    help="An optional tag that is logged alongside the experiment config for later identification.",
)
def run_experiment(model: str, evaluate_every: int, qubit_num: int, gate_count: int, seed: int, tag: str):
    parent_count = 100
    offspring_count = 100
    max_generations = 500
    logging_prefix = f"results/{qubit_num}qn{gate_count}gc{model}m{seed}s_{str(uuid4())}"

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create synthesis target using simulator
    target_circuit = random_circuit(qubit_num, gate_count)
    target = simulate_unitary(target_circuit)

    ea_params = EAParams(
        target=target,
        parent_count=parent_count,
        offspring_count=offspring_count,
        max_generations=max_generations,
        qubit_num=qubit_num,
        gate_count=gate_count,
        logging_prefix=logging_prefix,
        evaluate_every=evaluate_every
    )

    if model is None or model == "None":
        surrogate = None
    elif model == CIRCUIT_FEATURE_SURROGATE:
        surrogate: ISurrogate = CircuitFeatureSurrogate(qubit_num, neuron_counts=[512, 512], max_epochs=10_000)
    elif model == STATE_VECTOR_SURROGATE:
        surrogate: ISurrogate = StateVectorSurrogate(target=target)
    elif model == RANDOM_BASE_STATE_SURROGATE:
        surrogate: ISurrogate = RandomBaseStateSurrogate(target=target)
    elif model == SHOT_DISTANCE_SURROGATE:
        surrogate: ISurrogate = ShotDistanceSurrogate(target=target, shots=1000)
    elif model == RANDOM_FITNESS_SURROGATE:
        surrogate: ISurrogate = RandomFitnessSurrogate()
    elif model == GNN_SURROGATE:
        surrogate: ISurrogate = GNNSurrogate(channel_counts=[128, 128], max_epochs=10_000)
    else:
        raise NotImplementedError(
            f"No implementation found for surrogate model '{model}'.")

    if surrogate is None:
        surrogate_params = {}
    else:
        surrogate_params = surrogate.params

    log_experiment_details(
        ea_params=ea_params,
        surrogate_params=surrogate_params,
        seed=seed,
        tag=tag,
        logging_prefix=logging_prefix
    )

    ea = EvolutionaryAlgorithm(ea_params, surrogate)
    ea.run()

    log_end_timestamp(logging_prefix=logging_prefix)


if __name__ == "__main__":
    run_experiment()
