
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
from src.sas.surrogates import ISurrogate, GateFrequencySurrogate, \
    GATE_FREQUENCY_SURROGATE

from logging_ import log_experiment_details, log_end_timestamp


@click.command()
@click.option(
    "--model",
    "-m",
    type=click.STRING,
    default=None,
    help=("The surrogate model to be used. Default is None."
          f"Allowed: None, '{GATE_FREQUENCY_SURROGATE}'."
          )
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
def run_experiment(model: str, qubit_num: int, gate_count: int, seed: int, tag: str):
    parent_count = 10
    offspring_count = 10
    max_generations = 5000
    logging_prefix = f"results/{qubit_num}qn{gate_count}gc{model}m{seed}s"

    log_experiment_details(
        model=model,
        qubit_num=qubit_num,
        gate_count=gate_count,
        seed=seed,
        tag=tag,
        parent_count=parent_count,
        offspring_count=offspring_count,
        max_generations=max_generations,
        logging_prefix=logging_prefix
    )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create synthesis target using simulator
    target_circuit = random_circuit(qubit_num, gate_count)
    target = simulate_unitary(target_circuit)

    params = EAParams(
        target=target,
        parent_count=parent_count,
        offspring_count=offspring_count,
        max_generations=max_generations,
        qubit_num=qubit_num,
        gate_count=gate_count,
        logging_prefix=logging_prefix
    )

    if model is None:
        surrogate = None
    elif model == GATE_FREQUENCY_SURROGATE:
        surrogate: ISurrogate = GateFrequencySurrogate(qubit_num)
    else:
        raise NotImplementedError(
            f"No implementation found for surrogate model '{model}'.")

    # TODO: Log experiment parameters

    ea = EvolutionaryAlgorithm(params, surrogate)
    ea.run()

    log_end_timestamp(logging_prefix=logging_prefix)


if __name__ == "__main__":
    run_experiment()
