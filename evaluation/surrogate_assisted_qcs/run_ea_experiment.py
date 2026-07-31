
import click
import numpy as np
import random
from uuid import uuid4

from src.sas.ea import EvolutionaryAlgorithm, EAParams
from src.sas.evaluator import IEvaluator
from src.sas.utils.random_ import random_circuit


@click.command()
@click.option(
    "--evaluation-strategy",
    "-es",
    type=click.STRING,
    default="exact",
    help="The evaluation strategy to be used."  # TODO: Add available values.
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
def run_experiment(evaluation_strategy: str, qubit_num: int, gate_count: int, seed: int, tag: str):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create synthesis target using simulator

    # Setup Evaluation Strategy

    # Setup EA parameters

    # Setup EA

    # Run EA


if __name__ == "__main__":
    run_experiment()
