# Surrogate-assisted Unitary Synthesis

This repository contains the source code to the paper
_"State Vector-based Surrogates for Unitary Synthesis with Evolutionary Algorithms"_
by Stein, Patakov, Dalvi, Klikovits, and Wimmer from the [Institute of Business Informatics - Software Engineering](https://se.jku.at/)
at the [Johannes Kepler University]
(https://www.jku.at/en), Linz.
The `src/` directory contains the implementation of the surrogate models presented in the
paper as well as the evolutionary algorithm used during their evaluation.
The `evaluation/` directory contains the code used to produce the evaluation results
presented in the paper.

## Installation

We recommend to use the package and project manager [uv](https://docs.astral.sh/uv/) to
install dependencies and run our code.
Alternatively, you can find a description of how to use the python package manager
[pip](https://pip.pypa.io/en/stable/) below.

### Installation Using UV

After installing uv, navigate into the project directory and run

```bash
uv sync
```

Uv automatically creates a virtual environment behind the scenes
and installs all dependencies as specified in the `pyproject.toml`
file.

### Installation Using Pip

Alternative to uv, you can manually create your virtual environment and
use [pip](https://pip.pypa.io/en/stable/) to install all required dependencies.

First, create a virtual environment by running

```bash
python -m venv .venv
```

Then, activate the environment by running

```bash
source .venv/bin/activate
```

Finally, install the required dependencies as specified in the `pyproject.toml` file.

```bash
pip install <dependency_name>
```

## Evaluation

To run the experiments presented in the paper, navigate into the `evaluation/surrogate_assisted_qcs/`
directory.

From there, the `run_ea_experiment.py` script serves as the entry point for all experiments.

This script expects as parameters:

- the surrogate model to use during the search.
- the frequency at which exact evaluation should be carried out.
- the number of qubits to use for the synthesis target.
- the number of gates to use for the synthesis target.
- the random seed to use during this experiment.
- optionally, a tag which will be stored together with the experiment's meta data.

To run a single experiment, execute

```bash
python run_ea_experiment.py -m <surrogate> -ee <frequency> -qn <qubits> -gc <gates> -s <seed> -t "<tag>"
```

For an overview of the expected script parameters together with their allowed and default values,
execute

```bash
python run_ea_experiment.py --help
```

To analyse the data and generate the figures reported in the paper,
run the corresponding experiments from within the `evaluation/surrogate_assisted_qcs/`
directory.
Alternatively, you can download the data reported in the paper from the
accompagnying [online repository](https://figshare.com/articles/dataset/Experiment_data_for_the_paper_State_Vector-based_Surrogates_for_Unitary_Synthesis_with_Evolutionary_Algorithms_/33610927?file=68451406) and save it to the `results/` directory.

Then, open the `analysis.ipynb` notebook and run all cells in order.
The figures reported in the paper should automatically be shown and
saved to the `results/` directory.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
