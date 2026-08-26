from multiprocessing import Pool
import os
from typing import List

from run_ea_experiment import run_experiment


def execute(params: List):
    run_experiment(params, standalone_mode=False)


if __name__ == "__main__":
    qcs_setups = [
        (4, 20),
        (6, 18),
        (6, 30),
    ]

    evaluate_every_values = [
        # 5, 10, 15
        10
    ]

    models = [
        "calibrated_state_vector", "random_base_state_vector", "state_vector",
        "gnn", "circuit_feature", "random_fitness", "None"
    ]
    seed_num = 15
    seed_offset = 0

    tag = None

    experiment_configs = []
    for seed_i in range(seed_num):
        seed = seed_offset + seed_i
        for qubit_num, gate_count in qcs_setups:
            for model in models:
                for evaluate_every in evaluate_every_values:
                    experiment_configs.append([
                        "-m", model,
                        "-qn", qubit_num,
                        "-gc", gate_count,
                        "-ee", evaluate_every,
                        "-s", seed,
                        "-t", tag
                    ])

    max_cores = os.cpu_count()
    print(f"Available number of cores: {max_cores}")

    print(f"Starting {len(experiment_configs)} experiments.")

    with Pool(max_cores) as pool:

        pool.map(
            execute, experiment_configs
        )

    print(f"Finished {len(experiment_configs)} experiments.")
