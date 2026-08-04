
from typing import List

from sas.types.circuit import Circuit
from src.sas.utils.os import save_to_json
from src.sas.utils.time import duration_to_seconds, get_timestamp


def log_dataset_details(circuits: List[Circuit],
                        qubit_num: int,
                        gate_count: int,
                        circuit_count: int,
                        seed: int,
                        duration: int,
                        data_path: str) -> None:

    target_path = f"results/data_{qubit_num}qn{gate_count}gc{len(circuits)}cc{seed}s.json"

    dataset_details = {
        "meta": {
            "timestamp": get_timestamp()
        },
        "params": {
            "qubit_num": qubit_num,
            "gate_count": gate_count,
            "circuit_count": circuit_count,
            "seed": seed,
        },
        "data": {
            "fitness_scores": [
                circuit.fitness for circuit in circuits
            ],
            "duration": duration,
            "data_path": data_path
        }
    }

    save_to_json(dataset_details, target_path)
