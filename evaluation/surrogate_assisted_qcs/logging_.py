
from typing import List

from sas.types.circuit import Circuit
from src.sas.utils.os import save_to_json, load_from_json
from src.sas.utils.time import get_timestamp


def log_experiment_details(
    model: str,
    qubit_num: int,
    gate_count: int,
    seed: int,
    tag: str,
    parent_count: int,
    offspring_count: int,
    max_generations: int,
    logging_prefix: str
) -> None:
    target_path = logging_prefix + "_config.json"

    config = {
        "meta": {
            "start": get_timestamp(),
        },
        "params": {
            "logging_prefix": logging_prefix,
            "model": model,
            "qubit_num": qubit_num,
            "gate_count": gate_count,
            "seed": seed,
            "tag": tag,
            "parent_count": parent_count,
            "offspring_count": offspring_count,
            "max_generations": max_generations,
        }
    }
    save_to_json(config, target_path)


def log_end_timestamp(
    logging_prefix: str
) -> None:
    target_path = logging_prefix + "_config.json"
    config = load_from_json(target_path)

    config["meta"]["end"] = get_timestamp()

    save_to_json(config, target_path)
