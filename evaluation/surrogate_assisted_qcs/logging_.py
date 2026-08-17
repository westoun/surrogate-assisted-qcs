
from typing import List, Dict

from src.sas.ea import EAParams
from src.sas.utils.os import save_to_json, load_from_json
from src.sas.utils.time import get_timestamp


def log_experiment_details(
    ea_params: EAParams,
    surrogate_params: Dict,
    seed: int,
    tag: str,
    logging_prefix: str
) -> None:
    target_path = logging_prefix + "_config.json"

    config = {
        "meta": {
            "start": get_timestamp(),
        },
        "params": {
            "ea": {
                "parent_count": ea_params.parent_count,
                "offspring_count": ea_params.offspring_count,
                "max_generations": ea_params.max_generations,
                "qubit_num": ea_params.qubit_num,
                "gate_count": ea_params.gate_count,
                "evaluate_every": ea_params.evaluate_every
            },
            "surrogate": surrogate_params,
            "seed": seed,
            "tag": tag,
            "logging_prefix": logging_prefix
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
