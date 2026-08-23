
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
                "max_evaluations": ea_params.max_evaluations,
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


def update_experiment_details(
    logging_prefix: str,
    ea_params: EAParams = None,
    surrogate_params: Dict = None,
    log_end_timestamp: bool = False

) -> None:
    target_path = logging_prefix + "_config.json"
    config = load_from_json(target_path)

    if ea_params is not None:
        config["params"]["ea"] = ea_params

    if surrogate_params is not None:
        config["params"]["surrogate"] = surrogate_params

    if log_end_timestamp:
        config["meta"]["end"] = get_timestamp()

    save_to_json(config, target_path)
