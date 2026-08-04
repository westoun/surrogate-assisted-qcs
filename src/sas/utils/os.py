import json
from typing import Tuple, Any


def save_to_json(obj, path: str) -> None:
    with open(path, "w") as config_file:
        json.dump(obj, config_file)


def load_from_json(path: str) -> Any:
    with open(path, "r") as config_file:
        return json.load(config_file)
