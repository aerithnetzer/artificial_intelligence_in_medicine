import json
import pandas as pd
from artificial_intelligence_in_medicine.config import INTERIM_DATA_DIR

from typing import Any, List, Union

PathPart = Union[str, int]


def find_key_paths(obj: Any, target_key: str = "AbstractText") -> List[List[PathPart]]:
    """
    Recursively find all paths to `target_key` in a nested structure of dicts/lists.
    Returns a list of paths, where each path is a list of keys/indices.
    """
    paths = []

    def _walk(x: Any, path: List[PathPart]):
        if isinstance(x, dict):
            for k, v in x.items():
                new_path = path + [k]
                if k == target_key:
                    paths.append(new_path)
                _walk(v, new_path)

        elif isinstance(x, list):
            for i, v in enumerate(x):
                _walk(v, path + [i])

    _walk(obj, [])
    return paths


def _finditem(obj, key):
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            return _finditem(v, key)  # added return statement


with open(INTERIM_DATA_DIR / "ARTIFICIAL_INTELLIGENCE" / "batch_00000.json", "r") as f:
    data = json.load(f)

paths = find_key_paths(data, "AbstractText")
print(paths)
