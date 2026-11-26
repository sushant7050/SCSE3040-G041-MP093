"""Load params.yaml. Every other module gets its settings from here."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]   # src/mp093/config.py -> repo root
PARAMS_FILE = ROOT / "params.yaml"


def load_params(path=PARAMS_FILE):
    """Return params.yaml as a dict (yaml.safe_load, never yaml.load)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_path(params, name):
    """Absolute path of params['data'][name], e.g. data_path(p, 'train')."""
    return ROOT / params["data"][name]
