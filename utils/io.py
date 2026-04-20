"""
utils/io.py
-----------
All file I/O for the pipeline. Every other module imports from here.
No pipeline logic lives in this file.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV. Creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_config(path: str | Path) -> dict:
    """Load a .yaml or .json config file into a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml or .json")


def save_config(cfg: dict, path: str | Path) -> None:
    """Save a dict as .yaml or .json. Creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if path.suffix in (".yaml", ".yml"):
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == ".json":
            json.dump(cfg, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml or .json")


def load_pt(path: str | Path) -> Any:
    """Load a .pt file saved with torch.save."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f".pt file not found: {path}")
    return torch.load(path, map_location="cpu")


def save_pt(obj: Any, path: str | Path) -> None:
    """Save an object to a .pt file. Creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)