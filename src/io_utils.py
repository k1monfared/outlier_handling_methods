"""Path helpers and small readers/writers used across the pipeline."""
from __future__ import annotations

import json
import os

# Repo root is the parent of the directory holding this file (src/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(ROOT, "configs", "experiment_config.json")
DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
IMAGES_DIR = os.path.join(ROOT, "docs", "images")

BANK_PARQUET = os.path.join(DATA_DIR, "metric_bank.parquet")
GROUND_TRUTH_CSV = os.path.join(DATA_DIR, "ground_truth.csv")
SAMPLE_CSV = os.path.join(DATA_DIR, "metric_bank_sample.csv")


def ensure_dirs() -> None:
    for d in (DATA_DIR, OUTPUTS_DIR, IMAGES_DIR):
        os.makedirs(d, exist_ok=True)


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
        fh.write("\n")


def _json_default(o):
    # numpy scalar / array friendliness
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
