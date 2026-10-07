"""nikasha — one sealed exam, many decision gauges. Shared constants and tiny I/O helpers.

A gauge is any program that, for one item, returns a probability vector over LABELS: length 3,
every entry in [0, 1], sum within 1e-6 of 1. Gauges write results/<gauge>.json; the harness
(calibrate.py, score.py, readme.py) reads JSON and never imports an engine.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

LABELS = ["no-call", "one-call", "multi-call"]
SEED = 20261007
N_BOOT = 1000
SA_TARGET = 0.95
ECE_BINS = 15

ENGINE_NAME = "gemma-4-12b-8bit-text"
# Portable (Gate 5): override with NIKASHA_ENGINE / NIKASHA_BFCL; the defaults are the Studio layout.
ENGINE_PATH = os.environ.get("NIKASHA_ENGINE") or os.path.expanduser("~/mlx-models/gemma-4-12b-8bit-text")
BFCL_DIR = os.environ.get("NIKASHA_BFCL") or os.path.expanduser("~/data/bfcl")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "seta"
RESULTS_DIR = ROOT / "results"
CARDS_DIR = ROOT / "cards"
OPS_DIR = ROOT / "ops"
FIT_PATH = DATA_DIR / "fit.jsonl"
EXAM_PATH = DATA_DIR / "exam.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")


FINGERPRINT_RECIPE = (
    "sha256 of the UTF-8 text json.dumps([[id, logits_mean], ...], separators=(',', ':'), ensure_ascii=False) "
    "over the items with split == 'fit', sorted by id, logits as floats"
)


def fit_fingerprint(items) -> dict:
    """Binds a .calib.json to the gauge run it was fitted on: sha256 over the sorted fit ids and their
    logits_mean (the only inputs calibrate.py reads). A re-run gauge changes it, so score.py can refuse
    stale T / thresholds."""
    rows = sorted(
        ([str(it["id"]), [float(x) for x in it["logits_mean"]]]
         for it in items if isinstance(it, dict) and it.get("split") == "fit"),
        key=lambda r: r[0],
    )
    text = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    return {"sha256": sha256_text(text), "n_fit": len(rows), "recipe": FINGERPRINT_RECIPE}
