"""gauge_const — the constant baselines for set A (no engine, no item content).

Two gauges in the standard gauge schema (ops/gate3-interfaces.md), engine "constant":

  baseline-uniform   probs_raw = [1/3, 1/3, 1/3] for every item, logits_mean = [0, 0, 0].
  baseline-majority  one-hot on the fit split's most common label (any tie for the top count ->
                     "no-call" = LABELS[0], the pre-registered rule in PREREG.md and the brief);
                     logits_mean = 0.0 for the majority label and -40.0 for the others, so
                     softmax(logits_mean) reproduces the one-hot within 1e-6.

The timing block is all zeros (spec: "timing zeros"); a constant gauge measures nothing, and the
item count lives in n_fit / n_exam and the manifest.

Reads data/seta/fit.jsonl and data/seta/exam.jsonl for ids and labels ONLY. Never prints, logs or
stores any request or function text (rule 4). Writes results/baseline-uniform.json and
results/baseline-majority.json; the majority file carries one extra key, "majority_label".

Run from the repo root:  $PY -m nikasha.gauge_const
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter

from nikasha import (
    EXAM_PATH,
    FIT_PATH,
    LABELS,
    RESULTS_DIR,
    ROOT,
    SEED,
    read_json,
    read_jsonl,
    write_json,
)

UNIFORM_GAUGE = "baseline-uniform"
MAJORITY_GAUGE = "baseline-majority"
ENGINE = "constant"
NEG_LOGIT = -40.0  # exp(-40) ~ 4e-18, so softmax([0, -40, -40]) is one-hot far inside 1e-6


# ----------------------------------------------------------------------------------------------
# data: ids and labels only
# ----------------------------------------------------------------------------------------------
def load_ids_labels(path) -> list[tuple[str, str]]:
    """Return [(id, label), ...] in file order. Only these two keys are ever touched."""
    if not path.exists():
        raise SystemExit(f"gauge_const: missing {path} (run data_seta first)")
    rows = read_jsonl(path)
    out: list[tuple[str, str]] = []
    for i, r in enumerate(rows):
        if "id" not in r or "label" not in r:
            raise SystemExit(f"gauge_const: {path.name} row {i} lacks 'id' or 'label'")
        label = r["label"]
        if label not in LABELS:
            raise SystemExit(f"gauge_const: {path.name} row {i} (id {r['id']}) has a label outside LABELS")
        out.append((str(r["id"]), label))
    ids = [i for i, _ in out]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"gauge_const: {path.name} contains duplicate ids")
    if not out:
        raise SystemExit(f"gauge_const: {path.name} is empty")
    return out


def majority_of_fit(fit_labels: list[str]) -> tuple[str, Counter, bool]:
    """Most common fit label; ANY tie for the top count resolves to LABELS[0] ("no-call").

    This is the pre-registered rule (PREREG.md: "ties -> no-call"; the brief: "all equal -> no-call"),
    and it covers a two-way tie that excludes no-call as well. Returns (label, counts, tied).
    """
    counts = Counter(fit_labels)
    top = max(counts.get(lab, 0) for lab in LABELS)
    leaders = [lab for lab in LABELS if counts.get(lab, 0) == top]
    tied = len(leaders) > 1
    best = LABELS[0] if tied else leaders[0]
    return best, counts, tied


# ----------------------------------------------------------------------------------------------
# tiny numerics (stdlib only; this module has no reason to import numpy)
# ----------------------------------------------------------------------------------------------
def softmax3(logits: list[float]) -> list[float]:
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    s = sum(ex)
    return [e / s for e in ex]


def check_probs(p: list[float]) -> None:
    """The gauge contract: length 3, every entry in [0, 1], sum within 1e-6 of 1."""
    if len(p) != 3:
        raise AssertionError(f"probability vector has length {len(p)}, expected 3")
    if not all(0.0 <= x <= 1.0 for x in p):
        raise AssertionError(f"probability entry outside [0, 1]: {p}")
    if abs(sum(p) - 1.0) > 1e-6:
        raise AssertionError(f"probabilities sum to {sum(p)!r}, not 1 within 1e-6")


def check_logits_reproduce(logits: list[float], probs: list[float]) -> None:
    sm = softmax3(logits)
    gap = max(abs(a - b) for a, b in zip(sm, probs))
    if gap > 1e-6:
        raise AssertionError(f"softmax(logits_mean) differs from probs_raw by {gap:.3e} > 1e-6")


# ----------------------------------------------------------------------------------------------
# result assembly
# ----------------------------------------------------------------------------------------------
def build_result(gauge: str, fit: list[tuple[str, str]], exam: list[tuple[str, str]],
                 logits_mean: list[float], probs_raw: list[float]) -> dict:
    """One constant gauge file in the spec's schema (brief fields + spec extras)."""
    items = []
    for split, rows in (("fit", fit), ("exam", exam)):
        for item_id, label in rows:
            items.append({
                "id": item_id,
                "split": split,
                "label": label,
                "logits_mean": list(logits_mean),
                "logits_by_rotation": [],
                "probs_raw": list(probs_raw),
                "truncated": False,
                "prompt_tokens": 0,
            })
    return {
        "gauge": gauge,
        "engine": ENGINE,
        "engine_path": None,
        "license": None,
        "prompt_sha256": None,
        "prompt_version": None,
        "rotations": 0,
        "seed": SEED,
        "n_fit": len(fit),
        "n_exam": len(exam),
        "labels": list(LABELS),
        # spec: constant baselines carry "timing zeros" -- every field, n_items included
        "timing": {
            "s_per_item": 0.0,
            "wall_s": 0.0,
            "peak_gb": 0.0,
            "n_items": 0,
            "forwards_per_item": 0,
        },
        "items": items,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m nikasha.gauge_const",
        description=("Write the constant baselines (uniform and majority-of-fit) for set A to "
                     "results/baseline-uniform.json and results/baseline-majority.json. "
                     "Reads ids and labels only; prints counts and the majority label only."),
    )
    ap.add_argument("--force", action="store_true",
                    help="overwrite the baseline files even if they already carry a scored 'exam' block")
    args = ap.parse_args(argv)

    fit = load_ids_labels(FIT_PATH)
    exam = load_ids_labels(EXAM_PATH)

    overlap = {i for i, _ in fit} & {i for i, _ in exam}
    if overlap:
        raise SystemExit(f"gauge_const: fit and exam share {len(overlap)} id(s); the split is broken")

    # --- uniform -------------------------------------------------------------------------------
    uniform_probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    uniform_logits = [0.0, 0.0, 0.0]
    check_probs(uniform_probs)
    check_logits_reproduce(uniform_logits, uniform_probs)

    # --- majority of fit (design signal comes from the fit split only, rule 8) ----------------
    maj_label, fit_counts, tied = majority_of_fit([lab for _, lab in fit])
    k = LABELS.index(maj_label)
    maj_probs = [1.0 if j == k else 0.0 for j in range(3)]
    maj_logits = [0.0 if j == k else NEG_LOGIT for j in range(3)]
    check_probs(maj_probs)
    check_logits_reproduce(maj_logits, maj_probs)

    uniform = build_result(UNIFORM_GAUGE, fit, exam, uniform_logits, uniform_probs)
    majority = build_result(MAJORITY_GAUGE, fit, exam, maj_logits, maj_probs)
    majority["majority_label"] = maj_label

    # final contract check over every stored vector before anything is written
    for res in (uniform, majority):
        for it in res["items"]:
            check_probs(it["probs_raw"])

    uniform_path = RESULTS_DIR / f"{UNIFORM_GAUGE}.json"
    majority_path = RESULTS_DIR / f"{MAJORITY_GAUGE}.json"
    # Rule 8 guard: never silently discard a scored exam block (a rewrite would force a second exam
    # read of the same gauge version). --force is the explicit, logged override.
    if not getattr(args, "force", False):
        for path in (uniform_path, majority_path):
            if path.exists():
                try:
                    existing = read_json(path)
                except (OSError, ValueError):
                    existing = {}
                if isinstance(existing, dict) and "exam" in existing:
                    raise SystemExit(f"STOP: {path.relative_to(ROOT)} already carries an 'exam' block; "
                                     f"refusing to overwrite a scored gauge file (use --force to override)")
    write_json(uniform_path, uniform)
    write_json(majority_path, majority)

    # --- report: counts and the majority label only (no item content) -------------------------
    print(f"n_fit={len(fit)} n_exam={len(exam)} n_items={len(fit) + len(exam)}")
    print("fit label counts: " + " ".join(f"{lab}={fit_counts.get(lab, 0)}" for lab in LABELS))
    tie_note = " (tie -> no-call, pre-registered rule)" if tied else ""
    print(f"majority label (fit): {maj_label}{tie_note}")
    print(f"wrote {uniform_path.relative_to(uniform_path.parents[1])} "
          f"({len(uniform['items'])} items, probs_raw={uniform_probs})")
    print(f"wrote {majority_path.relative_to(majority_path.parents[1])} "
          f"({len(majority['items'])} items, probs_raw={maj_probs}, logits_mean={maj_logits})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
