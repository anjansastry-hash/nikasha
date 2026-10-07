"""calibrate.py — temperature, per-class thresholds and the abstain rule, fitted on the FIT split only.

For each gauge result file ``results/<gauge>.json`` this module

1. keeps ONLY the items whose ``split == "fit"`` (exam items are filtered out immediately and never
   looked at again — rule 8 of the brief);
2. fits one scalar temperature ``T`` on ``logits_mean`` by minimising fit NLL
   (``metrics.fit_temperature``: 200-point log grid on [0.05, 10] plus a bounded local refine);
   constant engines (``engine == "constant"``) get ``T = 1.0`` with ``T_method = "fixed"``;
3. calibrates ``probs_cal = softmax(logits_mean / T)``;
4. finds the global τ and the per-class thresholds ``taus`` with ``metrics.fit_thresholds`` so that the
   fit ask rate is minimal subject to selective accuracy ≥ ``SA_TARGET``;
5. writes ``results/<gauge>.calib.json`` (schema: ops/gate3-interfaces.md) and prints one summary line.

It never opens ``data/seta/*.jsonl``, never touches exam items and never prints item text — only ids,
counts, numbers. Every metric comes from ``nikasha/metrics.py``; nothing is reimplemented here.

Every .calib.json carries ``fit_fingerprint`` (``nikasha.fit_fingerprint``: sha256 over the sorted fit ids
and their logits_mean), which binds it to the gauge run it was fitted on; score.py refuses to score when
the gauge file's fingerprint differs. A gauge with two or more rotations also gets ``letter_bias_fit``:
mean over fit items of (r0 logit − mean-over-rotations logit), per label.

Usage (from the repo root)::

    $PY -m nikasha.calibrate                 # every gauge result that has no .calib.json yet
    $PY -m nikasha.calibrate --gauge NAME    # that one gauge (recomputed even if a .calib.json exists)
    $PY -m nikasha.calibrate --all           # recompute every gauge result
    $PY -m nikasha.calibrate --stamp         # existing .calib.json files: re-derive T and thresholds in
                                             # memory, refuse on any difference, then ADD fit_fingerprint
                                             # (and letter_bias_fit) without changing any stored number
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from nikasha import LABELS, RESULTS_DIR, SA_TARGET, fit_fingerprint, metrics, read_json, write_json

# The search grids metrics.py uses (documented in the calib.json "grid" block). These mirror the
# brief and the interface spec; metrics.fit_temperature / metrics.fit_thresholds own the actual search.
T_MIN, T_MAX, T_POINTS = 0.05, 10, 200
TAU_MIN, TAU_MAX, TAU_STEP = 0.34, 0.99, 0.01

CALIB_SUFFIX = ".calib.json"
PROBS_RAW_WARN_TOL = 1e-4  # warn when stored probs_raw disagree with softmax(logits_mean) by more than this


class CalibrationError(Exception):
    """A gauge result file that cannot be calibrated (malformed items, no fit split, bad labels ...)."""


# --------------------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------------------

def calib_path_for(result_path: Path) -> Path:
    """results/<gauge>.json -> results/<gauge>.calib.json (the stem pairs the two files for score.py)."""
    return result_path.with_name(result_path.stem + CALIB_SUFFIX)


def why_not_gauge_result(obj) -> str | None:
    """None for a dict with a string ``gauge`` and a non-empty ``items`` list whose first entry carries
    ``split`` and ``logits_mean``; otherwise a one-phrase reason. canary.json / ku2-hidden.json do not
    qualify (no ``gauge``, items without ``split``/``logits_mean``)."""
    if not isinstance(obj, dict):
        return "top level is not a JSON object"
    items = obj.get("items")
    if not isinstance(items, list):
        return "no 'items' list"
    if not isinstance(obj.get("gauge"), str):
        return "no string 'gauge' key"
    if not items:
        return "'items' is empty"
    first = items[0]
    if not (isinstance(first, dict) and "split" in first and "logits_mean" in first):
        return "items do not carry split/logits_mean"
    return None


def looks_like_gauge_result(obj) -> bool:
    return why_not_gauge_result(obj) is None


def has_exam_items(obj: dict) -> bool:
    """True when at least one item is on the exam split. Only the ``split`` field is inspected; a
    fit-only file (e.g. a --timing run written in the gauge schema) has nothing for score.py to score,
    so default discovery leaves it alone instead of giving it a .calib.json."""
    return any(isinstance(it, dict) and it.get("split") == "exam" for it in obj["items"])


def discover(results_dir: Path, recompute_all: bool) -> tuple[list[tuple[Path, dict]], int]:
    """All gauge result files in results/ (top level only, sorted), optionally skipping those that
    already have a .calib.json. Returns (targets, n_already_calibrated); the count is 0 under
    ``recompute_all``. Any file that has an ``items`` list but is not calibrated gets a stderr notice."""
    found: list[tuple[Path, dict]] = []
    n_done = 0
    for path in sorted(results_dir.glob("*.json")):
        if not path.is_file() or path.name.endswith(CALIB_SUFFIX):
            continue
        if not recompute_all and calib_path_for(path).exists():
            n_done += 1
            continue
        try:
            obj = read_json(path)
        except ValueError as e:  # json.JSONDecodeError is a ValueError
            print(f"calibrate: skipping results/{path.name}: not valid JSON ({e})", file=sys.stderr)
            continue
        why = why_not_gauge_result(obj)
        if why is not None:
            if isinstance(obj, dict) and isinstance(obj.get("items"), list):
                print(f"calibrate: skipping results/{path.name}: {why}", file=sys.stderr)
            continue
        if not has_exam_items(obj):
            print(f"calibrate: skipping results/{path.name}: no items with split == 'exam' (fit-only file, "
                  f"e.g. a timing run); use --gauge {path.stem} to calibrate it anyway", file=sys.stderr)
            continue
        found.append((path, obj))
    return found, n_done


def resolve_one(results_dir: Path, name: str) -> tuple[Path, dict]:
    """--gauge NAME -> (results/NAME.json, parsed object). NAME may carry a trailing .json; it must be
    a bare file stem (no path separators, not '.' or '..') so the read stays inside results/."""
    if name.endswith(".json"):
        name = name[: -len(".json")]
    if name.endswith(".calib"):
        name = name[: -len(".calib")]
    if not name or name in (".", "..") or Path(name).name != name:
        raise CalibrationError(f"--gauge NAME must be a bare file stem inside results/ (got {name!r})")
    path = results_dir / f"{name}.json"
    if not path.is_file():
        raise CalibrationError(f"no gauge result file at results/{path.name}")
    try:
        obj = read_json(path)
    except ValueError as e:  # json.JSONDecodeError is a ValueError
        raise CalibrationError(f"results/{path.name} is not valid JSON ({e})") from None
    if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
        raise CalibrationError(f"results/{path.name} has no 'items' list")
    return path, obj


# --------------------------------------------------------------------------------------------------
# calibration of one gauge (fit split only)
# --------------------------------------------------------------------------------------------------

def _r4(v):
    return None if v is None else round(float(v), 4)


def _fit_arrays(obj: dict, stem: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Filter to the fit split and pull (fit_items, logits_mean[n,3], y[n]). Exam items are dropped
    here and never referenced again."""
    fit_items = [it for it in obj["items"] if isinstance(it, dict) and it.get("split") == "fit"]
    n_fit = len(fit_items)
    if n_fit == 0:
        raise CalibrationError("no items with split == 'fit'")

    try:
        logits = np.asarray([it["logits_mean"] for it in fit_items], dtype=np.float64)
    except KeyError:
        raise CalibrationError("a fit item lacks 'logits_mean'") from None
    except (TypeError, ValueError) as e:
        raise CalibrationError(f"logits_mean is not a numeric {len(LABELS)}-vector on every fit item: {e}") from None
    if logits.ndim != 2 or logits.shape[1] != len(LABELS):
        raise CalibrationError(f"logits_mean has shape {logits.shape}, expected ({n_fit}, {len(LABELS)})")

    try:
        y = metrics.labels_to_idx([it["label"] for it in fit_items])
    except KeyError:
        raise CalibrationError("a fit item lacks 'label'") from None
    except ValueError as e:
        raise CalibrationError(f"a fit item has a label outside {LABELS}: {e}") from None

    declared = obj.get("n_fit")
    if declared is not None and declared != n_fit:
        print(f"calibrate: {stem}: warning: file declares n_fit={declared} but {n_fit} items have split == 'fit'",
              file=sys.stderr)
    return fit_items, logits, y


def _check_probs_raw(fit_items: list[dict], probs_raw: np.ndarray, stem: str) -> None:
    """Sanity check only: the stored probs_raw should be softmax(logits_mean) at T = 1."""
    try:
        stored = np.asarray([it["probs_raw"] for it in fit_items], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return
    if stored.shape != probs_raw.shape:
        return
    gap = float(np.max(np.abs(stored - probs_raw)))
    if gap > PROBS_RAW_WARN_TOL:
        print(f"calibrate: {stem}: warning: stored probs_raw differ from softmax(logits_mean) by up to {gap:.3e}",
              file=sys.stderr)


def _signed(v: float) -> str:
    """+0.36 / −1.63: two decimals, explicit sign, typographic minus."""
    return f"{v:+.2f}".replace("-", "−")


def letter_bias(fit_items: list[dict]) -> dict | None:
    """Mean over fit items of (r0 logit − mean-over-rotations logit), per label, in label order. None
    unless every fit item carries at least two rotations (a one-rotation gauge has no bias line)."""
    if not fit_items or not all(isinstance(it.get("logits_by_rotation"), list)
                                and len(it["logits_by_rotation"]) >= 2 for it in fit_items):
        return None
    k_labels = len(LABELS)
    raw = [sum(float(it["logits_by_rotation"][0][k]) - float(it["logits_mean"][k]) for it in fit_items)
           / len(fit_items) for k in range(k_labels)]
    return {
        "split": "fit",
        "n": len(fit_items),
        "definition": "mean over fit items of (rotation r0 restricted logit − mean over rotations), per label, label order",
        "value": [round(v, 4) for v in raw],
        "display": "[" + ", ".join(_signed(v) for v in raw) + "]",
    }


def calibrate_gauge(obj: dict, stem: str) -> dict:
    """Fit T and thresholds on the fit split of one gauge result; return the calib.json object."""
    gauge = obj.get("gauge") if isinstance(obj.get("gauge"), str) else stem
    if gauge != stem:
        print(f"calibrate: {stem}: warning: file's gauge name is {gauge!r}; output is paired by file stem",
              file=sys.stderr)
    engine = obj.get("engine")

    fit_items, logits, y = _fit_arrays(obj, stem)
    n_fit = len(fit_items)

    if engine == "constant":
        T, T_method = 1.0, "fixed"
    else:
        if not np.all(np.isfinite(logits)):
            raise CalibrationError("non-finite logits_mean on the fit split; cannot fit a temperature")
        T, _info = metrics.fit_temperature(logits, y)
        T, T_method = float(T), "grid+refine"
        if not (np.isfinite(T) and T > 0):
            raise CalibrationError(f"fit_temperature returned T={T!r}")

    probs_raw = metrics.softmax(logits, 1.0)
    probs_cal = metrics.softmax(logits, T)
    _check_probs_raw(fit_items, probs_raw, stem)

    thr = metrics.fit_thresholds(probs_cal, y, SA_TARGET)
    taus = [round(float(t), 2) for t in thr["taus"]]
    tau_global = None if thr["tau_global"] is None else round(float(thr["tau_global"]), 2)
    reachable = bool(thr["target_reachable"])
    taus_arr = np.asarray(taus, dtype=np.float64)

    fit = {
        "nll_raw": _r4(metrics.nll(probs_raw, y)),
        "nll_cal": _r4(metrics.nll(probs_cal, y)),
        "ece_raw": _r4(metrics.ece(probs_raw, y)),
        "ece_cal": _r4(metrics.ece(probs_cal, y)),
        "accuracy": _r4(metrics.accuracy(probs_cal, y)),
        "ask_rate": _r4(metrics.ask_rate(probs_cal, taus_arr)),
        "selective_accuracy": _r4(metrics.selective_accuracy(probs_cal, y, taus_arr)),
    }

    calib = {
        "gauge": gauge,
        "n_fit": n_fit,
        "sa_target": SA_TARGET,
        "T": T,
        "T_method": T_method,
        "tau_global": tau_global,
        "taus": taus,
        "target_reachable_on_fit": reachable,
        "fit": fit,
        "grid": {
            "T_min": T_MIN, "T_max": T_MAX, "T_points": T_POINTS,
            "tau_min": TAU_MIN, "tau_max": TAU_MAX, "tau_step": TAU_STEP,
        },
        "fit_fingerprint": fit_fingerprint(obj["items"]),
    }
    bias = letter_bias(fit_items)
    if bias is not None:
        calib["letter_bias_fit"] = bias
    return calib


# Keys whose stored values --stamp must reproduce exactly before it may add the fingerprint.
STAMP_VERIFY_KEYS = ("gauge", "n_fit", "sa_target", "T", "T_method", "tau_global", "taus",
                     "target_reachable_on_fit", "fit", "grid")
STAMP_ADD_KEYS = ("fit_fingerprint", "letter_bias_fit")


def stamp_existing(results_dir: Path) -> int:
    """Bind every existing .calib.json to its gauge run without changing a stored number: re-derive the
    calibration in memory from the gauge file's fit split, refuse when any stored key differs (the calib
    then does not belong to this run), and only ADD the missing STAMP_ADD_KEYS."""
    n_fail = n_done = 0
    for calib_path in sorted(results_dir.glob(f"*{CALIB_SUFFIX}")):
        stem = calib_path.name[: -len(CALIB_SUFFIX)]
        gauge_path = results_dir / f"{stem}.json"
        if not gauge_path.is_file():
            n_fail += 1
            print(f"calibrate --stamp: {stem}: no gauge file results/{gauge_path.name}", file=sys.stderr)
            continue
        stored = read_json(calib_path)
        try:
            fresh = calibrate_gauge(read_json(gauge_path), stem)
        except CalibrationError as e:
            n_fail += 1
            print(f"calibrate --stamp: {stem}: cannot re-derive: {e}", file=sys.stderr)
            continue
        diffs = [k for k in STAMP_VERIFY_KEYS if stored.get(k) != fresh.get(k)]
        diffs += [k for k in STAMP_ADD_KEYS if k in stored and stored[k] != fresh.get(k)]
        if diffs:
            n_fail += 1
            print(f"calibrate --stamp: {stem}: REFUSED — stored {diffs} differ from a re-derivation on the "
                  f"current gauge file; this calib was not fitted on this run", file=sys.stderr)
            continue
        added = [k for k in STAMP_ADD_KEYS if k not in stored and k in fresh]
        for k in added:
            stored[k] = fresh[k]
        if added:
            write_json(calib_path, stored)
        n_done += 1
        print(f"{stem}: re-derivation identical on {len(STAMP_VERIFY_KEYS)} keys; added {added or 'nothing'}; "
              f"fit_fingerprint {stored['fit_fingerprint']['sha256'][:16]}… (n_fit {stored['fit_fingerprint']['n_fit']})")
    print(f"calibrate --stamp: {n_done} calib file(s) bound, {n_fail} refused")
    return 1 if n_fail else 0


def summary_line(stem: str, calib: dict, out_path: Path) -> str:
    fit = calib["fit"]
    fmt = lambda v, nd=4: "null" if v is None else f"{v:.{nd}f}"  # noqa: E731
    taus_s = "[" + ", ".join(f"{t:.2f}" for t in calib["taus"]) + "]"
    return (
        f"{stem}: n_fit={calib['n_fit']} T={calib['T']:.4f} ({calib['T_method']}) "
        f"tau_global={fmt(calib['tau_global'], 2)} taus={taus_s} "
        f"fit ask_rate={fmt(fit['ask_rate'])} SA={fmt(fit['selective_accuracy'])}"
        f"{'' if calib['target_reachable_on_fit'] else ' [SA target NOT reachable on fit]'} | "
        f"acc={fmt(fit['accuracy'])} nll {fmt(fit['nll_raw'])}->{fmt(fit['nll_cal'])} "
        f"ece {fmt(fit['ece_raw'])}->{fmt(fit['ece_cal'])} -> results/{out_path.name}"
    )


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m nikasha.calibrate",
        description="Fit temperature and per-class abstain thresholds on the FIT split of each gauge "
                    "result in results/ and write results/<gauge>.calib.json.",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--gauge", metavar="NAME",
                      help="calibrate only results/NAME.json (recomputed even if NAME.calib.json exists)")
    mode.add_argument("--all", action="store_true",
                      help="recompute every gauge result, including those that already have a .calib.json")
    mode.add_argument("--stamp", action="store_true",
                      help="add fit_fingerprint (and letter_bias_fit) to existing .calib.json files after "
                           "verifying that a re-derivation reproduces every stored number")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    results_dir = Path(RESULTS_DIR)
    if args.stamp:
        return stamp_existing(results_dir)
    if args.gauge:
        try:
            targets = [resolve_one(results_dir, args.gauge)]
        except CalibrationError as e:
            print(f"calibrate: {e}", file=sys.stderr)
            return 2
    else:
        if not results_dir.is_dir():
            print(f"calibrate: nothing to do (no results directory at {results_dir})")
            return 0
        targets, n_done = discover(results_dir, recompute_all=args.all)
        if not targets:
            if n_done:
                print(f"calibrate: nothing to do ({n_done} gauge result(s) in results/ already have a "
                      f".calib.json; use --all to recompute)")
            else:
                print("calibrate: nothing to do (no gauge result files found in results/; "
                      "run the gauge and baseline first)")
            return 0

    n_fail = 0
    for path, obj in targets:
        stem = path.stem
        out_path = calib_path_for(path)
        try:
            calib = calibrate_gauge(obj, stem)
        except CalibrationError as e:
            n_fail += 1
            print(f"calibrate: {stem}: FAILED: {e}", file=sys.stderr)
            continue
        write_json(out_path, calib)
        print(summary_line(stem, calib, out_path))

    if n_fail:
        print(f"calibrate: {n_fail} of {len(targets)} gauge(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
