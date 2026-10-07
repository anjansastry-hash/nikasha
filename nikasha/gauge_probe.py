"""gauge ③ — frozen 12B hidden state + linear head (brief 10, Task B), and the head variants of the
labels-needed curve (Task C). Pre-registered in PREREG.md, "## Amendment 1 (Gate 4)".

Features (`features`): for every set-A item, prompt v1 at rotation r0 — rendered by gauge ①'s own code
(nikasha.ku2_hidden.render_prompt: TOOLS-block cap, system/user text, chat template, think-block assertion;
then Reader._ensure_single_bos) — goes through ONE forward pass of the inner text model at the KU2 path
`language_model.model`, which returns the final-normed hidden state (Gemma4TextModel.norm(h)). The vector at
the last position, cast to float32, is the item's feature: the exact vector the model's tied head consumes.
Cross-check per item: the model's own head (tied embed_tokens.as_linear + logit softcap) applied to that
last-position vector, restricted to the letter ids, must reproduce gauge ①'s stored r0 logits for the same
item (a different render would miss by whole logits; one bf16 ulp is 0.125 at magnitude 16–32).
Output: results/gauge3-features.npy (float32, n x hidden, fit rows in fit.jsonl order then exam rows in
exam.jsonl order) + results/gauge3-features.index.json (ids, splits, sha256, provenance).

Head (`head`): multinomial logistic regression (scikit-learn, L2 via l1_ratio=0, lbfgs, class_weight=None)
on standardised features — StandardScaler + LogisticRegression in one Pipeline, so scaling statistics come
from the training rows of each fit (fit split only; exam rows never contribute). C by 5-fold stratified CV
on the fit split (StratifiedKFold(shuffle=True, random_state=SEED)), grid logspace(-4, 2, 13), criterion:
pooled out-of-fold NLL (first minimum on ties = smallest C). The fit-split logits written to the gauge file
are the OUT-OF-FOLD decision values at the chosen C (same folds), flagged `oof: true`; calibrate.py fits T
and thresholds on them only. Exam logits come from the head refit on all 300 fit items. Output:
results/gauge3-probe.json in the gauge schema (logits_mean = head decision values, rotations 1, C, oof).

Labels-needed (`labels-needed`): head variants trained on n in {24, 48, 96, 192, 300} fit labels (equal per
class; 5 draws with seeds SEED+k, k = 0..4; n = 300 is the whole fit split, one draw). Per variant: C by the
same CV (3-fold if n < 96), OOF logits within the subsample, T + per-class thresholds on those OOF logits
(calibrate.calibrate_gauge — the same code as every gauge), refit on the n items, exam decision values.
Output: results/labels-needed.json (no exam label is read here; score.py --labels-needed scores it).

`--dry-run` (head / labels-needed): fit split only — runs the fit-side procedure on whatever features file
is given and prints fit numbers; writes nothing and needs no exam row (rule 8 design check).

Never prints item text — ids, counts, numbers and timings only.

  $PY -m nikasha.gauge_probe features [--splits all|fit] [--out-npy P --out-index P]
  $PY -m nikasha.gauge_probe head [--dry-run] [--features P --index P]
  $PY -m nikasha.gauge_probe labels-needed [--dry-run] [--features P --index P]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nikasha import (
    ENGINE_NAME,
    ENGINE_PATH,
    EXAM_PATH,
    FIT_PATH,
    LABELS,
    MANIFEST_PATH,
    RESULTS_DIR,
    SEED,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)

GAUGE = "gauge3-probe"
PREREG_SECTION = "## Amendment 1 (Gate 4)"
FEATURES_NPY = RESULTS_DIR / "gauge3-features.npy"
FEATURES_INDEX = RESULTS_DIR / "gauge3-features.index.json"
PROBE_PATH = RESULTS_DIR / f"{GAUGE}.json"
LABELS_NEEDED_PATH = RESULTS_DIR / "labels-needed.json"
GAUGE1_PATH = RESULTS_DIR / "gauge1-logit.json"
KU2_PATH = RESULTS_DIR / "ku2-hidden.json"

ROTATION = 0
C_GRID = [float(c) for c in np.logspace(-4, 2, 13)]
MAX_ITER = 20000
CV_FOLDS = 5
LN_SIZES = [24, 48, 96, 192, 300]
LN_SEEDS = 5            # seeds SEED + k, k = 0..4 (n = 300: one draw)
LN_SMALL_FOLDS = 3      # 3-fold CV when n < 96
LN_SMALL_N = 96
R0_STOP_DIFF = 1.0      # a feature whose head misses gauge ①'s r0 logits by > 1 logit was not the same render
PROGRESS_EVERY = 100


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stop(msg: str):
    raise SystemExit(f"STOP: {msg}")


# ----------------------------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------------------------
def load_split_items(which: str) -> list[tuple[str, dict]]:
    """[(split, item)] in fit.jsonl then exam.jsonl order, after the manifest sha256 check (the same
    gate gauge ① passes). `which` = "fit" or "all"."""
    from nikasha.gauge_logit import validate_items, verify_manifest_sha

    if not MANIFEST_PATH.exists():
        stop(f"{MANIFEST_PATH} missing — set A is not sealed")
    manifest = read_json(MANIFEST_PATH)
    verify_manifest_sha(manifest, FIT_PATH, "fit.jsonl")
    fit = read_jsonl(FIT_PATH)
    validate_items(fit, "fit")
    plan = [("fit", it) for it in fit]
    if which == "all":
        verify_manifest_sha(manifest, EXAM_PATH, "exam.jsonl")
        exam = read_jsonl(EXAM_PATH)
        validate_items(exam, "exam")
        if {it["id"] for it in fit} & {it["id"] for it in exam}:
            stop("fit and exam overlap")
        plan += [("exam", it) for it in exam]
    return plan


def gauge1_r0_logits() -> dict[str, list[float]]:
    """id -> gauge ①'s stored r0 restricted logits (label order; r0 letter order == label order)."""
    g1 = read_json(GAUGE1_PATH)
    return {it["id"]: [float(v) for v in it["logits_by_rotation"][0]] for it in g1["items"]}


def mode_features(which: str, out_npy: Path, out_index: Path) -> int:
    import mlx.core as mx

    from nikasha.gauge_logit import load_engine, peak_gb, clear_cache, prompt_sha256, PROMPT_VERSION, LETTERS
    from nikasha.ku2_hidden import _resolve, find_hidden_size, make_head, render_prompt

    ku2 = read_json(KU2_PATH)
    if not ku2.get("pass") or ku2.get("attr_path") != "language_model.model":
        stop("results/ku2-hidden.json does not record a passing check at language_model.model")
    plan = load_split_items(which)
    r0_ref = gauge1_r0_logits()
    missing = [it["id"] for _, it in plan if it["id"] not in r0_ref]
    if missing:
        stop(f"{len(missing)} items have no gauge ① r0 logits to cross-check, e.g. {missing[:3]}")

    t_start = time.perf_counter()
    reader, load_s = load_engine()
    model = reader.model
    inner, text_model = _resolve(model, ku2["attr_path"])
    hidden, _src = find_hidden_size(model)
    if inner is None or hidden != ku2.get("hidden_size"):
        stop(f"inner model at {ku2['attr_path']} not found or hidden size {hidden} != KU2's {ku2.get('hidden_size')}")
    head, head_desc, _info = make_head(inner, text_model, mx)
    letter_ids = mx.array(reader.letter_ids)
    print(f"loaded engine in {load_s:.2f} s; inner {ku2['attr_path']} hidden {hidden}; head {head_desc}; "
          f"letters {dict(zip(LETTERS, reader.letter_ids))}; items {len(plan)} ({which})", flush=True)

    feats = np.empty((len(plan), hidden), dtype=np.float32)
    ids, splits, n_tokens, truncated = [], [], [], []
    r0_diffs, r0_argmax_agree = [], 0
    t0 = time.perf_counter()
    for i, (split, item) in enumerate(plan):
        prompt, trunc = render_prompt(reader, item, ROTATION)
        toks = reader._ensure_single_bos(list(reader.tok.encode(prompt)))
        h = inner(mx.array(toks)[None])
        if tuple(h.shape) != (1, len(toks), hidden):
            stop(f"hidden state of {item['id']} has shape {tuple(h.shape)}, expected (1, {len(toks)}, {hidden})")
        last = h[:, -1]
        vec = last.astype(mx.float32)
        restricted = head(last)[0][letter_ids].astype(mx.float32)
        mx.eval(vec, restricted)
        v = np.array(vec, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(v)):
            stop(f"non-finite feature on {item['id']}")
        feats[i] = v
        got = [float(x) for x in restricted.tolist()]
        ref = r0_ref[item["id"]]
        d = max(abs(a - b) for a, b in zip(got, ref))
        if d > R0_STOP_DIFF:
            stop(f"{item['id']}: head on the feature misses gauge ①'s r0 logits by {d:.3f} — not the same render")
        r0_diffs.append(d)
        r0_argmax_agree += int(int(np.argmax(got)) == int(np.argmax(ref)))
        ids.append(item["id"])
        splits.append(split)
        n_tokens.append(len(toks))
        truncated.append(bool(trunc))
        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == len(plan):
            print(f"progress {i + 1}/{len(plan)} elapsed {time.perf_counter() - t0:.1f} s; "
                  f"max |head(feature) − gauge① r0| so far {max(r0_diffs):.4f}", flush=True)
            clear_cache()
    process_s = time.perf_counter() - t0

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, feats)
    size = out_npy.stat().st_size
    index = {
        "of_gauge": GAUGE,
        "file": out_npy.name,
        "sha256": sha256_file(out_npy),
        "bytes": size,
        "shape": [int(feats.shape[0]), int(feats.shape[1])],
        "dtype": "float32",
        "splits_included": which,
        "row_order": "fit rows in fit.jsonl order, then exam rows in exam.jsonl order",
        "engine": ENGINE_NAME,
        "engine_path": ENGINE_PATH,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "rotation": ROTATION,
        "attr_path": ku2["attr_path"],
        "vector": "last position of language_model.model(tokens) = Gemma4TextModel.norm(h) (final-normed), cast to float32",
        "r0_crosscheck": {
            "what": "model's own head (" + head_desc + ") on the last-position hidden state, restricted to the "
                    "letter ids, vs gauge ①'s stored r0 logits for the same item",
            "max_abs_diff": round(max(r0_diffs), 6),
            "mean_abs_diff": round(float(np.mean(r0_diffs)), 6),
            "argmax_agree": r0_argmax_agree,
            "n": len(r0_diffs),
            "stop_if_above": R0_STOP_DIFF,
        },
        "n_truncated": int(sum(truncated)),
        "mean_prompt_tokens": round(float(np.mean(n_tokens)), 1),
        "max_prompt_tokens": int(max(n_tokens)),
        "timing": {"s_per_item": round(process_s / len(plan), 4), "process_s": round(process_s, 3),
                   "wall_s": round(time.perf_counter() - t_start, 3), "load_s": round(load_s, 3),
                   "peak_gb": None if peak_gb() is None else round(peak_gb(), 3), "forwards_per_item": 1},
        "ids": ids,
        "splits": splits,
        "prompt_tokens": n_tokens,
        "truncated": truncated,
        "written_at": now_iso(),
    }
    write_json(out_index, index)
    print(f"features {feats.shape} float32 -> {out_npy} ({size / 1e6:.1f} MB, sha256 {index['sha256'][:16]}…)")
    print(f"r0 cross-check: max |diff| {index['r0_crosscheck']['max_abs_diff']}, mean {index['r0_crosscheck']['mean_abs_diff']}, "
          f"argmax agree {r0_argmax_agree}/{len(plan)}; truncated {index['n_truncated']}; "
          f"{index['timing']['s_per_item']} s/item; peak {index['timing']['peak_gb']} GB")
    return 0


def load_features(npy: Path, index_path: Path):
    index = read_json(index_path)
    if sha256_file(npy) != index.get("sha256"):
        stop(f"{npy.name} sha256 differs from {index_path.name}")
    X = np.load(npy)
    if list(X.shape) != index["shape"] or len(index["ids"]) != X.shape[0]:
        stop(f"{npy.name} shape {X.shape} does not match its index")
    return X, index


# ----------------------------------------------------------------------------------------------
# head
# ----------------------------------------------------------------------------------------------
def make_head_model(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, l1_ratio=0.0, class_weight=None, solver="lbfgs", max_iter=MAX_ITER)),
    ])


def fit_head(X: np.ndarray, y: np.ndarray, C: float):
    """Fit one head; returns (model, n_iter, n_convergence_warnings)."""
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        m = make_head_model(C).fit(X, y)
    if list(m.classes_) != list(range(len(LABELS))):
        stop(f"head classes {list(m.classes_)} != label indices")
    n_warn = sum(1 for x in w if issubclass(x.category, ConvergenceWarning))
    return m, int(np.max(m.named_steps["lr"].n_iter_)), n_warn


def cv_select(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> dict:
    """C by stratified k-fold CV (shuffle, random_state=seed) on pooled out-of-fold NLL over C_GRID;
    returns the chosen C, its OOF decision values (same folds) and the search record."""
    from sklearn.model_selection import StratifiedKFold

    from nikasha import metrics

    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))
    fold_of = np.empty(len(y), dtype=int)
    for k, (_, te) in enumerate(folds):
        fold_of[te] = k
    nll_by_C, oof_by_C, iters_by_C, warns_by_C = [], [], [], []
    for C in C_GRID:
        oof = np.empty((len(y), len(LABELS)), dtype=np.float64)
        iters, warns = 0, 0
        for tr, te in folds:
            m, it, nw = fit_head(X[tr], y[tr], C)
            oof[te] = m.decision_function(X[te])
            iters, warns = max(iters, it), warns + nw
        nll_by_C.append(float(metrics.nll(metrics.softmax(oof), y)))
        oof_by_C.append(oof)
        iters_by_C.append(iters)
        warns_by_C.append(warns)
    best = int(np.argmin(nll_by_C))
    return {
        "C": C_GRID[best], "best_index": best, "oof": oof_by_C[best], "fold_of": fold_of,
        "n_splits": n_splits, "seed": seed,
        "nll_by_C": [round(v, 6) for v in nll_by_C],
        "max_iter_by_C": iters_by_C, "convergence_warnings_by_C": warns_by_C,
    }


def softmax_rows(L: np.ndarray) -> np.ndarray:
    from nikasha import metrics
    return metrics.softmax(L)


def split_rows(index: dict, split: str) -> np.ndarray:
    return np.array([i for i, s in enumerate(index["splits"]) if s == split], dtype=int)


def fit_labels(index: dict) -> dict[str, str]:
    """id -> label for the fit split (fit.jsonl, after the manifest sha check)."""
    from nikasha.gauge_logit import verify_manifest_sha

    verify_manifest_sha(read_json(MANIFEST_PATH), FIT_PATH, "fit.jsonl")
    return {it["id"]: it["label"] for it in read_jsonl(FIT_PATH)}


def exam_labels() -> dict[str, str]:
    """id -> label for the exam split: only the gauge-schema `label` field is copied (as gauge ① does);
    nothing here scores, compares or prints it."""
    from nikasha.gauge_logit import verify_manifest_sha

    verify_manifest_sha(read_json(MANIFEST_PATH), EXAM_PATH, "exam.jsonl")
    return {it["id"]: it["label"] for it in read_jsonl(EXAM_PATH)}


def mode_head(npy: Path, index_path: Path, dry_run: bool) -> int:
    from nikasha import metrics
    from nikasha.gauge_logit import PROMPT_VERSION, prompt_sha256, read_license_line

    X, index = load_features(npy, index_path)
    fit_idx = split_rows(index, "fit")
    exam_idx = split_rows(index, "exam")
    flab = fit_labels(index)
    fit_ids = [index["ids"][i] for i in fit_idx]
    y_fit = metrics.labels_to_idx([flab[i] for i in fit_ids])
    if len(fit_ids) != 300 or set(fit_ids) != set(flab):
        stop(f"features carry {len(fit_ids)} fit rows; the fit split has {len(flab)}")

    t0 = time.perf_counter()
    cv = cv_select(X[fit_idx], y_fit, CV_FOLDS, SEED)
    oof = cv["oof"]
    P_oof = softmax_rows(oof)
    print(f"CV ({CV_FOLDS}-fold, seed {SEED}) pooled OOF NLL by C: "
          + ", ".join(f"{c:.3g}:{v:.4f}" for c, v in zip(C_GRID, cv["nll_by_C"])))
    print(f"chosen C = {cv['C']:.6g} (grid index {cv['best_index']}); OOF fit accuracy {metrics.accuracy(P_oof, y_fit):.4f}, "
          f"OOF NLL {cv['nll_by_C'][cv['best_index']]:.4f}; convergence warnings at chosen C "
          f"{cv['convergence_warnings_by_C'][cv['best_index']]}, max n_iter {cv['max_iter_by_C'][cv['best_index']]}")
    if dry_run:
        print(f"dry run: fit split only, nothing written ({time.perf_counter() - t0:.1f} s)")
        return 0
    if len(exam_idx) != 700:
        stop(f"features carry {len(exam_idx)} exam rows, expected 700 — run `features` on all splits first")

    final, n_iter, n_warn = fit_head(X[fit_idx], y_fit, cv["C"])
    exam_dec = final.decision_function(X[exam_idx])
    process_s = time.perf_counter() - t0
    elab = exam_labels()
    exam_ids = [index["ids"][i] for i in exam_idx]

    items = []
    for j, i in enumerate(fit_idx):
        lm = [float(v) for v in oof[j]]
        items.append({"id": index["ids"][i], "split": "fit", "label": flab[index["ids"][i]], "logits_mean": lm,
                      "logits_by_rotation": [], "probs_raw": [float(p) for p in softmax_rows(np.array([lm]))[0]],
                      "truncated": index["truncated"][i], "prompt_tokens": index["prompt_tokens"][i],
                      "oof": True, "fold": int(cv["fold_of"][j])})
    for j, i in enumerate(exam_idx):
        lm = [float(v) for v in exam_dec[j]]
        items.append({"id": exam_ids[j], "split": "exam", "label": elab[exam_ids[j]], "logits_mean": lm,
                      "logits_by_rotation": [], "probs_raw": [float(p) for p in softmax_rows(np.array([lm]))[0]],
                      "truncated": index["truncated"][i], "prompt_tokens": index["prompt_tokens"][i],
                      "oof": False})
    for it in items:
        p = it["probs_raw"]
        if len(p) != 3 or any(not (0.0 <= x <= 1.0) for x in p) or abs(sum(p) - 1.0) > 1e-6:
            stop(f"gauge contract violated on {it['id']}")

    doc = {
        "gauge": GAUGE,
        "engine": ENGINE_NAME,
        "engine_path": ENGINE_PATH,
        "license": read_license_line(),
        "prompt_sha256": prompt_sha256(),
        "prompt_version": PROMPT_VERSION,
        "rotations": 1,
        "seed": SEED,
        "n_fit": len(fit_idx),
        "n_exam": len(exam_idx),
        "labels": list(LABELS),
        "prereg_section": PREREG_SECTION,
        "method": "frozen final-normed hidden state (language_model.model, last position, rotation r0, float32) "
                  "+ multinomial logistic regression (L2, lbfgs, class_weight=None) on standardised features",
        "C": cv["C"],
        "oof": True,
        "oof_note": "fit-split logits_mean are 5-fold out-of-fold decision values at the chosen C (the folds of "
                    "the C search); exam logits_mean come from the head refit on all 300 fit items",
        "cv": {
            "folds": CV_FOLDS, "splitter": f"StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, random_state={SEED})",
            "C_grid": C_GRID, "criterion": "pooled out-of-fold NLL (softmax of decision values); first minimum",
            "nll_by_C": cv["nll_by_C"], "chosen_index": cv["best_index"],
            "max_n_iter_by_C": cv["max_iter_by_C"], "convergence_warnings_by_C": cv["convergence_warnings_by_C"],
        },
        "final_head": {"n_train": len(fit_idx), "n_iter": n_iter, "convergence_warnings": n_warn,
                       "scaler": "StandardScaler fitted on the 300 fit rows", "max_iter": MAX_ITER},
        "features": {"file": f"results/{npy.name}", "sha256": index["sha256"], "index": f"results/{index_path.name}",
                     "dim": index["shape"][1], "r0_crosscheck_max_abs_diff": index["r0_crosscheck"]["max_abs_diff"]},
        "timing": {"s_per_item": index["timing"]["s_per_item"], "wall_s": index["timing"]["wall_s"],
                   "peak_gb": index["timing"]["peak_gb"], "n_items": len(items), "forwards_per_item": 1,
                   "head_s": round(process_s, 3)},
        "written_at": now_iso(),
        "items": items,
    }
    write_json(PROBE_PATH, doc)
    print(f"final head: C {cv['C']:.6g}, n_iter {n_iter}, convergence warnings {n_warn}; "
          f"wrote {PROBE_PATH} ({len(fit_idx)} OOF fit + {len(exam_idx)} exam items)")
    return 0


# ----------------------------------------------------------------------------------------------
# labels-needed
# ----------------------------------------------------------------------------------------------
def ln_draws() -> list[tuple[int, int, int]]:
    """(n, k, seed) for every variant: k = 0..4 with seed SEED + k; n = 300 is one draw (k = 0)."""
    out = []
    for n in LN_SIZES:
        for k in range(1 if n == 300 else LN_SEEDS):
            out.append((n, k, SEED + k))
    return out


def ln_subsample(y_fit: np.ndarray, n: int, seed: int) -> np.ndarray:
    """n/3 fit rows per label, drawn without replacement with numpy default_rng(seed) from that label's
    rows in fit order; returned sorted (fit order). n = 300 -> every fit row."""
    per = n // len(LABELS)
    if per * len(LABELS) != n:
        stop(f"n = {n} is not divisible by {len(LABELS)}")
    rng = np.random.default_rng(seed)
    chosen = []
    for k in range(len(LABELS)):
        rows = np.flatnonzero(y_fit == k)
        if per > len(rows):
            stop(f"n = {n} needs {per} rows of label {LABELS[k]}, fit has {len(rows)}")
        chosen.extend(rows if per == len(rows) else rng.choice(rows, size=per, replace=False))
    return np.sort(np.asarray(chosen, dtype=int))


def mode_labels_needed(npy: Path, index_path: Path, dry_run: bool) -> int:
    from nikasha import calibrate, metrics

    X, index = load_features(npy, index_path)
    fit_idx = split_rows(index, "fit")
    exam_idx = split_rows(index, "exam")
    flab = fit_labels(index)
    fit_ids = [index["ids"][i] for i in fit_idx]
    y_fit = metrics.labels_to_idx([flab[i] for i in fit_ids])
    Xf = X[fit_idx]
    if not dry_run and len(exam_idx) != 700:
        stop(f"features carry {len(exam_idx)} exam rows, expected 700")
    elab = None if dry_run else exam_labels()
    exam_ids = [index["ids"][i] for i in exam_idx]

    variants = []
    t0 = time.perf_counter()
    for n, k, seed in ln_draws():
        sub = ln_subsample(y_fit, n, seed)
        n_splits = LN_SMALL_FOLDS if n < LN_SMALL_N else CV_FOLDS
        cv = cv_select(Xf[sub], y_fit[sub], n_splits, seed)
        fit_items = [{"id": fit_ids[r], "split": "fit", "label": LABELS[int(y_fit[r])],
                      "logits_mean": [float(v) for v in cv["oof"][j]]} for j, r in enumerate(sub)]
        calib = calibrate.calibrate_gauge({"gauge": f"ln-n{n}-k{k}", "engine": ENGINE_NAME, "items": fit_items},
                                          f"ln-n{n}-k{k}")
        v = {
            "name": f"n{n}-k{k}", "n": n, "k": k, "seed": seed, "per_label": n // len(LABELS),
            "cv_folds": n_splits, "C": cv["C"], "cv_nll_by_C": cv["nll_by_C"],
            "convergence_warnings_at_C": cv["convergence_warnings_by_C"][cv["best_index"]],
            "calib": {key: calib[key] for key in ("T", "T_method", "tau_global", "taus", "target_reachable_on_fit",
                                                   "fit", "fit_fingerprint")},
            "fit_items": [{"id": it["id"], "label": it["label"], "logits_mean": it["logits_mean"]} for it in fit_items],
        }
        line = (f"n={n:3d} k={k} seed={seed}: folds {n_splits} C {cv['C']:.3g} OOF acc "
                f"{metrics.accuracy(metrics.softmax(cv['oof']), y_fit[sub]):.3f} T {calib['T']:.3f} taus {calib['taus']} "
                f"fit ask {calib['fit']['ask_rate']} SA {calib['fit']['selective_accuracy']}")
        if not dry_run:
            final, n_iter, n_warn = fit_head(Xf[sub], y_fit[sub], cv["C"])
            dec = final.decision_function(X[exam_idx])
            v["final_head"] = {"n_iter": n_iter, "convergence_warnings": n_warn}
            v["exam_items"] = [{"id": exam_ids[j], "label": elab[exam_ids[j]],
                                "logits_mean": [float(x) for x in dec[j]]} for j in range(len(exam_ids))]
        variants.append(v)
        print(line, flush=True)
    if dry_run:
        print(f"dry run: fit split only, {len(variants)} variants, nothing written ({time.perf_counter() - t0:.1f} s)")
        return 0
    doc = {
        "what": "labels-needed curve: gauge ③ head variants trained on n fit labels (equal per class)",
        "prereg_section": PREREG_SECTION,
        "sizes": LN_SIZES, "seeds": [SEED + k for k in range(LN_SEEDS)], "n300_draws": 1,
        "subsample": "per label n/3 rows drawn without replacement by numpy default_rng(seed) from that label's "
                     "fit rows (fit order), sorted; n = 300 = the whole fit split",
        "cv": f"StratifiedKFold(shuffle=True, random_state=seed), {LN_SMALL_FOLDS}-fold if n < {LN_SMALL_N} else "
              f"{CV_FOLDS}-fold; C grid logspace(-4, 2, 13); pooled OOF NLL, first minimum",
        "calibration": "calibrate.calibrate_gauge on the variant's OOF fit logits (T grid+refine, per-class taus at "
                       "SA target on those OOF predictions)",
        "head": "StandardScaler + LogisticRegression(l1_ratio=0, lbfgs, class_weight=None), refit on the n rows",
        "features": {"file": f"results/{npy.name}", "sha256": index["sha256"]},
        "n_exam": len(exam_idx),
        "written_at": now_iso(),
        "variants": variants,
    }
    write_json(LABELS_NEEDED_PATH, doc)
    print(f"wrote {LABELS_NEEDED_PATH} ({len(variants)} variants, {time.perf_counter() - t0:.1f} s); "
          f"score with: score.py --labels-needed")
    return 0


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nikasha.gauge_probe",
                                 description="gauge 3: frozen hidden state + linear head; labels-needed head variants")
    sub = ap.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("features", help="extract last-position final-normed hidden states (r0, prompt v1)")
    f.add_argument("--splits", choices=("all", "fit"), default="all")
    f.add_argument("--out-npy", type=Path, default=FEATURES_NPY)
    f.add_argument("--out-index", type=Path, default=FEATURES_INDEX)
    for name in ("head", "labels-needed"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true", help="fit split only; print fit numbers; write nothing")
        p.add_argument("--features", type=Path, default=FEATURES_NPY)
        p.add_argument("--index", type=Path, default=FEATURES_INDEX)
    args = ap.parse_args(argv)
    if args.mode == "features":
        if args.splits == "all" and (args.out_npy != FEATURES_NPY or args.out_index != FEATURES_INDEX):
            ap.error("--splits all writes the canonical results/gauge3-features.* only")
        return mode_features(args.splits, args.out_npy, args.out_index)
    if args.mode == "head":
        return mode_head(args.features, args.index, args.dry_run)
    return mode_labels_needed(args.features, args.index, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
