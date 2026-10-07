"""score.py — exam metrics for every gauge: bootstrap CIs, abstain numbers, curve points, curve.png.

The exam split is read ONCE per gauge version (brief rule 8). This module is the only code that opens
data/seta/exam.jsonl (rule 4); it keeps only ids and labels from it and never prints item text. It reads
results JSON and never imports an engine. It refuses to score a gauge whose .calib.json fit_fingerprint
(sha256 over sorted fit ids + logits_mean) differs from the gauge file's, i.e. stale T / thresholds.

Usage (from the repo root, inside the venv):
  $PY -m nikasha.score                  # every results/*.json with "items" + a .calib.json and no "exam" block
  $PY -m nikasha.score --gauge NAME     # one gauge: results/NAME.json
  $PY -m nikasha.score --rescore        # also overwrite existing exam blocks (scoring-code bug fixes only; log it)
  $PY -m nikasha.score --fit-dry-run    # the identical metric code on the FIT split; prints numbers, writes nothing
  $PY -m nikasha.score --curve-only     # only redraw results/curve.png from the exam blocks on disk; exam not opened
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nikasha import (
    ECE_BINS,
    EXAM_PATH,
    FIT_PATH,
    LABELS,
    MANIFEST_PATH,
    N_BOOT,
    RESULTS_DIR,
    ROOT,
    SA_TARGET,
    SEED,
    fit_fingerprint,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from nikasha import metrics

CURVE_POINTS = 50
CURVE_PNG = RESULTS_DIR / "curve.png"
PREREG_PATH = ROOT / "PREREG.md"
METRIC_ORDER = [
    "accuracy", "macro_f1", "nll_raw", "nll_cal", "ece_raw", "ece_cal",
    "ask_rate", "selective_accuracy", "auroc",
]
# Categorical palette in a fixed slot order (adjacent pairs validated for CVD and normal-vision
# separation on a light surface). Colour follows the gauge's position in sorted file order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e5e4e0"
DASH = "—"


# ----------------------------------------------------------------------------- small helpers

def _fail(msg: str):
    print(f"score.py: STOP — {msg}", file=sys.stderr)
    sys.exit(2)


def _prereg_commit_or_fail() -> str:
    """Rule 9: PREREG.md must be committed and clean (no uncommitted edits) before the exam is opened.
    Returns the ISO date of its last commit, which is recorded in the exam block."""
    import subprocess
    try:
        log = subprocess.run(["git", "log", "-1", "--format=%cI", "--", "PREREG.md"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
        st = subprocess.run(["git", "status", "--porcelain", "--", "PREREG.md"], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:  # noqa: BLE001
        _fail(f"could not query git for PREREG.md ({type(e).__name__}) — rule 9 cannot be verified")
    if log.returncode != 0 or not log.stdout.strip():
        _fail("PREREG.md has no commit — commit and push it before the first exam read (rule 9)")
    if st.returncode != 0 or st.stdout.strip():
        _fail("PREREG.md has uncommitted changes — commit and push it before the first exam read (rule 9)")
    return log.stdout.strip()


def _prereg_section_or_fail(section: str) -> str:
    """Rule 12 (brief 10): a gauge that declares `prereg_section` is scored only when that heading is a line
    of the COMMITTED PREREG.md (HEAD), the working copy is clean (checked by _prereg_commit_or_fail), and the
    commit that introduced the heading is on origin/main, i.e. pushed. Returns that commit's ISO date."""
    import subprocess
    try:
        head = subprocess.run(["git", "show", "HEAD:PREREG.md"], cwd=ROOT, capture_output=True, text=True, timeout=30)
        intro = subprocess.run(["git", "log", "--format=%H %cI", "-S", section, "--", "PREREG.md"], cwd=ROOT,
                               capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:  # noqa: BLE001
        _fail(f"could not query git for {section!r} ({type(e).__name__})")
    if head.returncode != 0 or section not in head.stdout.splitlines():
        _fail(f"the committed PREREG.md has no line {section!r} — amend, commit and push it before this exam read")
    lines = [ln.split() for ln in intro.stdout.splitlines() if ln.strip()]
    if intro.returncode != 0 or not lines:
        _fail(f"cannot find the commit that introduced {section!r} in PREREG.md")
    sha, date = lines[-1][0], lines[-1][1]  # oldest commit whose diff adds the heading
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], cwd=ROOT,
                         capture_output=True, text=True, timeout=30)
    if anc.returncode != 0:
        _fail(f"{section!r} (commit {sha[:7]}) is not on origin/main — push it before this exam read (rule 12)")
    return date


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _calib_path(result_path: Path) -> Path:
    return result_path.with_name(result_path.stem + ".calib.json")


def _is_gauge_result(obj) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("gauge"), str) and isinstance(obj.get("items"), list)


def _gauge_results() -> list[tuple[Path, dict]]:
    """Every results/*.json that is a gauge result (a "gauge" string and an "items" list), sorted by name."""
    out = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        if p.name.endswith(".calib.json"):
            continue
        try:
            obj = read_json(p)
        except Exception as e:  # noqa: BLE001 — a non-gauge or unreadable file is simply not a target
            print(f"score.py: note — {p.name} is not readable JSON ({e.__class__.__name__}); skipped", file=sys.stderr)
            continue
        if _is_gauge_result(obj):
            out.append((p, obj))
    return out


def _load_truth(path: Path) -> dict[str, str]:
    """id -> label for a split file. Only ids and labels are kept; item text is never retained or printed."""
    if not path.exists():
        _fail(f"{path.name} not found at {path}")
    truth: dict[str, str] = {}
    for row in read_jsonl(path):
        iid, label = row.get("id"), row.get("label")
        if not isinstance(iid, str) or iid in truth:
            _fail(f"{path.name}: missing or duplicate id after {len(truth)} rows")
        if label not in LABELS:
            _fail(f"{path.name}: id {iid} carries a label outside the fixed label list")
        truth[iid] = label
    if not truth:
        _fail(f"{path.name} is empty")
    return truth


def _rows_of(gauge: str, items: list, split: str) -> list[dict]:
    """The gauge's items of one split, in file order. Needs no split file, so it runs before the exam
    is opened; fails on an empty split or duplicate ids."""
    rows = [it for it in items if isinstance(it, dict) and it.get("split") == split]
    if not rows:
        _fail(f"{gauge}: no items with split == {split!r}")
    ids = [it.get("id") for it in rows]
    if len(set(ids)) != len(ids):
        _fail(f"{gauge}: duplicate ids in the {split} split (n={len(ids)}, unique={len(set(ids))})")
    return rows


def _check_split(gauge: str, rows: list[dict], split: str, truth: dict[str, str], subset_ok: bool) -> None:
    """Rows of one split checked against the split file: same ids (or a subset when the manifest says
    exam_reduced) and the identical label per id."""
    ids = [it.get("id") for it in rows]
    got, want = set(ids), set(truth)
    extra = sorted(i for i in got - want if isinstance(i, str))
    if extra or len(got - want) != len(extra):
        _fail(f"{gauge}: {len(got - want)} {split} ids are not in {split}.jsonl, e.g. {extra[:5]}")
    missing = sorted(want - got)
    if missing and not subset_ok:
        _fail(f"{gauge}: {len(missing)} ids of {split}.jsonl are absent from the gauge, e.g. {missing[:5]}")
    bad = [i for i, it in zip(ids, rows) if it.get("label") != truth[i]]
    if bad:
        _fail(f"{gauge}: {len(bad)} {split} items whose label differs from {split}.jsonl, e.g. {bad[:5]}")


def _arrays(gauge: str, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    try:
        L = np.asarray([it["logits_mean"] for it in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as e:
        _fail(f"{gauge}: logits_mean missing or ragged ({e.__class__.__name__})")
    if L.ndim != 2 or L.shape[1] != len(LABELS):
        _fail(f"{gauge}: logits_mean has shape {tuple(L.shape)}, expected (n, {len(LABELS)})")
    if np.isnan(L).any() or np.isposinf(L).any():
        _fail(f"{gauge}: logits_mean contains NaN or +inf")
    y = np.asarray(metrics.labels_to_idx([it["label"] for it in rows]), dtype=int)
    return L, y


def _validate_probs(gauge: str, P: np.ndarray, what: str) -> None:
    """The gauge contract, per item: length 3, every entry in [0, 1], sum within 1e-6 of 1."""
    ok = (
        P.ndim == 2 and P.shape[1] == len(LABELS)
        and bool(np.isfinite(P).all()) and bool((P >= 0).all()) and bool((P <= 1).all())
        and bool((np.abs(P.sum(axis=1) - 1.0) <= 1e-6).all())
    )
    if not ok:
        _fail(f"{gauge}: {what} probabilities violate the gauge contract (n={len(P)})")


def _read_calib(gauge: str, path: Path) -> tuple[float, list[float], float | None]:
    if not path.exists():
        _fail(f"{gauge}: {path.name} not found — run calibrate.py first")
    c = read_json(path)
    if c.get("gauge") not in (None, gauge):
        _fail(f"{gauge}: {path.name} belongs to gauge {c.get('gauge')!r}")
    if c.get("sa_target") is not None and abs(float(c["sa_target"]) - SA_TARGET) > 1e-12:
        _fail(f"{gauge}: {path.name} sa_target {c['sa_target']} != pre-registered {SA_TARGET}")
    try:
        T = float(c["T"])
        taus = [float(t) for t in c["taus"]]
    except (KeyError, TypeError, ValueError):
        _fail(f"{gauge}: {path.name} lacks a numeric T and a taus list")
    if not (np.isfinite(T) and T > 0):
        _fail(f"{gauge}: T must be a positive finite number, got {T}")
    if len(taus) != len(LABELS) or not all(np.isfinite(t) for t in taus):
        _fail(f"{gauge}: taus must be {len(LABELS)} finite numbers, got {taus}")
    tg = c.get("tau_global")
    return T, taus, (None if tg is None else float(tg))


def _check_fingerprint(gauge: str, data: dict, calib_path: Path) -> str:
    """The .calib.json must have been fitted on THIS gauge run: its fit_fingerprint (sha256 over the
    sorted fit ids + their logits_mean) must equal the gauge file's. Refuses when absent or different."""
    stored = (read_json(calib_path).get("fit_fingerprint") or {}).get("sha256")
    if not isinstance(stored, str) or not stored:
        _fail(f"{gauge}: {calib_path.name} carries no fit_fingerprint — cannot prove it belongs to this gauge "
              f"run; run calibrate.py (or calibrate.py --stamp for a calib fitted before fingerprints existed)")
    actual = fit_fingerprint(data["items"])["sha256"]
    if actual != stored:
        _fail(f"{gauge}: fit fingerprint {actual[:16]}… of results/{gauge}.json differs from {calib_path.name}'s "
              f"{stored[:16]}… — the gauge was re-run after calibration; re-calibrate before scoring")
    return actual


# ----------------------------------------------------------------------------- the metric code

def compute(gauge: str, L: np.ndarray, y: np.ndarray, T: float, taus: list[float],
            n_boot: int = N_BOOT, seed: int = SEED) -> tuple[dict, list[dict], list[list[int]]]:
    """Every exam number for one gauge from its mean restricted logits. The real run and
    --fit-dry-run call this same function; only the index set differs."""
    n = int(len(y))
    P_raw = metrics.softmax(L)
    P_cal = metrics.softmax(L, T)
    _validate_probs(gauge, P_raw, "raw")
    _validate_probs(gauge, P_cal, "calibrated")

    fns = {
        "accuracy": lambda idx: metrics.accuracy(P_cal[idx], y[idx]),
        "macro_f1": lambda idx: metrics.macro_f1(P_cal[idx], y[idx]),
        "nll_raw": lambda idx: metrics.nll(P_raw[idx], y[idx]),
        "nll_cal": lambda idx: metrics.nll(P_cal[idx], y[idx]),
        "ece_raw": lambda idx: metrics.ece(P_raw[idx], y[idx], ECE_BINS),
        "ece_cal": lambda idx: metrics.ece(P_cal[idx], y[idx], ECE_BINS),
        "ask_rate": lambda idx: metrics.ask_rate(P_cal[idx], taus),
        "selective_accuracy": lambda idx: metrics.selective_accuracy(P_cal[idx], y[idx], taus),
        "auroc": lambda idx: metrics.auroc(P_cal[idx], y[idx]),
    }
    boot = metrics.bootstrap(fns, n, n_boot=n_boot, seed=seed)
    result = {name: dict(boot[name]) for name in METRIC_ORDER}
    _, answered = metrics.abstain(P_cal, taus)
    result["selective_accuracy"]["n_answered"] = int(np.count_nonzero(answered))

    curve = metrics.curve(P_cal, y, CURVE_POINTS)
    conf = metrics.confusion(P_cal, y)
    return result, curve, conf


def _num(v, scale: float, nd: int) -> str:
    return DASH if v is None else f"{float(v) * scale:.{nd}f}"


def _with_ci(m: dict, scale: float, nd: int) -> str:
    v, ci = m.get("value"), m.get("ci")
    if v is None:
        return DASH
    if not (isinstance(ci, (list, tuple)) and len(ci) == 2) or ci[0] is None or ci[1] is None:
        return f"{_num(v, scale, nd)} [{DASH}, {DASH}]"
    return f"{_num(v, scale, nd)} [{_num(ci[0], scale, nd)}, {_num(ci[1], scale, nd)}]"


def _arrow(raw: dict, cal: dict, nd: int) -> str:
    return f"{_num(raw.get('value'), 1, nd)} → {_num(cal.get('value'), 1, nd)}"


def display_strings(m: dict) -> dict:
    """What README prints verbatim: percent metrics as "71.2 [68.0, 74.4]", ECE/NLL as "raw → cal",
    AUROC with three decimals; any None shows as a dash."""
    return {
        "accuracy": _with_ci(m["accuracy"], 100, 1),
        "macro_f1": _with_ci(m["macro_f1"], 100, 1),
        "nll": _arrow(m["nll_raw"], m["nll_cal"], 3),
        "ece": _arrow(m["ece_raw"], m["ece_cal"], 3),
        "ask_rate": _with_ci(m["ask_rate"], 100, 1),
        "selective_accuracy": _with_ci(m["selective_accuracy"], 100, 1),
        "auroc": _with_ci(m["auroc"], 1, 3),
    }


def summary_line(gauge: str, split: str, n: int, T: float, taus: list[float], m: dict, d: dict) -> str:
    return (
        f"{gauge} [{split}] n={n} T={T:.4f} taus={[round(t, 4) for t in taus]} "
        f"acc={d['accuracy']} f1={d['macro_f1']} nll={d['nll']} ece={d['ece']} "
        f"ask={d['ask_rate']} sa={d['selective_accuracy']} n_answered={m['selective_accuracy']['n_answered']} "
        f"auroc={d['auroc']}"
    )


# ----------------------------------------------------------------------------- curve.png

def _mpl_config_dir() -> Path:
    """Where matplotlib may write its font cache. Its default (~/.matplotlib) is outside the directories
    rule 7 allows writes to; the venv (~/venvs/…) is inside them and outside the repo, so nothing lands
    in git. Only when not running from a venv (a rule-1 breach) does it fall back to the repo root."""
    base = Path(sys.prefix) if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else ROOT
    return base / ".mplconfig"


def draw_curve(out_path: Path = CURVE_PNG) -> bool:
    """Selective accuracy vs ask rate (global τ sweep) for every gauge with an exam block."""
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir()))  # before the first matplotlib import
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    series = []
    for _, data in _gauge_results():
        ex = data.get("exam")
        if isinstance(ex, dict) and isinstance(ex.get("curve"), list) and ex["curve"]:
            mm = ex.get("metrics", {})
            series.append((
                data["gauge"], ex["curve"],
                mm.get("ask_rate", {}).get("value"), mm.get("selective_accuracy", {}).get("value"),
            ))
    if not series:
        print("curve.png: no gauge has an exam block yet; not drawn")
        return False

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for i, (name, pts, ask_at_taus, sa_at_taus) in enumerate(series):
        colour = PALETTE[i] if i < len(PALETTE) else INK_MUTED
        x = np.asarray([p.get("ask_rate") for p in pts], dtype=float)
        yv = np.asarray([np.nan if p.get("selective_accuracy") is None else p["selective_accuracy"] for p in pts],
                        dtype=float)
        finite = np.isfinite(x) & np.isfinite(yv)
        # Decide line vs marker on DISTINCT finite points (the stored values are already rounded to 4 dp):
        # a constant gauge collapses to one point — the one-hot majority baseline has max p == 1.0 exactly,
        # so every τ gives the same finite pair — and a line through 50 coincident vertices is invisible.
        distinct = sorted({(float(a), float(b)) for a, b in zip(x[finite], yv[finite])})
        if len(distinct) >= 2:
            ax.plot(x, yv, color=colour, linewidth=1.8, label=name, zorder=3)
        else:
            ax.plot([a for a, _ in distinct], [b for _, b in distinct], color=colour, linestyle="none",
                    marker="s", markersize=6, label=name, zorder=3)
        if ask_at_taus is not None and sa_at_taus is not None:
            ax.plot([ask_at_taus], [sa_at_taus], color=colour, linestyle="none", marker="o", markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=4)

    ax.axhline(SA_TARGET, color=INK_MUTED, linestyle="--", linewidth=1.0, zorder=2,
               label=f"selective-accuracy target {SA_TARGET:.2f}")
    ax.set_xlabel("ask rate (abstained / n)", color=INK)
    ax.set_ylabel("selective accuracy (accuracy on answered items)", color=INK)
    ax.set_title("set A exam — selective accuracy vs ask rate, global τ sweep", color=INK, fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK_MUTED, linestyle="none", marker="o", markersize=7,
                          markeredgecolor="white", markeredgewidth=1.5))
    labels.append("operating point at the fitted per-class thresholds")
    ax.legend(handles, labels, loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"curve.png: {len(series)} gauge(s) drawn -> {out_path}")
    return True


# ----------------------------------------------------------------------------- target selection

def _select(args) -> list[tuple[Path, dict]]:
    """Which results files to score. --gauge names one; otherwise every gauge result that has a
    .calib.json and (unless --rescore / --fit-dry-run) no exam block yet."""
    if args.gauge:
        name = args.gauge[:-5] if args.gauge.endswith(".json") else args.gauge
        if "/" in name or name.endswith(".calib"):
            _fail(f"--gauge expects a bare gauge name, got {args.gauge!r}")
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            _fail(f"{path} not found")
        data = read_json(path)
        if not _is_gauge_result(data):
            _fail(f"{path.name} is not a gauge result (needs a 'gauge' string and an 'items' list)")
        if not _calib_path(path).exists():
            _fail(f"{_calib_path(path).name} not found — run calibrate.py first")
        if "exam" in data and not args.fit_dry_run and not args.rescore:
            _fail(f"{name} already has an exam block (scored_at {data['exam'].get('scored_at')}); "
                  f"the exam is read once per gauge version (rule 8). --rescore overrides, for scoring-code "
                  f"bug fixes only — log it in ops/COMMAND_LOG.md")
        return [(path, data)]
    targets = []
    for path, data in _gauge_results():
        if not _calib_path(path).exists():
            continue
        if "exam" in data and not (args.rescore or args.fit_dry_run):
            continue
        if not any(isinstance(it, dict) and it.get("split") == "exam" for it in data["items"]):
            # a fit-only file in the gauge schema (e.g. a timing run calibrated by hand) has nothing to
            # score; naming it with --gauge still fails loudly
            print(f"score.py: note — {path.name} has no items with split == 'exam'; skipped", file=sys.stderr)
            continue
        targets.append((path, data))
    return targets


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m nikasha.score",
        description="Score gauges on the sealed exam split: bootstrap CIs, abstain numbers, curve, curve.png. "
                    "The exam is read once per gauge version.",
    )
    ap.add_argument("--gauge", metavar="NAME", help="score one gauge: results/NAME.json (default: every gauge "
                    "result with a .calib.json and no exam block)")
    ap.add_argument("--rescore", action="store_true",
                    help="overwrite existing exam blocks (loud warning; scoring-code bug fixes only, logged)")
    ap.add_argument("--fit-dry-run", action="store_true",
                    help="run the identical metric code on the FIT split, print the numbers, write nothing")
    ap.add_argument("--curve-only", action="store_true",
                    help="only redraw results/curve.png from the exam blocks already on disk; scores nothing "
                         "and never opens exam.jsonl")
    args = ap.parse_args(argv)
    if args.fit_dry_run and args.rescore:
        ap.error("--fit-dry-run writes nothing; --rescore is meaningless with it")
    if args.curve_only:
        if args.gauge or args.rescore or args.fit_dry_run:
            ap.error("--curve-only takes no other option: it redraws from every exam block already on disk")
        return 0 if draw_curve(CURVE_PNG) else 1

    targets = _select(args)
    if not targets:
        print("score.py: nothing to do — no gauge result with a .calib.json and no exam block "
              "(use --gauge NAME, --rescore or --fit-dry-run; --curve-only redraws curve.png without scoring)")
        return 0

    # ---- design check on the fit split (rule 8): same code, nothing written -------------------
    if args.fit_dry_run:
        if MANIFEST_PATH.exists():
            expected_fit = (read_json(MANIFEST_PATH).get("sha256") or {}).get("fit.jsonl")
            if expected_fit and sha256_file(FIT_PATH) != expected_fit:
                _fail("fit.jsonl sha256 differs from the manifest")
        truth = _load_truth(FIT_PATH)
        print(f"fit-dry-run: fit.jsonl ids={len(truth)}; identical metric code, nothing is written")
        for path, data in targets:
            gauge = data["gauge"]
            rows = _rows_of(gauge, data["items"], "fit")
            _check_split(gauge, rows, "fit", truth, subset_ok=False)
            L, y = _arrays(gauge, rows)
            T, taus, _ = _read_calib(gauge, _calib_path(path))
            _check_fingerprint(gauge, data, _calib_path(path))
            m, _, conf = compute(gauge, L, y, T, taus)
            d = display_strings(m)
            print(summary_line(gauge, "fit-dry-run", len(rows), T, taus, m, d))
            print(f"  confusion (rows gold, cols pred, label order): {conf}")
        return 0

    # ---- phase 1: every check that needs no exam data, for EVERY target, before the sealed file is
    # opened and before anything is written — a misconfigured calib or a ragged gauge stops the run
    # with the exam untouched and no gauge half-scored
    prepared = []
    for path, data in targets:
        gauge = data["gauge"]
        T, taus, tau_global = _read_calib(gauge, _calib_path(path))
        fingerprint = _check_fingerprint(gauge, data, _calib_path(path))
        rows = _rows_of(gauge, data["items"], "exam")
        if isinstance(data.get("n_exam"), int) and data["n_exam"] != len(rows):
            _fail(f"{gauge}: n_exam={data['n_exam']} but {len(rows)} items carry split == 'exam'")
        L, y = _arrays(gauge, rows)
        _validate_probs(gauge, metrics.softmax(L), "raw")
        _validate_probs(gauge, metrics.softmax(L, T), "calibrated")
        prepared.append((path, data, rows, L, y, T, taus, tau_global, fingerprint))

    # ---- phase 2: the exam is opened here and nowhere else ------------------------------------
    if not PREREG_PATH.exists():
        _fail("PREREG.md is missing — pre-register before the first exam read (rule 9)")
    prereg_commit = _prereg_commit_or_fail()
    section_commit = {}
    for _path, data, *_ in prepared:
        section = data.get("prereg_section")
        if isinstance(section, str) and section and section not in section_commit:
            section_commit[section] = _prereg_section_or_fail(section)
    if not MANIFEST_PATH.exists():
        _fail(f"{MANIFEST_PATH} not found — build set A first")
    manifest = read_json(MANIFEST_PATH)
    expected = (manifest.get("sha256") or {}).get("exam.jsonl")
    if not expected:
        _fail("manifest has no sha256 for exam.jsonl")
    if not EXAM_PATH.exists():
        _fail(f"{EXAM_PATH} not found")
    actual = sha256_file(EXAM_PATH)
    if actual != expected:
        _fail(f"exam.jsonl sha256 {actual} != manifest {expected} — the sealed exam changed; stop")
    exam_reduced = bool(manifest.get("exam_reduced", False))
    truth = _load_truth(EXAM_PATH)
    declared = ((manifest.get("counts") or {}).get("exam") or {}).get("total")
    if isinstance(declared, int) and declared != len(truth):
        _fail(f"exam.jsonl has {len(truth)} ids but the manifest declares {declared}")
    print(f"exam.jsonl sha256 verified ({actual[:16]}…); ids={len(truth)}; exam_reduced={exam_reduced}; "
          f"n_boot={N_BOOT}; seed={SEED}; ece_bins={ECE_BINS}; sa_target={SA_TARGET}")

    # every target's exam ids and labels are checked against exam.jsonl before the first write
    for path, data, rows, *_ in prepared:
        _check_split(data["gauge"], rows, "exam", truth, subset_ok=exam_reduced)

    # ---- phase 3: compute and write; curve.png is redrawn from whatever is on disk even if a later
    # gauge stops the run (--curve-only redraws it at any time without opening the exam)
    scored = 0
    try:
        for path, data, rows, L, y, T, taus, tau_global, fingerprint in prepared:
            gauge = data["gauge"]
            if "exam" in data:
                print(f"!!! WARNING: --rescore is OVERWRITING the exam block of {gauge} "
                      f"(previously scored_at {data['exam'].get('scored_at')}). The exam is read once per gauge "
                      f"version (rule 8); this is allowed only for scoring-code bug fixes — log the reason. !!!")
            m, curve, conf = compute(gauge, L, y, T, taus)
            d = display_strings(m)
            data["exam"] = {
                "n_exam": len(rows),
                "n_boot": N_BOOT,
                "ci_level": 95,
                "seed": SEED,
                "ece_bins": ECE_BINS,
                "T": T,
                "taus": taus,
                "tau_global": tau_global,
                "exam_sha256": actual,
                "fit_fingerprint": fingerprint,
                "scored_at": _utc_now(),
                "prereg_commit": prereg_commit,
                **({"prereg_section": data["prereg_section"],
                    "prereg_section_commit": section_commit[data["prereg_section"]]}
                   if data.get("prereg_section") in section_commit else {}),
                "metrics": m,
                "curve": curve,
                "confusion": conf,
                "display": d,
            }
            write_json(path, data)
            scored += 1
            print(summary_line(gauge, "exam", len(rows), T, taus, m, d))
    finally:
        if scored:
            draw_curve(CURVE_PNG)
    print(f"score.py: {scored} gauge(s) scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
