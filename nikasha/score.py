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
            label = data["gauge"] if data.get("external") is not True else f"{data['gauge']} (n={ex.get('n_exam')})"
            series.append((
                label, ex["curve"],
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
    # lower right: the constant baselines sit at ask rate 0 / selective accuracy 1/3, under a lower-left legend
    ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=8)
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
        if data.get("external") is True:
            _fail(f"{name} is an external gauge with no fit split; it is scored by --like-for-like")
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
        if data.get("external") is True:
            print(f"score.py: note — {path.name} is external (no fit split); scored by --like-for-like", file=sys.stderr)
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


# ----------------------------------------------------------------------------- labels-needed (Task C)

LABELS_NEEDED_PATH = RESULTS_DIR / "labels-needed.json"
LABELS_NEEDED_PNG = RESULTS_DIR / "labels-needed.png"
GAUGE1_PATH = RESULTS_DIR / "gauge1-logit.json"


def _pct(v) -> str:
    return _num(v, 100, 1)


def _open_exam_for(section: str | None) -> tuple[str, str, dict | None, dict[str, str]]:
    """The exam is opened here for the Gate 4 modes, after the same guards as the main run: PREREG committed
    and clean, the declared amendment heading committed and pushed, exam sha256 equal to the manifest's.
    Returns (prereg_commit, exam_sha, section info, id -> label)."""
    if not PREREG_PATH.exists():
        _fail("PREREG.md is missing — pre-register before the first exam read (rule 9)")
    prereg_commit = _prereg_commit_or_fail()
    info = None
    if section:
        info = {"prereg_section": section, "prereg_section_commit": _prereg_section_or_fail(section)}
    manifest = read_json(MANIFEST_PATH)
    expected = (manifest.get("sha256") or {}).get("exam.jsonl")
    actual = sha256_file(EXAM_PATH)
    if not expected or actual != expected:
        _fail(f"exam.jsonl sha256 {actual} != manifest {expected} — the sealed exam changed; stop")
    truth = _load_truth(EXAM_PATH)
    print(f"exam.jsonl sha256 verified ({actual[:16]}…); ids={len(truth)}; n_boot={N_BOOT}; seed={SEED}")
    return prereg_commit, actual, info, truth


def score_labels_needed(rescore: bool) -> int:
    """Score every labels-needed variant once on the exam, then the per-n summary and the pre-registered
    'smallest n within noise of gauge ①'s or better' rule (PREREG Amendment 1)."""
    if not LABELS_NEEDED_PATH.exists():
        _fail(f"{LABELS_NEEDED_PATH.name} not found — run `gauge_probe labels-needed` first")
    doc = read_json(LABELS_NEEDED_PATH)
    if "exam" in doc and not rescore:
        _fail(f"{LABELS_NEEDED_PATH.name} is already scored (scored_at {doc['exam'].get('scored_at')}); "
              f"each variant is read once (rule 8)")
    variants = doc.get("variants") or []
    if len(variants) != 21:
        _fail(f"{LABELS_NEEDED_PATH.name} holds {len(variants)} variants, Amendment 1 fixes 21")
    g1 = read_json(GAUGE1_PATH)
    ref_ask = ((g1.get("exam") or {}).get("metrics") or {}).get("ask_rate") or {}
    if not (isinstance(ref_ask.get("ci"), list) and len(ref_ask["ci"]) == 2):
        _fail("gauge ①'s exam block carries no ask-rate CI to compare against")

    # phase 1 — no exam data: fingerprints, arrays, contract, one common exam order
    prepared = []
    order = None
    for v in variants:
        name = f"labels-needed {v['name']}"
        fit_items = [{"id": it["id"], "split": "fit", "logits_mean": it["logits_mean"]} for it in v["fit_items"]]
        if fit_fingerprint(fit_items)["sha256"] != v["calib"]["fit_fingerprint"]["sha256"]:
            _fail(f"{name}: fit fingerprint differs from its calibration")
        rows = [dict(it, split="exam") for it in v["exam_items"]]
        ids = [r["id"] for r in rows]
        if order is None:
            order = ids
        elif ids != order:
            _fail(f"{name}: exam item order differs from the first variant's")
        L, y = _arrays(name, rows)
        T = float(v["calib"]["T"])
        taus = [float(t) for t in v["calib"]["taus"]]
        _validate_probs(name, metrics.softmax(L), "raw")
        _validate_probs(name, metrics.softmax(L, T), "calibrated")
        prepared.append((v, name, rows, L, y, T, taus))

    # phase 2 — the exam is opened here
    prereg_commit, exam_sha, section_info, truth = _open_exam_for(doc.get("prereg_section"))
    if order != list(truth):
        _fail("labels-needed exam rows are not in exam.jsonl order")
    for v, name, rows, *_ in prepared:
        _check_split(name, rows, "exam", truth, subset_ok=False)

    # phase 3 — every variant, then the paired draw-mean CIs (the same resamples as every exam bootstrap)
    y_all = prepared[0][4]
    P_by = {}
    for v, name, rows, L, y, T, taus in prepared:
        m, _curve, conf = compute(name, L, y, T, taus)
        v["exam"] = {"n_exam": len(rows), "T": T, "taus": taus, "metrics": m, "confusion": conf,
                     "display": display_strings(m)}
        P_by[v["name"]] = (metrics.softmax(L, T), taus)
        print(f"{v['name']:>8}: acc {v['exam']['display']['accuracy']}  ask {v['exam']['display']['ask_rate']}  "
              f"sa {v['exam']['display']['selective_accuracy']}  (taus {taus})")
    sizes = doc["sizes"]
    by_n = {n: [v for v, *_ in prepared if v["n"] == n] for n in sizes}
    fns = {}
    for n in sizes:
        names = [v["name"] for v in by_n[n]]
        fns[f"ask_n{n}"] = (lambda idx, names=names: float(np.mean(
            [metrics.ask_rate(P_by[nm][0][idx], P_by[nm][1]) for nm in names])))
        fns[f"acc_n{n}"] = (lambda idx, names=names: float(np.mean(
            [metrics.accuracy(P_by[nm][0][idx], y_all[idx]) for nm in names])))
    boot = metrics.bootstrap(fns, len(y_all), n_boot=N_BOOT, seed=SEED)
    lo_ref, hi_ref = ref_ask["ci"]
    summary, smallest = [], None
    for n in sizes:
        acc = [v["exam"]["metrics"]["accuracy"]["value"] for v in by_n[n]]
        ask = [v["exam"]["metrics"]["ask_rate"]["value"] for v in by_n[n]]
        ask_ci = boot[f"ask_n{n}"]["ci"]
        acc_ci = boot[f"acc_n{n}"]["ci"]
        qualifies = bool(ask_ci is not None and ask_ci[0] <= hi_ref)
        if qualifies and smallest is None:
            smallest = n
        row = {
            "n": n, "per_label": n // len(LABELS), "draws": len(by_n[n]),
            "accuracy": {"mean": boot[f"acc_n{n}"]["value"], "min": min(acc), "max": max(acc), "ci": acc_ci},
            "ask_rate": {"mean": boot[f"ask_n{n}"]["value"], "min": min(ask), "max": max(ask), "ci": ask_ci},
            "within_noise_or_better_than_gauge1": qualifies,
        }
        rng_txt = lambda d: (f"{_pct(d['mean'])}" + (f" ({_pct(d['min'])}–{_pct(d['max'])})" if row["draws"] > 1 else "")
                             + (f" [{_pct(d['ci'][0])}, {_pct(d['ci'][1])}]" if d["ci"] else ""))
        row["display"] = {"accuracy": rng_txt(row["accuracy"]), "ask_rate": rng_txt(row["ask_rate"]),
                          "within_noise_or_better": "yes" if qualifies else "no"}
        summary.append(row)
        print(f"n={n:3d}: accuracy {row['display']['accuracy']}  ask {row['display']['ask_rate']}  "
              f"within noise of ① or better: {row['display']['within_noise_or_better']}")
    doc["summary"] = summary
    doc["gauge1_reference"] = {"ask_rate": {"value": ref_ask.get("value"), "ci": ref_ask["ci"]},
                               "accuracy": g1["exam"]["metrics"]["accuracy"],
                               "display": {"ask_rate": g1["exam"]["display"]["ask_rate"],
                                           "accuracy": g1["exam"]["display"]["accuracy"]},
                               "source": "results/gauge1-logit.json exam block (Gate 3)"}
    doc["rule"] = ("smallest n whose draw-mean ask-rate CI has its lower end <= the upper end of gauge ①'s exam "
                   "ask-rate CI (overlap = within noise; entirely below = better); none if no n qualifies")
    doc["smallest_n_within_noise_or_better"] = smallest
    doc["smallest_n_display"] = DASH if smallest is None else str(smallest)
    doc["exam"] = {"n_exam": len(y_all), "n_boot": N_BOOT, "ci_level": 95, "seed": SEED, "sa_target": SA_TARGET,
                   "exam_sha256": exam_sha, "scored_at": _utc_now(), "prereg_commit": prereg_commit,
                   **(section_info or {})}
    annotate_labels_needed_sa(doc)
    write_json(LABELS_NEEDED_PATH, doc)
    draw_labels_needed(doc)
    print(f"smallest n within noise of gauge ①'s ask rate or better: {doc['smallest_n_display']} "
          f"(gauge ① {g1['exam']['display']['ask_rate']}); wrote {LABELS_NEEDED_PATH.name}, {LABELS_NEEDED_PNG.name}")
    return 0


def annotate_labels_needed_sa(doc: dict) -> dict:
    """Descriptive context for the ask-rate rule, from the per-variant exam blocks already stored (the exam is
    not opened): per n, the selective accuracy each draw achieved on the exam at its own fit-chosen thresholds —
    mean, min–max, and how many draws met the SA target. A low ask rate with selective accuracy below the
    target is under-asking, not an improvement at equal selective accuracy."""
    by_n: dict[int, list[float]] = {}
    for v in doc["variants"]:
        sa = v["exam"]["metrics"]["selective_accuracy"]["value"]
        by_n.setdefault(v["n"], []).append(sa)
    below = []
    for row in doc["summary"]:
        vals = [x for x in by_n[row["n"]] if x is not None]
        mean = round(float(np.mean(vals)), 4) if vals else None
        met = sum(1 for x in vals if x + 1e-12 >= SA_TARGET)
        row["selective_accuracy"] = {"mean": mean, "min": min(vals) if vals else None, "max": max(vals) if vals else None,
                                     "draws_meeting_target": met, "source": "per-variant exam blocks (no new exam read)"}
        row["display"]["selective_accuracy"] = (_pct(mean) + (f" ({_pct(min(vals))}–{_pct(max(vals))})"
                                                              if row["draws"] > 1 else "")) if vals else DASH
        row["display"]["draws_meeting_target"] = f"{met}/{row['draws']}"
        if mean is None or mean + 1e-12 < SA_TARGET:
            below.append(row["n"])
    g1_exam = read_json(GAUGE1_PATH)["exam"]
    doc["gauge1_reference"]["selective_accuracy"] = g1_exam["metrics"]["selective_accuracy"]
    doc["gauge1_reference"]["display"]["selective_accuracy"] = g1_exam["display"]["selective_accuracy"]
    doc["n_with_mean_sa_below_target"] = below
    doc["sa_note"] = ("thresholds fitted on small out-of-fold sets under-ask: where the draw-mean selective accuracy "
                      "achieved on the exam is below the SA target, a lower ask rate is not an improvement at equal "
                      "selective accuracy")
    return doc


def draw_labels_needed(doc: dict, out_path: Path = LABELS_NEEDED_PNG) -> None:
    """Exam ask rate @ SA target and accuracy vs number of fit labels: one dot per draw, the draw mean with its
    paired CI, gauge ①'s value and CI as a reference band."""
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D

    ref = doc["gauge1_reference"]
    with_sa = all("selective_accuracy" in row for row in doc["summary"]) and "selective_accuracy" in ref
    panels = [("ask_rate", "exam ask rate @ SA 0.95 (%)", ref["ask_rate"])]
    if with_sa:
        panels.append(("selective_accuracy", "selective accuracy achieved on the exam (%)", ref["selective_accuracy"]))
    panels.append(("accuracy", "exam accuracy (%)", ref["accuracy"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.3), dpi=150)
    fig.patch.set_facecolor("white")
    sizes = [row["n"] for row in doc["summary"]]
    for ax, (key, ylabel, r) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.axhspan(100 * r["ci"][0], 100 * r["ci"][1], color=PALETTE[1], alpha=0.12, linewidth=0, zorder=1)
        ax.axhline(100 * r["value"], color=PALETTE[1], linewidth=1.2, linestyle="--", zorder=2)
        if key == "selective_accuracy":
            ax.axhline(100 * SA_TARGET, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2)
        for v in doc["variants"]:
            val = v["exam"]["metrics"][key]["value"]
            if val is not None:
                ax.plot([v["n"]], [100 * val], linestyle="none", marker="o", markersize=3.5,
                        color=PALETTE[0], alpha=0.45, zorder=3)
        means = [100 * row[key]["mean"] for row in doc["summary"]]
        if all(isinstance(row[key].get("ci"), list) for row in doc["summary"]):
            lo = [100 * row[key]["ci"][0] for row in doc["summary"]]
            hi = [100 * row[key]["ci"][1] for row in doc["summary"]]
            ax.fill_between(sizes, lo, hi, color=PALETTE[0], alpha=0.15, linewidth=0, zorder=2)
        ax.plot(sizes, means, color=PALETTE[0], linewidth=1.8, marker="s", markersize=5, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel("fit labels used to train the head (equal per class)", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
    handles = [
        Line2D([], [], color=PALETTE[0], linewidth=1.8, marker="s", markersize=5),
        Line2D([], [], color=PALETTE[0], linestyle="none", marker="o", markersize=3.5, alpha=0.45),
        Line2D([], [], color=PALETTE[1], linewidth=1.2, linestyle="--"),
    ]
    labels = ["gauge ③ head: mean over draws (band: paired 95% CI where computed)", "one draw",
              "gauge ① logit read, 95% CI band"]
    if with_sa:
        handles.append(Line2D([], [], color=INK_MUTED, linewidth=1.0, linestyle=":"))
        labels.append(f"selective-accuracy target {SA_TARGET:.2f}")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=7.5)
    fig.suptitle("set A exam — labels needed by the gauge ③ head (thresholds fitted on each draw's out-of-fold "
                 "predictions)", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"labels-needed.png -> {out_path}")


# ----------------------------------------------------------------------------- like-for-like + Jev (Task D)

LIKE_PATH = RESULTS_DIR / "like-for-like.json"
JEV_PATH = RESULTS_DIR / "jev.json"
GAUGE3_PATH = RESULTS_DIR / "gauge3-probe.json"
SUBSAMPLE_PATH = ROOT / "data" / "seta" / "jev-subsample.json"
GATE_TAU = SA_TARGET  # the unfitted gate: answer iff max p >= 0.95 (PREREG Amendment 1, like-for-like group only)
NA_NO_FIT = "n/a (no fit-split outputs)"
NA_NO_CONF = "n/a (no confidence returned)"


def compute_no_thresholds(gauge: str, L: np.ndarray, y: np.ndarray, T: float, na_text: str,
                          confidence: bool) -> tuple[dict, list[dict], list[list[int]]]:
    """compute() for a gauge with no fitted thresholds (external, exam-only): the same bootstrap and metric code,
    ask rate / selective accuracy null. Without returned confidence, ECE, NLL and AUROC are null too (a one-hot
    vector is a recorded choice, not a confidence)."""
    n = int(len(y))
    P_raw = metrics.softmax(L)
    P_cal = metrics.softmax(L, T)
    _validate_probs(gauge, P_raw, "raw")
    _validate_probs(gauge, P_cal, "calibrated")
    fns = {
        "accuracy": lambda idx: metrics.accuracy(P_cal[idx], y[idx]),
        "macro_f1": lambda idx: metrics.macro_f1(P_cal[idx], y[idx]),
    }
    if confidence:
        fns.update({
            "nll_raw": lambda idx: metrics.nll(P_raw[idx], y[idx]),
            "nll_cal": lambda idx: metrics.nll(P_cal[idx], y[idx]),
            "ece_raw": lambda idx: metrics.ece(P_raw[idx], y[idx], ECE_BINS),
            "ece_cal": lambda idx: metrics.ece(P_cal[idx], y[idx], ECE_BINS),
            "auroc": lambda idx: metrics.auroc(P_cal[idx], y[idx]),
        })
    boot = metrics.bootstrap(fns, n, n_boot=N_BOOT, seed=SEED)
    result = {}
    for name in METRIC_ORDER:
        if name in boot:
            result[name] = dict(boot[name])
        else:
            result[name] = {"value": None, "ci": None, "n_boot_used": 0,
                            "na": na_text if name in ("ask_rate", "selective_accuracy") else NA_NO_CONF}
    result["selective_accuracy"]["n_answered"] = None
    curve = metrics.curve(P_cal, y, CURVE_POINTS) if confidence else []
    return result, curve, metrics.confusion(P_cal, y)


def gate_block(P: np.ndarray, y: np.ndarray) -> dict:
    """The unfitted gate 'answer iff max p >= 0.95' with the same paired resamples as every other bootstrap."""
    taus = [GATE_TAU] * len(LABELS)
    boot = metrics.bootstrap({"ask_rate": lambda idx: metrics.ask_rate(P[idx], taus),
                              "selective_accuracy": lambda idx: metrics.selective_accuracy(P[idx], y[idx], taus)},
                             len(y), n_boot=N_BOOT, seed=SEED)
    _, answered = metrics.abstain(P, taus)
    boot["selective_accuracy"]["n_answered"] = int(np.count_nonzero(answered))
    return {"tau": GATE_TAU, "ask_rate": boot["ask_rate"], "selective_accuracy": boot["selective_accuracy"],
            "display": {"ask_rate": _with_ci(boot["ask_rate"], 100, 1),
                        "selective_accuracy": _with_ci(boot["selective_accuracy"], 100, 1)}}


def score_like_for_like(rescore: bool) -> int:
    """PREREG Amendment 1: Jev's exam block, and gauges ①, ③ and Jev on the same items (subsample ids where Jev
    succeeded, exam.jsonl order) with paired resamples; plus the unfitted gate for all three."""
    for p in (JEV_PATH, GAUGE1_PATH, GAUGE3_PATH, SUBSAMPLE_PATH):
        if not p.exists():
            _fail(f"{p.name} not found")
    jev, g1, g3 = read_json(JEV_PATH), read_json(GAUGE1_PATH), read_json(GAUGE3_PATH)
    sub = read_json(SUBSAMPLE_PATH)
    if not rescore and ("exam" in jev or LIKE_PATH.exists()):
        _fail("the Jev column / like-for-like group is already scored; it is read once (rule 8)")
    if jev.get("subsample", {}).get("sha256") != sha256_file(SUBSAMPLE_PATH):
        _fail("jev.json was built from a different jev-subsample.json")

    # phase 1 — no exam data: calibrations, fingerprints, the common item list, arrays
    gauges = {"gauge1-logit": g1, "gauge3-probe": g3, "jev": jev}
    calibs = {}
    for name, data in gauges.items():
        cp = RESULTS_DIR / f"{name}.calib.json"
        if not cp.exists():
            _fail(f"{cp.name} not found — run calibrate.py first")
        _check_fingerprint(name, data, cp)
        calibs[name] = read_json(cp)
    jev_ok = {it["id"] for it in jev["items"]}
    common = [i for i in sub["ids"] if i in jev_ok]
    failed = [f["id"] for f in jev.get("failed", [])]
    if len(common) + len(failed) != len(sub["ids"]):
        _fail(f"Jev successes ({len(common)}) + failures ({len(failed)}) != subsample ({len(sub['ids'])})")
    rows = {}
    for name, data in gauges.items():
        by_id = {it["id"]: it for it in data["items"] if it.get("split") == "exam"}
        missing = [i for i in common if i not in by_id]
        if missing:
            _fail(f"{name}: {len(missing)} like-for-like ids have no exam item, e.g. {missing[:3]}")
        rows[name] = [by_id[i] for i in common]
        L, y = _arrays(name, rows[name])
        _validate_probs(name, metrics.softmax(L, float(calibs[name]["T"])), "calibrated")

    # phase 2 — the exam is opened here
    prereg_commit, exam_sha, section_info, truth = _open_exam_for(jev.get("prereg_section"))
    for name in gauges:
        _check_split(name, rows[name], "exam", truth, subset_ok=True)
    ex_order = [i for i in truth if i in set(common)]
    if ex_order != common:
        _fail("the like-for-like ids are not in exam.jsonl order")

    # phase 3 — compute (same n, seed and order for every row -> paired resamples)
    confidence = jev.get("confidence") == "probabilities"
    na_jev = NA_NO_FIT if jev.get("confidence") in ("probabilities", "mixed") else NA_NO_CONF
    out_rows = {}
    for name in gauges:
        L, y = _arrays(name, rows[name])
        c = calibs[name]
        T = float(c["T"])
        if name == "jev":
            m, curve, conf = compute_no_thresholds(name, L, y, T, na_jev, confidence)
            d = display_strings(m)
            d["ask_rate"] = d["selective_accuracy"] = na_jev
            if not confidence:
                d["ece"] = d["nll"] = d["auroc"] = NA_NO_CONF
        else:
            taus = [float(t) for t in c["taus"]]
            m, curve, conf = compute(name, L, y, T, taus)
            d = display_strings(m)
        gate = gate_block(metrics.softmax(L, T), y) if (name != "jev" or confidence) else None
        out_rows[name] = {"T": T, "taus": c.get("taus"), "metrics": m, "display": d, "confusion": conf,
                          "unfitted_gate": gate, "curve": curve}
        print(f"{name:>12} [like-for-like n={len(y)}] acc {d['accuracy']}  ece {d['ece']}  ask {d['ask_rate']}  "
              f"sa {d['selective_accuracy']}  auroc {d['auroc']}"
              + (f"  | gate max p>={GATE_TAU}: ask {gate['display']['ask_rate']} sa {gate['display']['selective_accuracy']}"
                 if gate else ""))
    scored_at = _utc_now()
    exam_common = {"n_exam": len(common), "n_boot": N_BOOT, "ci_level": 95, "seed": SEED, "ece_bins": ECE_BINS,
                   "exam_sha256": exam_sha, "scored_at": scored_at, "prereg_commit": prereg_commit, **(section_info or {})}
    jr = out_rows["jev"]
    jev["exam"] = {**exam_common, "T": jr["T"], "taus": None, "tau_global": None,
                   "fit_fingerprint": calibs["jev"]["fit_fingerprint"]["sha256"], "n_failed": len(failed),
                   "metrics": jr["metrics"], "curve": jr["curve"], "confusion": jr["confusion"],
                   "display": jr["display"], "unfitted_gate": jr["unfitted_gate"]}
    write_json(JEV_PATH, jev)
    like = {
        "what": "like-for-like group: gauges ①, ③ and Jev on the same exam items (the Jev subsample minus Jev "
                "failures, exam.jsonl order); one bootstrap seed and order for every row, so resamples are paired",
        "prereg_section": jev.get("prereg_section"),
        "ids_file": "data/seta/jev-subsample.json", "subsample_sha256": jev["subsample"]["sha256"],
        "n": len(common), "n_jev_failed": len(failed), "jev_failed_ids": failed,
        "gate_rule": f"unfitted gate: answer iff max p >= {GATE_TAU} (each gauge's own final probabilities; "
                     f"no threshold fitted)",
        "rows": {name: {k: v for k, v in r.items() if k != "curve"} for name, r in out_rows.items()},
        "exam": exam_common,
    }
    write_json(LIKE_PATH, like)
    draw_curve(CURVE_PNG)
    print(f"wrote {JEV_PATH.name} exam block and {LIKE_PATH.name} (n={len(common)}, Jev failed {len(failed)})")
    return 0


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
    ap.add_argument("--labels-needed", action="store_true",
                    help="score the labels-needed head variants (results/labels-needed.json) once; draw labels-needed.png")
    ap.add_argument("--like-for-like", action="store_true",
                    help="score the Jev column and the like-for-like group (①, ③, Jev on the same items) once")
    args = ap.parse_args(argv)
    if args.labels_needed:
        if args.gauge or args.fit_dry_run or args.curve_only or args.like_for_like:
            ap.error("--labels-needed takes only --rescore")
        return score_labels_needed(args.rescore)
    if args.like_for_like:
        if args.gauge or args.fit_dry_run or args.curve_only:
            ap.error("--like-for-like takes only --rescore")
        return score_like_for_like(args.rescore)
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
