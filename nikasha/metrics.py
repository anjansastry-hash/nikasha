"""nikasha.metrics — pure metric functions shared by calibrate.py and score.py.

Everything here is a pure function of in-memory numpy arrays: no file I/O and no printing
(the only output is the one self-check line in main()). Label order is LABELS from the package.

Conventions
- probs : array (n, 3); each row is a probability vector in LABELS order.
- y     : int array (n,); gold label indices (see labels_to_idx).
- taus  : 3 floats, per-class thresholds; pred = argmax(probs); answered iff probs[i, pred] >= taus[pred].
- Metrics that are undefined on a sample (no answered item; correctness all-0 or all-1) return
  None instead of raising, so bootstrap() can skip that resample. An EMPTY sample (n = 0) raises
  ValueError in every metric, ask_rate included (abstain is the rule, not a metric, and simply
  returns empty arrays).
- argmax ties resolve to the lowest index (numpy argmax).
- Returned numbers are plain floats; only bootstrap() rounds (value / ci to 4 dp, per the spec).
  Consumers (calibrate.py, score.py) apply any display rounding themselves.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

from nikasha import ECE_BINS, LABELS, N_BOOT, SA_TARGET, SEED

K = len(LABELS)

# Grid constants (calibrate.py may copy these into the .calib.json "grid" block).
T_GRID_MIN, T_GRID_MAX, T_GRID_POINTS = 0.05, 10.0, 200
TAU_MIN, TAU_MAX, TAU_STEP = 0.34, 0.99, 0.01

_P_CLIP = 1e-12   # probability floor inside NLL
_GE_EPS = 1e-12   # tolerance for "selective accuracy >= target" (ratios k/m can never sit this close below)
_FLAT_TOL = 1e-12  # grid-NLL spread at or below this = flat surface (every logit row constant): T is undetermined


# ----------------------------------------------------------------------------- helpers

def _probs(probs) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] != K:
        raise ValueError(f"probs must have shape (n, {K}); got {p.shape}")
    return p


def _pair(probs, y) -> tuple[np.ndarray, np.ndarray]:
    p = _probs(probs)
    yy = np.asarray(y).reshape(-1).astype(int)
    if yy.shape[0] != p.shape[0]:
        raise ValueError(f"probs has {p.shape[0]} rows but y has {yy.shape[0]} entries")
    if p.shape[0] == 0:
        raise ValueError("empty sample")
    if yy.min() < 0 or yy.max() >= K:
        raise ValueError(f"y must hold label indices in [0, {K})")
    return p, yy


def _taus(taus) -> np.ndarray:
    t = np.asarray(taus, dtype=float).reshape(-1)
    if t.shape[0] != K:
        raise ValueError(f"taus must have {K} entries; got {t.shape[0]}")
    return t


def _float_or_none(v):
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def _r4(v):
    return None if v is None else round(float(v), 4)


def _meets(sa, target: float) -> bool:
    return sa is not None and sa + _GE_EPS >= target


def _confusion_array(p: np.ndarray, yy: np.ndarray) -> np.ndarray:
    pred = p.argmax(axis=1)
    return np.bincount(yy * K + pred, minlength=K * K).reshape(K, K)


def temperature_grid() -> np.ndarray:
    """geomspace(0.05, 10, 200)."""
    return np.geomspace(T_GRID_MIN, T_GRID_MAX, T_GRID_POINTS)


def tau_grid() -> np.ndarray:
    """arange(0.34, 0.995, 0.01) rounded to 2 dp: 0.34, 0.35, ..., 0.99 (66 points)."""
    return np.round(np.arange(TAU_MIN, TAU_MAX + TAU_STEP / 2, TAU_STEP), 2)


# ----------------------------------------------------------------------------- public API

def labels_to_idx(labels: list[str]) -> np.ndarray:
    """Label strings -> int indices via LABELS.index (raises ValueError on an unknown label)."""
    return np.array([LABELS.index(lab) for lab in labels], dtype=int)


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Row-wise, numerically stable softmax of logits / T along the last axis."""
    if not T > 0:
        raise ValueError(f"T must be > 0; got {T}")
    z = np.asarray(logits, dtype=float) / float(T)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(probs, y) -> float:
    """Mean -log p[y], with p clipped to [1e-12, 1]."""
    p, yy = _pair(probs, y)
    py = np.clip(p[np.arange(p.shape[0]), yy], _P_CLIP, 1.0)
    return float(-np.log(py).mean())


def ece(probs, y, bins: int = ECE_BINS) -> float:
    """Expected calibration error: confidence = max p; `bins` equal-width bins on [0, 1]
    (edges linspace(0, 1, bins+1), bin b = [e_b, e_{b+1}) except the last, which is closed so
    confidence 1.0 lands in it); ECE = sum_b (n_b / n) * |acc_b - conf_b|."""
    p, yy = _pair(probs, y)
    n = p.shape[0]
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == yy).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    b = np.digitize(conf, edges[1:-1], right=False)  # 0 .. bins-1
    sum_conf = np.bincount(b, weights=conf, minlength=bins)
    sum_acc = np.bincount(b, weights=correct, minlength=bins)
    # (n_b/n)*|acc_b - conf_b| == |sum_acc_b - sum_conf_b| / n; empty bins contribute 0 either way.
    return float(np.abs(sum_acc - sum_conf).sum() / n)


def accuracy(probs, y) -> float:
    """mean(argmax(probs) == y)."""
    p, yy = _pair(probs, y)
    return float((p.argmax(axis=1) == yy).mean())


def macro_f1(probs, y) -> float:
    """Unweighted mean F1 over the K labels; a label with no gold and no predictions scores F1 = 0."""
    p, yy = _pair(probs, y)
    c = _confusion_array(p, yy).astype(float)
    tp = np.diag(c)
    fp = c.sum(axis=0) - tp
    fn = c.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1.0), 0.0)
    return float(f1.mean())


def confusion(probs, y) -> list[list[int]]:
    """Confusion matrix as nested lists: rows = gold, cols = predicted, both in LABELS order."""
    p, yy = _pair(probs, y)
    return [[int(v) for v in row] for row in _confusion_array(p, yy)]


def abstain(probs, taus) -> tuple[np.ndarray, np.ndarray]:
    """(pred int array, answered bool array): answered iff probs[i, pred] >= taus[pred]."""
    p = _probs(probs)
    t = _taus(taus)
    pred = p.argmax(axis=1)
    answered = p[np.arange(p.shape[0]), pred] >= t[pred]
    return pred, answered


def ask_rate(probs, taus) -> float:
    """1 - answered / n. Raises ValueError on an empty sample, like every other metric."""
    _, answered = abstain(probs, taus)
    n = answered.shape[0]
    if n == 0:
        raise ValueError("empty sample")
    return float(1.0 - answered.sum() / n)


def selective_accuracy(probs, y, taus) -> float | None:
    """Accuracy over answered items; None if no item is answered."""
    p, yy = _pair(probs, y)
    pred, answered = abstain(p, taus)
    if not answered.any():
        return None
    return float((pred[answered] == yy[answered]).mean())


def auroc(probs, y) -> float | None:
    """sklearn roc_auc_score(correct, max p); None when correctness is all-0 or all-1."""
    p, yy = _pair(probs, y)
    correct = (p.argmax(axis=1) == yy).astype(int)
    if correct.min() == correct.max():
        return None
    return float(roc_auc_score(correct, p.max(axis=1)))


def fit_temperature(logits, y) -> tuple[float, dict]:
    """Temperature minimising NLL(softmax(logits / T), y).

    Grid geomspace(0.05, 10, 200) on NLL, then scipy minimize_scalar(method="bounded") between
    the grid neighbours of the grid minimum. The refined point is kept only if its NLL is not
    worse than the grid minimum (safety against a flat or noisy local refine). If the grid NLL is
    flat (spread <= 1e-12: every logit row constant, e.g. a uniform baseline) every T minimises
    NLL equally, so the identity T = 1.0 is returned instead of the grid floor that argmin would
    pick. Returns (T, info); info has grid_T, grid_nll (200 each), grid_best_T, grid_best_nll,
    refine_bounds, refine_T, refine_nll (raw scalar-minimiser result), refined_nll (NLL at the
    returned T), T, method: "grid+refine" (refined point kept), "grid" (refine rejected, grid
    point returned) or "flat" (T = 1.0 returned on a flat surface).
    """
    L = np.asarray(logits, dtype=float)
    if L.ndim != 2 or L.shape[1] != K:
        raise ValueError(f"logits must have shape (n, {K}); got {L.shape}")
    yy = np.asarray(y).reshape(-1).astype(int)

    def f(T: float) -> float:
        return nll(softmax(L, T), yy)

    grid = temperature_grid()
    grid_nll = np.array([f(float(T)) for T in grid])
    i = int(np.argmin(grid_nll))
    lo = float(grid[max(i - 1, 0)])
    hi = float(grid[min(i + 1, grid.shape[0] - 1)])
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    refine_T = float(res.x)
    refine_nll = float(res.fun)
    if float(np.ptp(grid_nll)) <= _FLAT_TOL:
        T_best, nll_best, method = 1.0, f(1.0), "flat"
    elif np.isfinite(refine_nll) and refine_nll <= grid_nll[i]:
        T_best, nll_best, method = refine_T, refine_nll, "grid+refine"
    else:
        T_best, nll_best, method = float(grid[i]), float(grid_nll[i]), "grid"
    info = {
        "method": method,
        "grid_T": [float(t) for t in grid],
        "grid_nll": [float(v) for v in grid_nll],
        "grid_best_T": float(grid[i]),
        "grid_best_nll": float(grid_nll[i]),
        "refine_bounds": [lo, hi],
        "refine_T": refine_T,
        "refine_nll": refine_nll,
        "T": float(T_best),
        "refined_nll": float(nll_best),
    }
    return float(T_best), info


def fit_thresholds(probs_cal, y, sa_target: float = SA_TARGET) -> dict:
    """Per-class thresholds: minimal ask rate subject to selective accuracy >= sa_target.

    1. tau_global = smallest tau on arange(0.34, 0.995, 0.01) (2 dp) with SA(tau, tau, tau) >= target
       and at least one item answered. None reachable -> taus [1.01]*3 (abstain on everything).
    2. taus = [tau_global]*3; coordinate descent: for k = 0, 1, 2 lower taus[k] by 0.01 (never
       below 0.34) while SA >= target still holds with >= 1 answered; repeat passes until a full
       pass moves nothing.
    Returns {"tau_global", "taus", "target_reachable", "ask_rate", "selective_accuracy"} with
    ask_rate / selective_accuracy evaluated at the final taus as plain (unrounded) floats.
    """
    p, yy = _pair(probs_cal, y)
    tau_global = None
    for t in tau_grid():
        if _meets(selective_accuracy(p, yy, [t] * K), sa_target):
            tau_global = float(t)
            break
    if tau_global is None:
        return {"tau_global": None, "taus": [1.01] * K, "target_reachable": False,
                "ask_rate": 1.0, "selective_accuracy": None}

    taus = [tau_global] * K
    moved = True
    while moved:
        moved = False
        for k in range(K):
            while True:
                cand = round(taus[k] - TAU_STEP, 2)
                if cand < TAU_MIN - 1e-9:
                    break
                trial = list(taus)
                trial[k] = cand
                if _meets(selective_accuracy(p, yy, trial), sa_target):
                    taus = trial
                    moved = True
                else:
                    break
    return {
        "tau_global": tau_global,
        "taus": [float(t) for t in taus],
        "target_reachable": True,
        "ask_rate": ask_rate(p, taus),
        "selective_accuracy": selective_accuracy(p, yy, taus),
    }


def curve(probs_cal, y, n_points: int = 50) -> list[dict]:
    """Selective accuracy vs ask rate for a global tau in linspace(1/3, 1.0, n_points);
    answered iff max p >= tau. Each point: {"tau", "ask_rate", "selective_accuracy", "n_answered"}
    as plain floats; the stored tau is exactly the sweep value the answered test used, so
    n_answered is reproducible from the stored point."""
    p, yy = _pair(probs_cal, y)
    n = p.shape[0]
    conf = p.max(axis=1)
    correct = p.argmax(axis=1) == yy
    out = []
    for t in np.linspace(1.0 / 3.0, 1.0, n_points):
        answered = conf >= t
        k = int(answered.sum())
        sa = float(correct[answered].mean()) if k > 0 else None
        out.append({"tau": float(t), "ask_rate": float(1.0 - k / n),
                    "selective_accuracy": sa, "n_answered": k})
    return out


def bootstrap(fns: dict[str, Callable], n: int, n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, dict]:
    """Percentile bootstrap over item indices.

    rng = np.random.default_rng(seed); for each of n_boot resamples idx = rng.integers(0, n, n) and the
    SAME idx is handed to every fn. Each fn(idx) returns float or None (None / non-finite = undefined
    on that resample and skipped). Returns {name: {"value": fn(arange(n)), "ci": [p2.5, p97.5] over the
    defined resamples (None if fewer than 2), "n_boot_used": k}}, values and ci rounded to 4 dp.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("bootstrap needs n > 0")
    full = np.arange(n)
    values = {name: _float_or_none(fn(full)) for name, fn in fns.items()}
    samples: dict[str, list[float]] = {name: [] for name in fns}
    rng = np.random.default_rng(seed)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        for name, fn in fns.items():
            v = _float_or_none(fn(idx))
            if v is not None:
                samples[name].append(v)
    out: dict[str, dict] = {}
    for name in fns:
        s = samples[name]
        if len(s) >= 2:
            lo, hi = np.percentile(s, [2.5, 97.5])
            ci = [_r4(lo), _r4(hi)]
        else:
            ci = None
        out[name] = {"value": _r4(values[name]), "ci": ci, "n_boot_used": len(s)}
    return out


# ----------------------------------------------------------------------------- self-check

def _self_check() -> None:
    rng = np.random.default_rng(SEED)

    # softmax: rows sum to 1, stable under large logits, T rescales
    L = rng.normal(size=(7, K)) * 3
    P = softmax(L)
    assert P.shape == (7, K) and np.allclose(P.sum(axis=1), 1.0, atol=1e-12) and np.all(P >= 0)
    assert np.allclose(softmax(L, T=2.0).sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(softmax(np.array([[1000.0, 0.0, 0.0]])), [[1.0, 0.0, 0.0]])
    assert np.allclose(softmax(np.zeros((2, K))), 1.0 / K)

    # labels_to_idx
    assert labels_to_idx(LABELS).tolist() == list(range(K))
    assert labels_to_idx([]).shape == (0,)

    # nll
    assert nll(np.array([[1.0, 0.0, 0.0]]), [0]) == 0.0
    assert abs(nll(np.array([[0.0, 1.0, 0.0]]), [0]) - (-np.log(1e-12))) < 1e-9

    # ece: perfectly calibrated two-point example (conf 0.75 x4 with 3 right, conf 0.5 x2 with 1 right)
    probs = np.array([[0.75, 0.15, 0.10]] * 4 + [[0.5, 0.3, 0.2]] * 2)
    y = np.array([0, 0, 0, 1, 0, 1])
    assert abs(ece(probs, y)) < 1e-12
    # fully miscalibrated: conf 0.75, all wrong -> ECE 0.75; conf 1.0 (last bin) all right -> 0
    assert abs(ece(np.array([[0.75, 0.15, 0.10]] * 4), np.array([1, 1, 1, 1])) - 0.75) < 1e-12
    assert ece(np.eye(K), np.arange(K)) == 0.0

    # accuracy / macro-F1 / confusion
    assert accuracy(probs, y) == 4 / 6
    eye = np.eye(K)
    assert macro_f1(eye, np.arange(K)) == 1.0
    assert abs(macro_f1(eye[[0, 0, 1, 1]], np.array([0, 0, 1, 1])) - 2 / 3) < 1e-12  # label 2 absent -> F1 0
    assert confusion(probs, y) == [[4, 0, 0], [2, 0, 0], [0, 0, 0]]
    assert confusion(eye, np.array([1, 2, 0])) == [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

    # abstain / ask_rate / selective_accuracy
    pred, ans = abstain(probs, [0.6, 0.6, 0.6])
    assert pred.tolist() == [0] * 6 and ans.tolist() == [True] * 4 + [False] * 2
    assert abs(ask_rate(probs, [0.6] * K) - 2 / 6) < 1e-12
    assert selective_accuracy(probs, y, [0.6] * K) == 0.75
    assert selective_accuracy(probs, y, [1.01] * K) is None
    assert ask_rate(probs, [1.01] * K) == 1.0
    assert ask_rate(probs, [0.34] * K) == 0.0
    try:  # an empty sample raises, like every other metric
        ask_rate(np.zeros((0, K)), [0.5] * K)
        raise AssertionError("ask_rate accepted an empty sample")
    except ValueError:
        pass

    # auroc
    assert auroc(eye, np.arange(K)) is None
    assert auroc(np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]]), np.array([0, 1])) == 1.0

    # fit_temperature: logits [2,0,0] with gold 0 half the time -> optimum where p0 = 0.5, T* = 2/ln 2
    Lt = np.array([[2.0, 0.0, 0.0]] * 10)
    yt = np.array([0] * 5 + [1] * 3 + [2] * 2)
    T, info = fit_temperature(Lt, yt)
    assert abs(T - 2.0 / np.log(2.0)) < 1e-3
    assert len(info["grid_T"]) == T_GRID_POINTS and len(info["grid_nll"]) == T_GRID_POINTS
    assert info["refined_nll"] <= min(info["grid_nll"]) + 1e-12
    assert T_GRID_MIN <= T <= T_GRID_MAX
    assert info["method"] == "grid+refine" and info["T"] == T
    # flat surface (every row constant -> uniform probs at any T): identity T = 1, not the grid floor
    Tf, info_f = fit_temperature(np.zeros((4, K)), np.array([0, 1, 2, 0]))
    assert Tf == 1.0 and info_f["method"] == "flat"
    assert abs(info_f["refined_nll"] - np.log(K)) < 1e-12 and len(info_f["grid_nll"]) == T_GRID_POINTS

    # fit_thresholds, easy case: gold 0 always right at conf .60; gold 1 right at .95;
    # gold 2: 20 right at .95, 10 wrong (pred 1) at .80  ->  tau_global .81, taus [.34, .81, .34]
    pf = np.array([[0.60, 0.25, 0.15]] * 30 + [[0.03, 0.95, 0.02]] * 30
                  + [[0.02, 0.03, 0.95]] * 20 + [[0.10, 0.80, 0.10]] * 10)
    yf = np.array([0] * 30 + [1] * 30 + [2] * 30)
    th = fit_thresholds(pf, yf)
    assert th["target_reachable"] and th["tau_global"] == 0.81
    assert all(t <= th["tau_global"] for t in th["taus"])
    assert th["taus"] == [0.34, 0.81, 0.34]
    assert th["selective_accuracy"] == 1.0 and abs(th["ask_rate"] - 10 / 90) < 1e-12  # unrounded
    assert tau_grid().shape == (66,) and tau_grid()[0] == 0.34 and tau_grid()[-1] == 0.99
    # unreachable: every prediction wrong
    bad = fit_thresholds(np.array([[0.9, 0.05, 0.05]] * 10), np.ones(10, dtype=int))
    assert bad == {"tau_global": None, "taus": [1.01] * K, "target_reachable": False,
                   "ask_rate": 1.0, "selective_accuracy": None}

    # curve
    cv = curve(pf, yf)
    assert len(cv) == 50 and set(cv[0]) == {"tau", "ask_rate", "selective_accuracy", "n_answered"}
    assert cv[0]["n_answered"] == 90 and cv[0]["ask_rate"] == 0.0
    assert cv[-1]["n_answered"] == 0 and cv[-1]["selective_accuracy"] is None and cv[-1]["ask_rate"] == 1.0
    assert cv[0]["tau"] == 1.0 / 3.0 and cv[-1]["tau"] == 1.0  # stored tau is the exact sweep value
    conf_f = pf.max(axis=1)
    assert all(pt["n_answered"] == int((conf_f >= pt["tau"]).sum()) for pt in cv)  # reproducible from the point

    # bootstrap: ci of length 2, None-valued fn skipped, same idx for every fn, deterministic
    n = yf.shape[0]
    fns = {"acc": lambda idx: accuracy(pf[idx], yf[idx]),
           "none": lambda idx: None,
           "sa": lambda idx: selective_accuracy(pf[idx], yf[idx], [0.81] * K)}
    bs = bootstrap(fns, n, n_boot=50, seed=SEED)
    assert len(bs["acc"]["ci"]) == 2 and 0.0 <= bs["acc"]["ci"][0] <= bs["acc"]["ci"][1] <= 1.0
    assert bs["acc"]["n_boot_used"] == 50 and bs["acc"]["value"] == round(80 / 90, 4)
    assert bs["none"] == {"value": None, "ci": None, "n_boot_used": 0}
    assert bs["sa"]["value"] == 1.0 and len(bs["sa"]["ci"]) == 2
    assert bootstrap(fns, n, n_boot=50, seed=SEED) == bs
    seen: list[np.ndarray] = []
    bootstrap({"a": lambda idx: seen.append(idx.copy()) or 0.0,
               "b": lambda idx: seen.append(idx.copy()) or 0.0}, n, n_boot=3, seed=SEED)
    assert len(seen) == 8 and np.array_equal(seen[0], np.arange(n))
    assert np.array_equal(seen[2], seen[3]) and np.array_equal(seen[4], seen[5])
    # the spec's exact protocol: rng = default_rng(seed); resample b uses idx = rng.integers(0, n, n),
    # in order, with no other draws in between (deterministic, hand-reproducible)
    ref = np.random.default_rng(SEED)
    assert np.array_equal(seen[2], ref.integers(0, n, n)) and np.array_equal(seen[4], ref.integers(0, n, n))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m nikasha.metrics",
        description="Pure metric functions for nikasha (no I/O). Running the module executes a "
                    "self-check on synthetic arrays; no set-A data is touched.")
    parser.parse_args()
    _self_check()
    print("metrics self-check OK")


if __name__ == "__main__":
    main()
