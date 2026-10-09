"""gauge J — JSON emission: ask the 12B engine to emit a decision field and a confidence, then trust it
(or gate it). Pre-registered in PREREG.md, "## Amendment 2 — gauge J (JSON emission)", before any exam read.

The common practice this measures: "ask the LLM to emit a JSON field and trust it". Same engine as gauge ①
(ENGINE_PATH), same item content (gauge ①'s TOOLS block and request, rendered by gauge ①'s own code), minus the
lettered options. The model is asked to reply with only
    {"decision": "none" | "one" | "several", "confidence": <number 0–1>}
Greedy decoding, at most MAX_NEW_TOKENS new tokens, one generation per item, no retries.

Parsing (pre-registered): the first JSON object in the output = json.JSONDecoder().raw_decode at the first "{";
`decision` must be one of the three strings exactly and `confidence` a JSON number (not a boolean) in [0, 1];
other keys are ignored. Anything else is a parse failure.

This is not a probability-vector gauge, so it writes no results/<gauge>.json in the gauge schema (calibrate.py,
score.py's default mode and readme.py's main table never pick it up). Files:
  results/gaugeJ_fit.jsonl       header line, 300 item lines, footer line (fit split)
  results/gaugeJ_calib.json      per-class thresholds on the self-reported confidence, fitted on the fit split
  results/gaugeJ_latency.json    gauge ① and gauge J seconds per decision, same session, first 50 fit items
  results/gaugeJ_exam.jsonl      header (prereg_commit = the Amendment 2 commit), 700 item lines, footer
The exam file is created exclusively ("x" mode) and written as it goes, so a second exam read cannot overwrite it.

CLI (repo root, venv python):
  python -m nikasha.gauge_json --prompt-sha       prompt spec and sha256; no model, no data
  python -m nikasha.gauge_json --smoke            two synthetic (non set-A) items; prints their raw output
  python -m nikasha.gauge_json --fit              the 300 fit items -> results/gaugeJ_fit.jsonl
  python -m nikasha.gauge_json --calibrate        fit thresholds -> results/gaugeJ_calib.json (no model)
  python -m nikasha.gauge_json --latency 50       gauge ① vs gauge J timing on the first N fit items
  python -m nikasha.gauge_json --exam             the 700 exam items, once -> results/gaugeJ_exam.jsonl
  python -m nikasha.gauge_json --all              each step above whose output is missing, in that order

This module never prints the text of any set-A item or any model output on a set-A item — ids, counts,
numbers and timings only (--smoke prints raw output of its own synthetic items).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from nikasha import (
    ENGINE_NAME,
    ENGINE_PATH,
    EXAM_PATH,
    FIT_PATH,
    LABELS,
    MANIFEST_PATH,
    RESULTS_DIR,
    ROOT,
    SA_TARGET,
    SEED,
    metrics,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
)

GAUGE = "gaugeJ"
PROMPT_VERSION = "j-v1"
AMENDMENT2_HEADING = "## Amendment 2 — gauge J (JSON emission)"

# The three decision values, in label order, and the gauge ① phrase each stands for.
DECISIONS = {"none": "no-call", "one": "one-call", "several": "multi-call"}
DECISION_ORDER = ["none", "one", "several"]
JSON_INSTRUCTION = 'Reply with only {"decision": "none" | "one" | "several", "confidence": <number 0–1>}'
USER_TEMPLATE_J = "TOOLS:\n{tools}\nREQUEST:\n{request}\nReply with only the JSON object."
LETTER_SENTENCE = " Answer with a single letter."  # dropped from gauge ①'s system sentence
MAX_NEW_TOKENS = 48
TEMPERATURE = 0.0
LATENCY_N = 50
J_BOOT = 2000  # Amendment 2: 2,000 bootstrap resamples (seed SEED), paired with gauge ① on the exam

FIT_OUT = RESULTS_DIR / "gaugeJ_fit.jsonl"
EXAM_OUT = RESULTS_DIR / "gaugeJ_exam.jsonl"
CALIB_OUT = RESULTS_DIR / "gaugeJ_calib.json"
LATENCY_OUT = RESULTS_DIR / "gaugeJ_latency.json"
CODE_PATHS = ["nikasha/gauge_json.py", "nikasha/gauge_logit.py", "nikasha/__init__.py", "nikasha/metrics.py"]
PROGRESS_EVERY = 50

UNREACHABLE = "target not reachable"
FINGERPRINT_RECIPE_J = (
    "sha256 of the UTF-8 text json.dumps([[id, decision, confidence, parse_ok], ...], separators=(',', ':'), "
    "ensure_ascii=False) over the fit item lines, sorted by id"
)


# ----------------------------------------------------------------------------------------------
# Prompt j-v1 (built from gauge ①'s own strings, so the item content is identical)
# ----------------------------------------------------------------------------------------------
def system_text_j() -> str:
    from nikasha.gauge_logit import PHRASES, SYSTEM_SENTENCE

    if not SYSTEM_SENTENCE.endswith(LETTER_SENTENCE):
        raise SystemExit("STOP: gauge ①'s system sentence no longer ends with the letter instruction")
    meaning = ", ".join(f'"{d}" = {PHRASES[DECISIONS[d]]}' for d in DECISION_ORDER)
    return (SYSTEM_SENTENCE[: -len(LETTER_SENTENCE)] + "\n" + JSON_INSTRUCTION
            + f", where {meaning}, and confidence is your confidence that the decision is correct.")


def user_text_j(tools_block: str, request: str) -> str:
    return USER_TEMPLATE_J.format(tools=tools_block, request=request)


def prompt_spec() -> dict:
    from nikasha.gauge_logit import TOOLS_CAP_TOKENS, TOOLS_LINE

    return {
        "version": PROMPT_VERSION,
        "system": system_text_j(),
        "user_template": USER_TEMPLATE_J,
        "tools_line": TOOLS_LINE,
        "tools_cap_tokens": TOOLS_CAP_TOKENS,
        "decisions": DECISIONS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    }


def prompt_canonical_json() -> str:
    return json.dumps(prompt_spec(), sort_keys=True, ensure_ascii=False)


def prompt_sha256() -> str:
    return sha256_text(prompt_canonical_json())


# ----------------------------------------------------------------------------------------------
# Parsing and the gate (pure; score.py and selftest.py import these)
# ----------------------------------------------------------------------------------------------
def parse_output(text: str) -> dict:
    """{"decision": label or None, "decision_raw", "confidence", "parse_ok", "parse_error"}."""
    def fail(why, raw=None):
        return {"decision": None, "decision_raw": raw, "confidence": None, "parse_ok": False, "parse_error": why}

    i = text.find("{")
    if i < 0:
        return fail("no '{' in the output")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, i)
    except json.JSONDecodeError:
        return fail("no JSON object at the first '{'")
    if not isinstance(obj, dict):
        return fail("not a JSON object")
    d, c = obj.get("decision"), obj.get("confidence")
    if not isinstance(d, str) or d not in DECISIONS:
        return fail("decision missing or not one of none / one / several")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 <= c <= 1:
        return fail("confidence missing or not a number in [0, 1]", d)
    return {"decision": DECISIONS[d], "decision_raw": d, "confidence": float(c), "parse_ok": True, "parse_error": None}


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(y, decision index or -1, confidence or 0, parse_ok) from item lines, in their order."""
    y = metrics.labels_to_idx([r["label"] for r in rows])
    dec = np.array([LABELS.index(r["decision"]) if r["parse_ok"] else -1 for r in rows], dtype=int)
    conf = np.array([float(r["confidence"]) if r["parse_ok"] else 0.0 for r in rows], dtype=float)
    ok = np.array([bool(r["parse_ok"]) for r in rows], dtype=bool)
    return y, dec, conf, ok


def score_matrix(dec: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Row i holds the self-reported confidence at the emitted decision and 0 elsewhere (all 0 on a parse
    failure). metrics.abstain on it is exactly the pre-registered rule: answer iff the item parsed and
    confidence >= τ[decision]; every τ is >= 0.34, so a zero row (parse failure) is always an ask."""
    S = np.zeros((len(dec), len(LABELS)), dtype=float)
    ok = dec >= 0
    S[np.flatnonzero(ok), dec[ok]] = conf[ok]
    return S


def trust_accuracy(y: np.ndarray, dec: np.ndarray) -> float:
    """J-trust: the field as given, never ask; a parse failure counts as wrong."""
    return float((dec == y).mean())


def fit_gate(y: np.ndarray, dec: np.ndarray, conf: np.ndarray) -> dict:
    """Per-class thresholds on the self-reported confidence (class = emitted decision) with gauge ①'s procedure,
    metrics.fit_thresholds at SA_TARGET. Pre-stated fallback: if the target is not reachable, the global τ on the
    same grid with the highest fit selective accuracy (ties -> the smallest τ, i.e. the lowest ask rate)."""
    S = score_matrix(dec, conf)
    thr = metrics.fit_thresholds(S, y, SA_TARGET)
    if thr["target_reachable"]:
        taus = [round(float(t), 2) for t in thr["taus"]]
        tau_global = round(float(thr["tau_global"]), 2)
        mode = "per-class thresholds at the target"
    else:
        best = None
        for t in metrics.tau_grid():
            sa = metrics.selective_accuracy(S, y, [float(t)] * len(LABELS))
            if sa is not None and (best is None or sa > best[1] + 1e-12):
                best = (round(float(t), 2), sa)
        tau_global = None if best is None else best[0]
        taus = [1.01] * len(LABELS) if best is None else [best[0]] * len(LABELS)
        mode = f"{UNREACHABLE}: global τ with the best fit selective accuracy"
    return {
        "target_reachable_on_fit": bool(thr["target_reachable"]),
        "mode": mode,
        "tau_global": tau_global,
        "taus": taus,
        "fit": {
            "n": int(len(y)),
            "trust_accuracy": round(trust_accuracy(y, dec), 4),
            "parse_failure_rate": round(float((dec < 0).mean()), 4),
            "ask_rate": round(metrics.ask_rate(S, taus), 4),
            "selective_accuracy": (None if metrics.selective_accuracy(S, y, taus) is None
                                   else round(metrics.selective_accuracy(S, y, taus), 4)),
        },
    }


def fit_fingerprint_j(rows: list[dict]) -> dict:
    data = sorted(([str(r["id"]), r["decision"], r["confidence"], bool(r["parse_ok"])] for r in rows),
                  key=lambda r: r[0])
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return {"sha256": sha256_text(text), "n_fit": len(data), "recipe": FINGERPRINT_RECIPE_J}


def read_run(path) -> tuple[dict, list[dict], dict | None]:
    """(header, item lines, footer or None) of a gaugeJ_*.jsonl file; STOP on any other shape."""
    lines = read_jsonl(path)
    if not lines or lines[0].get("kind") != "header":
        raise SystemExit(f"STOP: {path.name} does not start with a header line")
    headers = [ln for ln in lines if ln.get("kind") == "header"]
    footers = [ln for ln in lines if ln.get("kind") == "footer"]
    items = [ln for ln in lines if ln.get("kind") == "item"]
    if len(headers) != 1 or len(footers) > 1 or len(headers) + len(footers) + len(items) != len(lines):
        raise SystemExit(f"STOP: {path.name}: {len(headers)} header(s), {len(footers)} footer(s), "
                         f"{len(lines) - len(headers) - len(footers) - len(items)} unknown line(s)")
    if footers and lines[-1].get("kind") != "footer":
        raise SystemExit(f"STOP: {path.name}: the footer is not the last line")
    return lines[0], items, (footers[0] if footers else None)


# ----------------------------------------------------------------------------------------------
# git guards (Amendment 2 committed; the code that reads the exam committed and clean)
# ----------------------------------------------------------------------------------------------
def _git(*args: str) -> tuple[int, str]:
    r = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=60, check=False)
    return r.returncode, r.stdout.strip()


def amendment2_commit() -> tuple[str, str] | None:
    """(sha, committer ISO date) of the oldest commit whose PREREG.md diff adds the Amendment 2 heading."""
    rc, out = _git("log", "--format=%H %cI", "-S", AMENDMENT2_HEADING, "--", "PREREG.md")
    rows = [ln.split() for ln in out.splitlines() if ln.strip()] if rc == 0 else []
    return (rows[-1][0], rows[-1][1]) if rows else None


def exam_guard() -> dict:
    """Everything that must hold before the exam is opened. Returns the provenance recorded in the header."""
    rc, head_prereg = _git("show", "HEAD:PREREG.md")
    if rc != 0 or AMENDMENT2_HEADING not in head_prereg.splitlines():
        raise SystemExit(f"STOP: the committed PREREG.md has no line {AMENDMENT2_HEADING!r} — commit Amendment 2 first")
    found = amendment2_commit()
    if found is None:
        raise SystemExit("STOP: cannot find the commit that introduced Amendment 2")
    sha, date = found
    if _git("merge-base", "--is-ancestor", sha, "HEAD")[0] != 0:
        raise SystemExit(f"STOP: Amendment 2 commit {sha[:7]} is not on HEAD")
    section = head_prereg.split(AMENDMENT2_HEADING, 1)[1]
    psha = prompt_sha256()
    if psha not in section:
        raise SystemExit(f"STOP: prompt sha256 {psha[:12]}… is not the one Amendment 2 registers — the prompt changed")
    _, dirty = _git("status", "--porcelain", "--", "PREREG.md", *CODE_PATHS)
    if dirty:
        raise SystemExit("STOP: PREREG.md or the gauge J code has uncommitted changes — the exam is read by committed code only")
    _, head = _git("rev-parse", "HEAD")
    on_origin = _git("merge-base", "--is-ancestor", sha, "origin/main")[0] == 0
    return {"prereg_commit": sha, "prereg_commit_date": date, "prereg_section": AMENDMENT2_HEADING,
            "code_commit": head, "amendment_on_origin_main": on_origin}


# ----------------------------------------------------------------------------------------------
# Engine side
# ----------------------------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_sampler():
    from mlx_lm.sample_utils import make_sampler as _ms

    return _ms(temp=TEMPERATURE)


def generate_item(reader, item: dict, sampler) -> dict:
    """One greedy generation for one item; seconds cover TOOLS block, render, encode, generate and parse."""
    from mlx_lm import stream_generate

    from nikasha.gauge_logit import assert_think_empty

    t0 = time.perf_counter()
    tools_block, truncated = reader.build_tools_block(item.get("functions") or [])
    msgs = [{"role": "system", "content": system_text_j()},
            {"role": "user", "content": user_text_j(tools_block, str(item.get("request", "")))}]
    prompt = reader.render(msgs)
    assert_think_empty(prompt)
    toks = reader._ensure_single_bos(list(reader.tok.encode(prompt)))
    text, n_new, finish = "", 0, None
    for resp in stream_generate(reader.model, reader.tok, toks, max_tokens=MAX_NEW_TOKENS, sampler=sampler):
        text += resp.text
        n_new = int(resp.generation_tokens)
        finish = resp.finish_reason
    parsed = parse_output(text)
    seconds = time.perf_counter() - t0
    return {"raw": text, **parsed, "seconds": round(seconds, 4), "n_new_tokens": n_new, "finish_reason": finish,
            "prompt_tokens": len(toks), "truncated": truncated}


def load():
    from nikasha.gauge_logit import load_engine

    reader, load_s = load_engine()
    sampler = make_sampler()
    warm = {"id": "warmup", "request": "ping", "functions": [{"name": "noop", "description": "does nothing", "params": []}]}
    t0 = time.perf_counter()
    generate_item(reader, warm, sampler)
    reader.gauge_item(warm, [0, 1, 2])
    reader.n_bos_fixed = 0
    return reader, sampler, load_s, time.perf_counter() - t0


def verify_sealed() -> None:
    from nikasha.gauge_logit import verify_manifest_sha

    manifest = read_json(MANIFEST_PATH)
    verify_manifest_sha(manifest, FIT_PATH, "fit.jsonl")
    verify_manifest_sha(manifest, EXAM_PATH, "exam.jsonl")


def header(split: str, n: int, extra: dict | None = None) -> dict:
    return {"kind": "header", "gauge": GAUGE, "split": split, "n_items": n, "engine": ENGINE_NAME,
            "engine_path_env": "NIKASHA_ENGINE or ~/mlx-models/" + ENGINE_NAME,
            "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha256(), "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE, "seed": SEED, "started_at": now_iso(), **(extra or {})}


def run_split(split: str, items: list[dict], out_path, extra: dict | None = None) -> None:
    """Write header, one line per item as it is produced (flushed), footer. "x" mode: never overwrites."""
    from nikasha.gauge_logit import clear_cache, validate_items

    validate_items(items, split)
    reader, sampler, load_s, warm_s = load()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    t_start = time.perf_counter()
    with open(out_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(header(split, len(items), {"load_s": round(load_s, 3), "warmup_s": round(warm_s, 3),
                                                     **(extra or {})}), ensure_ascii=False) + "\n")
        f.flush()
        for i, it in enumerate(items, 1):
            out = generate_item(reader, it, sampler)
            n_ok += int(out["parse_ok"])
            f.write(json.dumps({"kind": "item", "id": it["id"], "split": split, "label": it["label"], **out},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % PROGRESS_EVERY == 0 or i == len(items):
                print(f"{split} {i}/{len(items)} parsed {n_ok} elapsed {time.perf_counter() - t_start:.1f} s", flush=True)
                clear_cache()
        f.write(json.dumps({"kind": "footer", "n_items": len(items), "n_parse_ok": n_ok, "finished_at": now_iso(),
                            "process_s": round(time.perf_counter() - t_start, 3)}) + "\n")
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------------------------
def mode_prompt_sha() -> int:
    print(f"gauge: {GAUGE}\nprompt_version: {PROMPT_VERSION}\nprompt_sha256: {prompt_sha256()}")
    print("--- system ---\n" + system_text_j())
    print("--- user template ---\n" + USER_TEMPLATE_J)
    print("--- canonical JSON hashed (sort_keys=True, ensure_ascii=False, UTF-8) ---\n" + prompt_canonical_json())
    return 0


SMOKE_ITEMS = [
    {"id": "smoke-1", "request": "What's the weather in Paris and in Rome right now?",
     "functions": [{"name": "get_weather", "description": "Current weather for one city.", "params": ["city"]}]},
    {"id": "smoke-2", "request": "Write me a haiku about autumn.",
     "functions": [{"name": "get_stock_price", "description": "Latest price of one ticker.", "params": ["ticker"]}]},
]


def mode_smoke() -> int:
    reader, sampler, load_s, warm_s = load()
    print(f"load_s {load_s:.2f} warmup_s {warm_s:.2f}")
    for it in SMOKE_ITEMS:
        out = generate_item(reader, it, sampler)
        print(json.dumps({"id": it["id"], **out}, ensure_ascii=False))
    return 0


def mode_fit(force: bool) -> int:
    if EXAM_OUT.exists():
        raise SystemExit(f"STOP: {EXAM_OUT.name} exists — the fit split is frozen once the exam has been read")
    if FIT_OUT.exists():
        if not force:
            raise SystemExit(f"STOP: {FIT_OUT.name} exists; --force re-runs the fit split (before the exam only)")
        FIT_OUT.unlink()
    verify_sealed()
    run_split("fit", read_jsonl(FIT_PATH), FIT_OUT)
    return 0


def mode_calibrate() -> int:
    if EXAM_OUT.exists() and CALIB_OUT.exists():
        raise SystemExit(f"STOP: {EXAM_OUT.name} exists — thresholds never change after the exam read")
    head, rows, foot = read_run(FIT_OUT)
    if foot is None or len(rows) != foot["n_items"] or head["split"] != "fit":
        raise SystemExit(f"STOP: {FIT_OUT.name} is incomplete")
    y, dec, conf, _ok = arrays(rows)
    gate = fit_gate(y, dec, conf)
    calib = {"row": GAUGE, "what": "per-class thresholds on gauge J's self-reported confidence (class = emitted "
                                   "decision), fitted on the fit split only",
             "prereg_section": AMENDMENT2_HEADING, "n_fit": len(rows), "sa_target": SA_TARGET,
             "procedure": "metrics.fit_thresholds (gauge ①'s): global τ grid 0.34–0.99 step 0.01, then per-class "
                          "coordinate descent; a parse failure is an ask",
             **gate, "fit_fingerprint": fit_fingerprint_j(rows), "fit_file_sha256": sha256_file(FIT_OUT),
             "fitted_at": now_iso()}
    write_json(CALIB_OUT, calib)
    f = gate["fit"]
    print(f"gaugeJ calib: reachable {gate['target_reachable_on_fit']} taus {gate['taus']} tau_global "
          f"{gate['tau_global']} | fit trust acc {f['trust_accuracy']} parse-fail {f['parse_failure_rate']} "
          f"ask {f['ask_rate']} SA {f['selective_accuracy']} -> {CALIB_OUT.name}")
    return 0


def _pct(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"median_s": round(float(np.median(a)), 3), "p90_s": round(float(np.percentile(a, 90)), 3),
            "mean_s": round(float(a.mean()), 3)}


def mode_latency(n: int) -> int:
    """Same process, same engine load: per item, gauge ① (its full three-rotation read) then gauge J."""
    if LATENCY_OUT.exists():
        raise SystemExit(f"STOP: {LATENCY_OUT.name} exists")
    verify_sealed()
    items = read_jsonl(FIT_PATH)[:n]
    reader, sampler, load_s, warm_s = load()
    t1, tj, same = [], [], 0
    fit_raw = {}
    if FIT_OUT.exists():
        fit_raw = {r["id"]: r["raw"] for r in read_run(FIT_OUT)[1]}
    for it in items:
        a = time.perf_counter()
        reader.gauge_item(it, [0, 1, 2])
        t1.append(time.perf_counter() - a)
        out = generate_item(reader, it, sampler)
        tj.append(out["seconds"])
        same += int(fit_raw.get(it["id"]) == out["raw"])
    doc = {"what": "seconds per decision, gauge ① (three-rotation restricted-logit read) and gauge J (one greedy "
                   "generation + parse), measured per item in one session, gauge ① then gauge J on each item",
           "prereg_section": AMENDMENT2_HEADING, "split": "fit", "n_items": len(items),
           "item_ids": [it["id"] for it in items], "load_s": round(load_s, 3), "warmup_s": round(warm_s, 3),
           "gauge1": {**_pct(t1), "forwards_per_decision": 3}, "gaugeJ": {**_pct(tj), "max_new_tokens": MAX_NEW_TOKENS},
           "gaugeJ_raw_identical_to_fit_run": same if fit_raw else None,
           "percentile_rule": "numpy.percentile, linear interpolation", "measured_at": now_iso()}
    write_json(LATENCY_OUT, doc)
    print(f"latency n {len(items)}: gauge ① median {doc['gauge1']['median_s']} s p90 {doc['gauge1']['p90_s']} s | "
          f"gauge J median {doc['gaugeJ']['median_s']} s p90 {doc['gaugeJ']['p90_s']} s | J raw identical to fit "
          f"run {same}/{len(items)} -> {LATENCY_OUT.name}")
    return 0


def mode_exam() -> int:
    if EXAM_OUT.exists():
        raise SystemExit(f"STOP: {EXAM_OUT.name} exists — the exam is read once (Amendment 2)")
    if not CALIB_OUT.exists():
        raise SystemExit("STOP: fit the thresholds first (--calibrate)")
    _h, rows, _f = read_run(FIT_OUT)
    if read_json(CALIB_OUT)["fit_fingerprint"]["sha256"] != fit_fingerprint_j(rows)["sha256"]:
        raise SystemExit("STOP: gaugeJ_calib.json was not fitted on gaugeJ_fit.jsonl")
    prov = exam_guard()
    verify_sealed()
    exam = read_jsonl(EXAM_PATH)
    print(f"exam guard passed: Amendment 2 {prov['prereg_commit'][:7]} on HEAD, code {prov['code_commit'][:7]} clean; "
          f"reading {len(exam)} exam items once", flush=True)
    run_split("exam", exam, EXAM_OUT, {**prov, "calib_sha256": sha256_file(CALIB_OUT)})
    return 0


def mode_all() -> int:
    if not FIT_OUT.exists():
        mode_fit(False)
    if not CALIB_OUT.exists():
        mode_calibrate()
    if not LATENCY_OUT.exists():
        mode_latency(LATENCY_N)
    if not EXAM_OUT.exists():
        mode_exam()
    else:
        print(f"{EXAM_OUT.name} exists — the exam has been read; nothing to do")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nikasha.gauge_json",
                                 description="gauge J: JSON emission by the 12B engine (never prints item text)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt-sha", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--latency", type=int, metavar="N")
    mode.add_argument("--exam", action="store_true")
    mode.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true", help="--fit only: re-run the fit split (before the exam only)")
    args = ap.parse_args(argv)
    if args.force and not args.fit:
        ap.error("--force applies to --fit only")
    if args.prompt_sha:
        return mode_prompt_sha()
    if args.smoke:
        return mode_smoke()
    if args.fit:
        return mode_fit(args.force)
    if args.calibrate:
        return mode_calibrate()
    if args.latency is not None:
        return mode_latency(args.latency)
    if args.exam:
        return mode_exam()
    return mode_all()


if __name__ == "__main__":
    sys.exit(main())
