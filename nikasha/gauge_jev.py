"""Jev column — an external, hosted decision model on 300 exam items (brief 10, Task D). Pre-registered in
PREREG.md, "## Amendment 1 (Gate 4)", before any call.

Modes
  subsample   draw the stratified 300-item exam subsample (100 per label, seed SEED) with gauge ①'s own
              pre-registered reducer nikasha.gauge_logit.select_exam_subset and write
              data/seta/jev-subsample.json. Reads exam.jsonl ids + labels only (after the manifest sha check);
              prints counts only.
  prompt-sha  print the request template (question, options, state keys) and its sha256; no data, no call.
  run         one POST per subsample item to the decisions endpoint (model pinned), sequentially. Every raw
              response body is stored verbatim in results/jev-raw/<id>.json (failed attempts in
              results/jev-raw/<id>.attempt<k>.err.json); every HTTP call is appended to
              results/jev-raw/_calls.jsonl, which enforces the spend cap across runs. Items already stored are
              not called again. Then `build`.
  build       assemble results/jev.json from the stored raw responses (no call).

Request (template sha256 in ops/PROMPTS.md): the question of prompt v1 — its system sentence without the
letter instruction — as a `choice` question whose three options are the plain phrases of prompt v1
(criteria: phrase -> the same phrase, label order); `state` = {"tools": the TOOLS block of prompt v1 (gauge ①'s
own renderer; every set-A item was untruncated in gauge ①, asserted per item), "request": the item's request}.

Mapping (pre-registered): `answers.tool_calls.probabilities` (per option, vendor-calibrated) -> probability
vector in label order, normalised to sum 1 (raw sum recorded); option keys outside the three phrases make the
response invalid; a missing option key counts 0. A response without `probabilities` is recorded one-hot on its
`choice` with confidence "none" (the `confidence` field measures how concentrated the distribution is; it is
stored verbatim and never used as a probability). logits_mean = ln(max(p, 1e-12)); probs_raw = softmax of that.

Retries: up to 3 retries per item (backoff 2, 4, 8 s) on a timeout / connection error, HTTP 408, 409, 425,
429, 5xx, 524, 529 or an unparsable / incomplete 200 body; HTTP 400 and 413 fail the item at once; HTTP 401,
402, 403, 404 abort the run (they are not item-specific). Spend cap: no call after the 400th. Items still
failing are listed in `failed` and counted in the README — never dropped silently.

Key: read from the environment variable OPENROUTER_API_KEY, which the calling command sets with
`source ~/.openrouter-env`. It is never printed, logged, stored or committed; every string this module prints
or writes passes redact() first.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

from nikasha import (
    CARDS_DIR,
    DATA_DIR,
    EXAM_PATH,
    LABELS,
    MANIFEST_PATH,
    RESULTS_DIR,
    SEED,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
)
from nikasha.gauge_logit import PHRASES, SYSTEM_SENTENCE, TOOLS_LINE, Reader

GAUGE = "jev"
ENGINE = "jev-1.13"
PREREG_SECTION = "## Amendment 1 (Gate 4)"
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
KEY_ENV = "OPENROUTER_API_KEY"
QUESTION = "tool_calls"
LETTER_INSTRUCTION = " Answer with a single letter."
if not SYSTEM_SENTENCE.endswith(LETTER_INSTRUCTION):
    raise SystemExit("STOP: prompt v1 system sentence no longer ends with the letter instruction")
INSTRUCTIONS = SYSTEM_SENTENCE[: -len(LETTER_INSTRUCTION)]
OPTIONS = [PHRASES[label] for label in LABELS]  # label order
PROMPT_VERSION = "jev-v1"
DOCS_READ = [
    "https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request",
    "https://openrouter.ai/docs/guides/community/jev",
    "https://openrouter.ai/docs/cookbook/building-agents/gate-tool-calls-with-jev",
    "https://openrouter.ai/blog/insights/what-is-jev/",
]

SUBSAMPLE_PATH = DATA_DIR / "jev-subsample.json"
SUBSAMPLE_N = 300
RAW_DIR = RESULTS_DIR / "jev-raw"
LEDGER = RAW_DIR / "_calls.jsonl"
RESULT_PATH = RESULTS_DIR / f"{GAUGE}.json"
CARD_PATH = CARDS_DIR / f"{ENGINE}.md"
GAUGE1_PATH = RESULTS_DIR / "gauge1-logit.json"

SPEND_CAP = 400
MAX_RETRIES = 3
BACKOFF_S = [2, 4, 8]
TIMEOUT_S = 60.0
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 524, 529}
FAIL_STATUS = {400, 413}
ABORT_STATUS = {401, 402, 403, 404}
P_FLOOR = 1e-12

# key-shaped tokens: the prefix (assembled so the literal never appears in this file) at a token start,
# followed by at least 8 key characters — so ordinary words that merely contain the prefix are untouched
_KEY_PREFIX_RE = re.compile(r"(?<![A-Za-z0-9])" + "sk" + "-" + "or" + r"-[-_A-Za-z0-9]{8,}")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def redact(text: str) -> str:
    """Remove the key value (if set) and anything shaped like an OpenRouter key from `text`."""
    key = os.environ.get(KEY_ENV) or ""
    if key and key in text:
        text = text.replace(key, "<redacted>")
    return _KEY_PREFIX_RE.sub("<redacted>", text)


def say(msg: str) -> None:
    print(redact(msg), flush=True)


def stop(msg: str):
    raise SystemExit(redact(f"STOP: {msg}"))


# ----------------------------------------------------------------------------------------------
# request template
# ----------------------------------------------------------------------------------------------
def prompt_spec() -> dict:
    return {
        "version": PROMPT_VERSION,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "question_name": QUESTION,
        "question_type": "choice",
        "instructions": INSTRUCTIONS,
        "criteria": {ph: ph for ph in OPTIONS},
        "option_order": OPTIONS,
        "state_keys": ["tools", "request"],
        "tools_line": TOOLS_LINE,
        "tools_block": "prompt v1 TOOLS block (gauge_logit.Reader.build_tools_block; every item untruncated)",
    }


def prompt_sha256() -> str:
    return sha256_text(json.dumps(prompt_spec(), sort_keys=True, ensure_ascii=False))


class _UncappedReader(Reader):
    """gauge ①'s TOOLS-block renderer without a tokenizer: the cap never triggers. Valid only for items
    gauge ① recorded as untruncated (asserted per item), whose block the cap never touched."""

    def __init__(self):  # noqa: D401 — no engine, no tokenizer
        pass

    def count_tokens(self, text: str) -> int:
        return 0


def request_body(item: dict) -> dict:
    tools_block, truncated = _UncappedReader().build_tools_block(item.get("functions") or [])
    if truncated:
        stop(f"{item.get('id')}: TOOLS block truncated without a tokenizer")
    return {
        "model": MODEL,
        "state": {"tools": tools_block, "request": str(item.get("request", ""))},
        "questions": {QUESTION: {"type": "choice", "instructions": INSTRUCTIONS,
                                 "criteria": {ph: ph for ph in OPTIONS}}},
    }


# ----------------------------------------------------------------------------------------------
# subsample
# ----------------------------------------------------------------------------------------------
def load_exam_items() -> list[dict]:
    from nikasha.gauge_logit import validate_items, verify_manifest_sha

    verify_manifest_sha(read_json(MANIFEST_PATH), EXAM_PATH, "exam.jsonl")
    exam = read_jsonl(EXAM_PATH)
    validate_items(exam, "exam")
    return exam


def mode_subsample(force: bool) -> int:
    from nikasha.gauge_logit import select_exam_subset

    if SUBSAMPLE_PATH.exists() and not force:
        stop(f"{SUBSAMPLE_PATH} exists; the subsample is drawn once (use --force only for a logged redraw)")
    exam = load_exam_items()
    subset, counts = select_exam_subset(exam, SUBSAMPLE_N)
    doc = {
        "what": "Jev subsample: stratified 300 exam items (100 per label), drawn before any call",
        "method": "nikasha.gauge_logit.select_exam_subset(exam.jsonl rows, 300): per label in LABELS order "
                  "random.Random(20261007).sample(that label's row indices, 100) with ONE generator; kept rows "
                  "in exam.jsonl order",
        "seed": SEED,
        "n": len(subset),
        "by_label": counts,
        "exam_sha256": sha256_file(EXAM_PATH),
        "drawn_at": now_iso(),
        "ids": [it["id"] for it in subset],
    }
    write_json(SUBSAMPLE_PATH, doc)
    say(f"wrote {SUBSAMPLE_PATH}: n={doc['n']} by_label={counts}")
    return 0


def load_subsample() -> dict:
    sub = read_json(SUBSAMPLE_PATH)
    if sub.get("n") != SUBSAMPLE_N or len(sub.get("ids", [])) != SUBSAMPLE_N or len(set(sub["ids"])) != SUBSAMPLE_N:
        stop(f"{SUBSAMPLE_PATH.name} does not hold {SUBSAMPLE_N} distinct ids")
    return sub


# ----------------------------------------------------------------------------------------------
# calls
# ----------------------------------------------------------------------------------------------
def ledger_rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def ledger_append(row: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(redact(json.dumps(row, ensure_ascii=False)) + "\n")


def parse_answer(text: str) -> tuple[dict | None, str | None]:
    """(answer object, None) for a usable 200 body, else (None, reason)."""
    try:
        body = json.loads(text)
    except ValueError:
        return None, "body is not JSON"
    ans = ((body.get("answers") or {}).get(QUESTION)) if isinstance(body, dict) else None
    if not isinstance(ans, dict):
        return None, f"no answers.{QUESTION}"
    if ans.get("type") != "choice" or ans.get("choice") not in OPTIONS:
        return None, "answer is not a choice among the three phrases"
    probs = ans.get("probabilities")
    if probs is not None:
        if not isinstance(probs, dict) or not set(probs) <= set(OPTIONS):
            return None, "probabilities carry keys outside the three phrases"
        try:
            vals = [float(probs.get(ph, 0.0)) for ph in OPTIONS]
        except (TypeError, ValueError):
            return None, "non-numeric probability"
        if any((not math.isfinite(v)) or v < 0 or v > 1 for v in vals) or not sum(vals) > 0:
            return None, "probabilities outside [0, 1] or summing to 0"
    return ans, None


def call_one(client, item: dict, key: str, budget: dict) -> tuple[str, str | None]:
    """Up to 1 + MAX_RETRIES attempts. Returns ("ok", None), ("failed", reason) or ("abort", reason)."""
    import httpx

    body = request_body(item)
    for attempt in range(MAX_RETRIES + 1):
        if budget["calls"] >= SPEND_CAP:
            return "failed", f"spend cap {SPEND_CAP} reached"
        budget["calls"] += 1
        t0 = time.perf_counter()
        status, err, text = None, None, ""
        try:
            r = client.post(ENDPOINT, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT_S)
            status, text = r.status_code, r.text
        except httpx.HTTPError as e:
            err = type(e).__name__
        elapsed = round(time.perf_counter() - t0, 3)
        reason = None
        if status == 200:
            ans, reason = parse_answer(text)
            if ans is not None:
                (RAW_DIR / f"{item['id']}.json").write_text(redact(text), encoding="utf-8")
                ledger_append({"n": budget["calls"], "id": item["id"], "attempt": attempt, "status": status,
                               "error": None, "elapsed_s": elapsed, "at": now_iso()})
                return "ok", None
        ledger_append({"n": budget["calls"], "id": item["id"], "attempt": attempt, "status": status,
                       "error": err or reason, "elapsed_s": elapsed, "at": now_iso()})
        if status is not None and status != 200 or reason:
            write_json(RAW_DIR / f"{item['id']}.attempt{attempt}.err.json",
                       {"status": status, "reason": reason, "body": redact(text)})
        if status in ABORT_STATUS:
            return "abort", f"HTTP {status}"
        if status in FAIL_STATUS:
            return "failed", f"HTTP {status}"
        retryable = err is not None or status in RETRY_STATUS or status == 200
        if not retryable:
            return "failed", f"HTTP {status}"
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_S[attempt])
    return "failed", f"still failing after {MAX_RETRIES} retries (last: {err or reason or f'HTTP {status}'})"


def mode_run() -> int:
    import httpx

    key = os.environ.get(KEY_ENV)
    if not key:
        stop(f"{KEY_ENV} is not set — run with `source ~/.openrouter-env` inside the same command")
    if not RESULT_PATH.parent.exists():
        stop("results/ missing")
    sub = load_subsample()
    exam = {it["id"]: it for it in load_exam_items()}
    g1 = {it["id"]: it for it in read_json(GAUGE1_PATH)["items"]}
    for iid in sub["ids"]:
        if iid not in exam:
            stop(f"subsample id {iid} is not in exam.jsonl")
        if g1.get(iid, {}).get("truncated") is not False:
            stop(f"{iid}: gauge ① did not record an untruncated TOOLS block")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    budget = {"calls": len(ledger_rows())}
    say(f"calls already made: {budget['calls']} (cap {SPEND_CAP}); items {len(sub['ids'])}; model {MODEL}")
    failed: dict[str, str] = {}
    n_ok_new = n_skip = 0
    t0 = time.perf_counter()
    with httpx.Client() as client:
        for i, iid in enumerate(sub["ids"], 1):
            if (RAW_DIR / f"{iid}.json").exists():
                n_skip += 1
                continue
            outcome, reason = call_one(client, exam[iid], key, budget)
            if outcome == "abort":
                say(f"ABORT at item {i}/{len(sub['ids'])} ({iid}): {reason}; calls {budget['calls']}")
                failed[iid] = reason
                break
            if outcome == "failed":
                failed[iid] = reason
                say(f"item {iid}: failed ({reason})")
            else:
                n_ok_new += 1
            if i % 25 == 0:
                say(f"progress {i}/{len(sub['ids'])}: ok_new {n_ok_new}, reused {n_skip}, failed {len(failed)}, "
                    f"calls {budget['calls']}, elapsed {time.perf_counter() - t0:.1f} s")
    say(f"run done: ok_new {n_ok_new}, reused {n_skip}, failed {len(failed)}, calls total {budget['calls']} "
        f"(cap {SPEND_CAP}), {time.perf_counter() - t0:.1f} s")
    return mode_build()


# ----------------------------------------------------------------------------------------------
# build results/jev.json
# ----------------------------------------------------------------------------------------------
def map_answer(ans: dict) -> tuple[list[float], dict]:
    probs = ans.get("probabilities")
    if isinstance(probs, dict):
        raw = [float(probs.get(ph, 0.0)) for ph in OPTIONS]
        s = sum(raw)
        return [v / s for v in raw], {"mapping": "probabilities", "raw_probabilities": raw, "raw_sum": round(s, 6),
                                      "missing_keys": [ph for ph in OPTIONS if ph not in probs]}
    k = OPTIONS.index(ans["choice"])
    return [1.0 if j == k else 0.0 for j in range(len(OPTIONS))], {"mapping": "choice-only", "confidence": "none"}


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def read_license_line() -> str:
    for line in CARD_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("license:"):
            return line.strip()
    stop(f"no license line in {CARD_PATH}")


def mode_build() -> int:
    sub = load_subsample()
    exam_labels = {it["id"]: it["label"] for it in load_exam_items()}
    ledger = ledger_rows()
    items, failed = [], []
    models, providers = Counter(), Counter()
    cost, tok_in, tok_out = 0.0, 0, 0
    n_choice_disagree = n_ties = 0
    for iid in sub["ids"]:
        path = RAW_DIR / f"{iid}.json"
        if not path.exists():
            last = [r for r in ledger if r.get("id") == iid]
            failed.append({"id": iid, "reason": (last[-1].get("error") or f"HTTP {last[-1].get('status')}") if last
                           else "never called"})
            continue
        text = path.read_text(encoding="utf-8")
        ans, reason = parse_answer(text)
        if ans is None:
            failed.append({"id": iid, "reason": f"stored response unusable: {reason}"})
            continue
        body = json.loads(text)
        models[str(body.get("model"))] += 1
        providers[str(body.get("provider"))] += 1
        usage = body.get("usage") or {}
        cost += float(usage.get("cost") or 0.0)
        tok_in += int(usage.get("input_tokens") or 0)
        tok_out += int(usage.get("output_tokens") or 0)
        p, meta = map_answer(ans)
        logits = [math.log(max(v, P_FLOOR)) for v in p]
        probs_raw = softmax(logits)
        if len(probs_raw) != 3 or any(not 0.0 <= v <= 1.0 for v in probs_raw) or abs(sum(probs_raw) - 1) > 1e-6:
            stop(f"gauge contract violated on {iid}")
        top = max(p)
        n_ties += int(sum(1 for v in p if v == top) > 1)
        choice_label = LABELS[OPTIONS.index(ans["choice"])]
        n_choice_disagree += int(choice_label != LABELS[p.index(top)])
        items.append({
            "id": iid, "split": "exam", "label": exam_labels[iid], "logits_mean": logits, "logits_by_rotation": [],
            "probs_raw": probs_raw, "truncated": False, "prompt_tokens": int(usage.get("input_tokens") or 0),
            "choice": choice_label, "confidence_field": ans.get("confidence"), **meta,
            "raw_file": f"results/jev-raw/{iid}.json",
        })
    mappings = Counter(it["mapping"] for it in items)
    if items and mappings.get("probabilities", 0) == len(items):
        confidence = "probabilities"
    elif items and mappings.get("choice-only", 0) == len(items):
        confidence = "none"
    else:
        confidence = "mixed" if items else "none"
    calls = len(ledger)
    doc = {
        "gauge": GAUGE,
        "engine": ENGINE,
        "engine_path": None,
        "external": True,
        "endpoint": ENDPOINT,
        "model_requested": MODEL,
        "models_returned": dict(models),
        "providers_returned": dict(providers),
        "license": read_license_line(),
        "prompt_sha256": prompt_sha256(),
        "prompt_version": PROMPT_VERSION,
        "rotations": 1,
        "seed": SEED,
        "n_fit": 0,
        "n_exam": SUBSAMPLE_N,
        "labels": list(LABELS),
        "prereg_section": PREREG_SECTION,
        "subsample": {"file": "data/seta/jev-subsample.json", "sha256": sha256_file(SUBSAMPLE_PATH), "n": sub["n"]},
        "confidence": confidence,
        "mapping_counts": dict(mappings),
        "n_success": len(items),
        "n_failed": len(failed),
        "failed": failed,
        "n_choice_disagrees_with_argmax": n_choice_disagree,
        "n_argmax_ties": n_ties,
        "calls": {"total": calls, "cap": SPEND_CAP, "retries": sum(1 for r in ledger if r.get("attempt", 0) > 0),
                  "max_retries_per_item": MAX_RETRIES, "backoff_s": BACKOFF_S, "timeout_s": TIMEOUT_S},
        "usage": {"cost_usd": round(cost, 6), "input_tokens": tok_in, "output_tokens": tok_out},
        "docs_read": DOCS_READ,
        "timing": {"s_per_item": round(sum(r.get("elapsed_s") or 0 for r in ledger) / max(1, calls), 4),
                   "wall_s": round(sum(r.get("elapsed_s") or 0 for r in ledger), 3), "peak_gb": None,
                   "n_items": len(items), "forwards_per_item": 1},
        "built_at": now_iso(),
        "items": items,
    }
    write_json(RESULT_PATH, doc)
    say(f"wrote {RESULT_PATH}: success {len(items)}, failed {len(failed)}, confidence {confidence}, "
        f"mapping {dict(mappings)}, models {dict(models)}, calls {calls}, cost ${cost:.6f}, "
        f"choice != argmax {n_choice_disagree}, argmax ties {n_ties}")
    return 0


# ----------------------------------------------------------------------------------------------
def mode_prompt_sha() -> int:
    say(f"gauge: {GAUGE}  prompt_version: {PROMPT_VERSION}  prompt_sha256: {prompt_sha256()}")
    say(json.dumps(prompt_spec(), sort_keys=True, ensure_ascii=False))
    fake = {"id": "synthetic", "request": "<request text>",
            "functions": [{"name": "<name>", "description": "<description>", "params": ["<p1>", "<p2>"]}]}
    say("request body for a synthetic item: " + json.dumps(request_body(fake), ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nikasha.gauge_jev", description="Jev column (external, hosted)")
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("subsample")
    s.add_argument("--force", action="store_true")
    sub.add_parser("prompt-sha")
    sub.add_parser("run")
    sub.add_parser("build")
    args = ap.parse_args(argv)
    if args.mode == "subsample":
        return mode_subsample(args.force)
    if args.mode == "prompt-sha":
        return mode_prompt_sha()
    if args.mode == "run":
        return mode_run()
    return mode_build()


if __name__ == "__main__":
    sys.exit(main())
