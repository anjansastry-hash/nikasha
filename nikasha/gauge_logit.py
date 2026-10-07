"""gauge ① — restricted-logit read of the 12B text tower on set A.

Method (ported from the Toronto preflight kit / mcp-ask-gate host/model.py): the prompt names the
three options as single letters; each letter is asserted to be exactly one token on this tokenizer;
the rendered prompt is asserted to carry no open or non-empty thought block; nothing is generated —
one forward pass, read the logits at the last position, restrict to the three letter ids.

Three cyclic letter rotations (r0: A=no-call B=one-call C=multi-call; r1, r2 shifted). Per item the
raw output is the mean over rotations of the restricted logits ALIGNED to label order; the
probability vector is softmax of that mean at T = 1. Per-rotation logits are stored too.

CLI (run from the repo root with the venv python):
  python -m nikasha.gauge_logit --prompt-sha            # version, sha256, template strings; no model, no data
  python -m nikasha.gauge_logit --timing 50             # first N fit items x 3 rotations -> results/timing-gauge1.json
  python -m nikasha.gauge_logit --run [--rotations 3|1] [--exam-cap 300] [--force]
                                                        # all fit then all exam -> results/gauge1-logit.json
                                                        #   + results/gauge1-provenance.json (diagnostics, no items)
--rotations / --exam-cap / --force belong to --run and are rejected with --prompt-sha or --timing.

Files under results/: gauge1-logit.json is exactly the interface spec's gauge schema. Diagnostics (letter
ids and forms, rotation maps, the letter-bias line, load / warm-up / process seconds, token statistics)
go to gauge1-provenance.json, which carries no "items" list, so no downstream results/*.json glob can
mistake it for a gauge result. The every-100-items checkpoint gauge1-logit.partial.json carries
"partial_of" instead of a top-level "gauge" key for the same reason, and a stale checkpoint left by a
crashed run is removed when the next --run starts processing.

This module never prints the text of any set-A item — ids, counts, numbers and timings only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone

from nikasha import (
    CARDS_DIR,
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
    sha256_text,
    write_json,
)

try:  # mlx is only needed for --timing / --run; --prompt-sha must work without it
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None

# ----------------------------------------------------------------------------------------------
# Fixed strings and constants of prompt v1 (the spec's "Prompt v1" section, exact strings)
# ----------------------------------------------------------------------------------------------
GAUGE = "gauge1-logit"
PROMPT_VERSION = "v1"
SYSTEM_SENTENCE = (
    "You are the tool-routing gate of a coding agent. Given the user's request and the tools "
    "available, decide how many tool calls the next step needs. Answer with a single letter."
)
PHRASES = {"no-call": "no tool call", "one-call": "one tool call", "multi-call": "more than one tool call"}
USER_TEMPLATE = "TOOLS:\n{tools}\nREQUEST:\n{request}\nAnswer with one letter."
TOOLS_LINE = "{name}: {description} ({params})"
TOOLS_CAP_TOKENS = 1500
MIN_DESC_CHARS = 24
DESC_CUT_RATIO = 0.7
ELLIPSIS = "…"

LETTERS = ["A", "B", "C"]
N_ROTATIONS = 3
# ROTATION_MAPS[r][letter] = label; r0 A=no-call B=one-call C=multi-call; r1, r2 cyclically shifted
ROTATION_MAPS = [{LETTERS[j]: LABELS[(j + r) % 3] for j in range(3)} for r in range(N_ROTATIONS)]

THINK_FORMS = (("<think>", "</think>"), ("<|channel>thought", "<channel|>"))

RESULT_PATH = RESULTS_DIR / "gauge1-logit.json"
PARTIAL_PATH = RESULTS_DIR / "gauge1-logit.partial.json"
PROVENANCE_PATH = RESULTS_DIR / "gauge1-provenance.json"  # diagnostics only; never carries "items"
TIMING_PATH = RESULTS_DIR / "timing-gauge1.json"
CARD_PATH = CARDS_DIR / f"{ENGINE_NAME}.md"

CHECKPOINT_EVERY = 100
PROGRESS_EVERY = 50
TIMING_RULE_S = 5.0  # brief: > 5 s per item (3 rotations) -> r0 only; still > 5 s -> exam shrinks to 300
DEFAULT_REDUCED_EXAM = 300


# ----------------------------------------------------------------------------------------------
# Prompt v1 construction
# ----------------------------------------------------------------------------------------------
def prompt_spec() -> dict:
    """The dictionary whose canonical JSON is hashed into prompt_sha256 (spec's recipe)."""
    return {
        "version": PROMPT_VERSION,
        "system": SYSTEM_SENTENCE,
        "phrases": PHRASES,
        "user_template": USER_TEMPLATE,
        "tools_line": TOOLS_LINE,
        "tools_cap_tokens": TOOLS_CAP_TOKENS,
        "rotations": ROTATION_MAPS,
    }


def prompt_canonical_json() -> str:
    return json.dumps(prompt_spec(), sort_keys=True, ensure_ascii=False)


def prompt_sha256() -> str:
    return sha256_text(prompt_canonical_json())


def option_lines(rotation: int) -> list:
    """One line per letter, in A, B, C order, under the rotation's letter -> label map."""
    return [f"{letter}: {PHRASES[ROTATION_MAPS[rotation][letter]]}" for letter in LETTERS]


def system_text(rotation: int) -> str:
    return SYSTEM_SENTENCE + "\n" + "\n".join(option_lines(rotation))


def user_text(tools_block: str, request: str) -> str:
    return USER_TEMPLATE.format(tools=tools_block, request=request)


def letter_index_for_label(label_index: int, rotation: int) -> int:
    """Index into LETTERS of the letter that `rotation` assigns to LABELS[label_index]."""
    return (label_index - rotation) % 3


def softmax3(xs: list) -> list:
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


def check_probs(probs: list, item_id: str) -> None:
    if len(probs) != 3 or any((not math.isfinite(p)) or p < 0.0 or p > 1.0 for p in probs) \
            or abs(sum(probs) - 1.0) > 1e-6:
        raise SystemExit(f"STOP: gauge contract violated on item {item_id}: probs={probs}")


# ----------------------------------------------------------------------------------------------
# Think-block assertion (ported from preflight kit.py; the offending body is reported by length,
# never by content, so no item text can ever reach stdout)
# ----------------------------------------------------------------------------------------------
def assert_think_empty(prompt: str, forms=THINK_FORMS) -> bool:
    for open_tag, close_tag in forms:
        n_open, n_close = prompt.count(open_tag), prompt.count(close_tag)
        if n_open != n_close:
            raise SystemExit(
                f"STOP: rendered prompt has {n_open} {open_tag} and {n_close} {close_tag} "
                f"— unbalanced reasoning block, refusing to classify"
            )
        idx = 0
        for _ in range(n_open):
            i = prompt.find(open_tag, idx)
            j = prompt.find(close_tag, i + len(open_tag))
            if j < 0:
                raise SystemExit(f"STOP: rendered prompt closes {open_tag} before it opens — refusing to classify")
            body = prompt[i + len(open_tag):j]
            if body.strip():
                raise SystemExit(
                    f"STOP: rendered prompt has a NON-EMPTY {open_tag} block "
                    f"({len(body.strip())} chars) — refusing to classify"
                )
            idx = j + len(close_tag)
    return True


# ----------------------------------------------------------------------------------------------
# Engine wrapper
# ----------------------------------------------------------------------------------------------
def peak_gb() -> float | None:
    if mx is None:
        return None
    candidates = [getattr(mx, "get_peak_memory", None)]
    metal = getattr(mx, "metal", None)
    if metal is not None:
        candidates.append(getattr(metal, "get_peak_memory", None))
    for fn in candidates:
        if fn is None:
            continue
        try:
            return float(fn()) / 1e9
        except Exception:
            continue
    return None


def clear_cache() -> None:
    if mx is None:
        return
    for owner in (mx, getattr(mx, "metal", None)):
        fn = getattr(owner, "clear_cache", None) if owner is not None else None
        if fn is not None:
            try:
                fn()
            except Exception:
                pass
            return


class Reader:
    """Holds the loaded engine and performs the restricted-logit read."""

    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.bos_id = getattr(tok, "bos_token_id", None)
        # Number of FORWARD PASSES (not items) whose token list needed a BOS fix; warmup() resets it,
        # so a run reports forwards on set-A prompts only. Reported as "n_bos_fixed_forwards".
        self.n_bos_fixed = 0
        self.template_fallback = False
        self.letter_ids = []
        self.letter_forms = []
        for letter in LETTERS:
            tid, form = self._letter_id(letter)
            self.letter_ids.append(tid)
            self.letter_forms.append(form)

    # -- tokenizer helpers -----------------------------------------------------------------
    def encode_plain(self, text: str) -> list:
        """Token ids with no special tokens (tok._tokenizer.encode(add_special_tokens=False);
        fallback tok.encode with a leading BOS stripped)."""
        inner = getattr(self.tok, "_tokenizer", None)
        if inner is not None:
            try:
                return list(inner.encode(text, add_special_tokens=False))
            except TypeError:
                pass
        ids = list(self.tok.encode(text))
        if self.bos_id is not None and ids and ids[0] == self.bos_id:
            ids = ids[1:]
        return ids

    def count_tokens(self, text: str) -> int:
        return len(self.encode_plain(text))

    def _letter_id(self, letter: str):
        ids = self.encode_plain(letter)
        if len(ids) == 1:
            return int(ids[0]), letter
        ids2 = self.encode_plain(" " + letter)
        if len(ids2) == 1:
            return int(ids2[0]), " " + letter
        raise SystemExit(
            f"STOP: label '{letter}' is {len(ids)} tokens on this tokenizer ({ids}) and "
            f"' {letter}' is {len(ids2)} ({ids2}) — the restricted-logit method requires exactly one"
        )

    def render(self, msgs: list) -> str:
        try:
            try:
                return self.tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
                )
            except TypeError:
                self.template_fallback = True
                return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        except SystemExit:
            raise
        except Exception as exc:  # e.g. a template that rejects the system role
            # Only the exception's type is reported: a template error's text could echo message
            # content, and this module never prints item text.
            raise SystemExit(
                f"STOP: chat template failed to render ({type(exc).__name__}) — refusing to classify"
            ) from exc

    def _ensure_single_bos(self, toks: list) -> list:
        if self.bos_id is None:
            return toks
        while len(toks) >= 2 and toks[0] == self.bos_id and toks[1] == self.bos_id:
            toks = toks[1:]
            self.n_bos_fixed += 1
        if not toks or toks[0] != self.bos_id:
            toks = [self.bos_id] + toks
            self.n_bos_fixed += 1
        return toks

    # -- the read ---------------------------------------------------------------------------
    def read_letters(self, prompt: str):
        """One forward pass; returns (restricted logits in LETTER order as floats, n prompt tokens)."""
        assert_think_empty(prompt)
        toks = self._ensure_single_bos(list(self.tok.encode(prompt)))
        logits = self.model(mx.array(toks)[None])[0, -1]
        restricted = logits[mx.array(self.letter_ids)].astype(mx.float32)
        mx.eval(restricted)
        vals = [float(v) for v in restricted.tolist()]
        return vals, len(toks)

    # -- TOOLS block with the deterministic cap ----------------------------------------------
    def build_tools_block(self, functions: list):
        entries = []
        for fn in functions:
            params = fn.get("params")
            if params is None:
                params = []
            entries.append({
                "name": str(fn.get("name", "") or ""),
                "base": str(fn.get("description", "") or ""),
                "cut": False,
                "params": [str(p) for p in params],
            })

        def render_block(es):
            lines = []
            for e in es:
                desc = e["base"] + (ELLIPSIS if e["cut"] else "")
                lines.append(TOOLS_LINE.format(name=e["name"], description=desc, params=", ".join(e["params"])))
            return "\n".join(lines)

        truncated = False
        block = render_block(entries)
        while entries and self.count_tokens(block) > TOOLS_CAP_TOKENS:
            truncated = True
            # the function with the longest description (ties -> first; max() keeps the first maximum)
            j = max(range(len(entries)), key=lambda i: len(entries[i]["base"]))
            e = entries[j]
            new_len = max(MIN_DESC_CHARS, int(len(e["base"]) * DESC_CUT_RATIO))
            if new_len < len(e["base"]):
                e["base"] = e["base"][:new_len]
                e["cut"] = True
            else:
                # every description is already at the minimum: drop functions from the end
                entries.pop()
            block = render_block(entries)
        return block, truncated

    # -- one item -----------------------------------------------------------------------------
    def gauge_item(self, item: dict, rotations: list, rot_seconds: list | None = None) -> dict:
        tools_block, truncated = self.build_tools_block(item.get("functions") or [])
        user = user_text(tools_block, str(item.get("request", "")))
        by_rotation = []
        prompt_tokens = None
        for r in rotations:
            msgs = [{"role": "system", "content": system_text(r)}, {"role": "user", "content": user}]
            # rot_seconds[r] accumulates one whole rotation pass: template render + think assertion +
            # encode + forward. Only the TOOLS-block build above is shared across rotations, so a
            # one-rotation run costs rot_seconds[0] plus that shared overhead — nothing else.
            t0 = time.perf_counter()
            prompt = self.render(msgs)
            vals, n_tok = self.read_letters(prompt)
            if rot_seconds is not None:
                rot_seconds[r] += time.perf_counter() - t0
            # align letter-order logits to label order
            aligned = [vals[letter_index_for_label(k, r)] for k in range(3)]
            by_rotation.append(aligned)
            if prompt_tokens is None:
                prompt_tokens = n_tok
        logits_mean = [sum(row[k] for row in by_rotation) / len(by_rotation) for k in range(3)]
        if any(not math.isfinite(x) for x in logits_mean):
            raise SystemExit(f"STOP: non-finite restricted logits on item {item.get('id')}: {logits_mean}")
        probs = softmax3(logits_mean)
        check_probs(probs, str(item.get("id")))
        return {
            "logits_mean": logits_mean,
            "logits_by_rotation": by_rotation,
            "probs_raw": probs,
            "truncated": truncated,
            "prompt_tokens": prompt_tokens,
        }


def load_engine():
    if mx is None:
        raise SystemExit("STOP: mlx is not importable in this interpreter")
    from mlx_lm import load  # imported lazily so --prompt-sha never touches mlx-lm

    t0 = time.perf_counter()
    model, tok = load(ENGINE_PATH)
    load_s = time.perf_counter() - t0
    reader = Reader(model, tok)
    return reader, load_s


def warmup(reader: Reader) -> float:
    """One forward on a synthetic (non set-A) prompt so compile/alloc costs are not charged to item 1."""
    fake = {"id": "warmup", "request": "ping", "functions": [{"name": "noop", "description": "does nothing", "params": []}]}
    t0 = time.perf_counter()
    reader.gauge_item(fake, [0])
    reader.n_bos_fixed = 0  # the warm-up forward must not count towards the reported number
    return time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------------------------
def read_license_line() -> str:
    if not CARD_PATH.exists():
        raise SystemExit(f"STOP: license card missing: {CARD_PATH}")
    for line in CARD_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("license:"):
            return line.strip()
    raise SystemExit(f"STOP: no line starting with 'license:' in {CARD_PATH}")


def validate_items(items: list, split: str) -> None:
    for it in items:
        if "id" not in it or it.get("label") not in LABELS:
            raise SystemExit(f"STOP: {split} item without id or with unknown label (id={it.get('id')!r})")
        if not isinstance(it.get("request", ""), str) or not isinstance(it.get("functions", []), list):
            raise SystemExit(f"STOP: {split} item {it.get('id')!r} has malformed request/functions fields")


def verify_manifest_sha(manifest: dict, path, name: str) -> None:
    """STOP unless the manifest seals `name` with a sha256 that matches the file on disk. Task 1 seals
    both splits before this gauge ever runs, so a missing entry means set A is not sealed."""
    want = (manifest.get("sha256") or {}).get(name)
    if not isinstance(want, str) or not want:
        raise SystemExit(f"STOP: manifest carries no sha256 for {name} — set A is not sealed; run data_seta first")
    have = sha256_file(path)
    if have != want:
        raise SystemExit(f"STOP: sha256({name}) = {have} does not match the manifest's {want}")


def select_exam_subset(exam_items: list, cap: int):
    """Stratified subset of exam items: cap//3 per label, remainder to the first labels,
    drawn with random.Random(SEED); kept items keep their exam.jsonl order."""
    rng = random.Random(SEED)
    per = [cap // 3] * 3
    for k in range(cap % 3):
        per[k] += 1
    kept = set()
    counts = {}
    for k, label in enumerate(LABELS):
        idxs = [i for i, it in enumerate(exam_items) if it["label"] == label]
        n = min(per[k], len(idxs))
        chosen = rng.sample(idxs, n)
        kept.update(chosen)
        counts[label] = n
    subset = [exam_items[i] for i in sorted(kept)]
    return subset, counts


def label_counts(items: list) -> dict:
    return {lab: sum(1 for it in items if it["label"] == lab) for lab in LABELS}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------------------------
def mode_prompt_sha() -> int:
    print(f"gauge: {GAUGE}")
    print(f"prompt_version: {PROMPT_VERSION}")
    print(f"prompt_sha256: {prompt_sha256()}")
    print("--- system sentence ---")
    print(SYSTEM_SENTENCE)
    print("--- option lines per rotation (letter: phrase) ---")
    for r in range(N_ROTATIONS):
        print(f"r{r}: " + " | ".join(option_lines(r)))
    print("--- user template ---")
    print(USER_TEMPLATE)
    print("--- tools line ---")
    print(TOOLS_LINE)
    print(f"tools_cap_tokens: {TOOLS_CAP_TOKENS} (trim longest description to {int(DESC_CUT_RATIO * 100)}%, "
          f"min {MIN_DESC_CHARS} chars, then drop functions from the end)")
    print("--- canonical JSON hashed (sort_keys=True, ensure_ascii=False, UTF-8) ---")
    print(prompt_canonical_json())
    return 0


def mode_timing(n: int) -> int:
    if not FIT_PATH.exists():
        raise SystemExit(f"STOP: fit split missing: {FIT_PATH}")
    fit = read_jsonl(FIT_PATH)
    validate_items(fit, "fit")
    items = fit[:n]
    if not items:
        raise SystemExit("STOP: no fit items to time")

    reader, load_s = load_engine()
    print(f"load_s {load_s:.2f}  letter_ids {dict(zip(LETTERS, reader.letter_ids))}  "
          f"letter_forms {reader.letter_forms}", flush=True)
    warm_s = warmup(reader)

    rotations = list(range(N_ROTATIONS))
    rot_seconds = [0.0] * N_ROTATIONS
    tok_counts = []
    n_trunc = 0
    t0 = time.perf_counter()
    for i, it in enumerate(items, 1):
        out = reader.gauge_item(it, rotations, rot_seconds)
        tok_counts.append(out["prompt_tokens"])
        n_trunc += int(out["truncated"])
        if i % PROGRESS_EVERY == 0:
            print(f"timing progress {i}/{len(items)} items, elapsed {time.perf_counter() - t0:.1f} s", flush=True)
    process_s = time.perf_counter() - t0

    n_items = len(items)
    pass_s = sum(rot_seconds)                     # render + assert + encode + forward, all rotations
    overhead_s = max(0.0, process_s - pass_s)     # per-item work shared by every rotation (TOOLS block build)
    s_per_item = process_s / n_items
    s_per_pass = pass_s / (n_items * N_ROTATIONS)
    s_per_item_r0 = (rot_seconds[0] + overhead_s) / n_items   # what a --rotations 1 run would cost per item
    if s_per_item <= TIMING_RULE_S:
        rec_rot, rec_cap, decision = 3, None, f"3 rotations: {s_per_item:.2f} s/item <= {TIMING_RULE_S:g} s"
    elif s_per_item_r0 <= TIMING_RULE_S:
        rec_rot, rec_cap, decision = 1, None, (f"r0 only: 3-rotation {s_per_item:.2f} s/item > {TIMING_RULE_S:g} s, "
                                               f"r0-only {s_per_item_r0:.2f} s/item <= {TIMING_RULE_S:g} s")
    else:
        rec_rot, rec_cap, decision = 1, DEFAULT_REDUCED_EXAM, (
            f"r0 only AND exam reduced to {DEFAULT_REDUCED_EXAM}: r0-only {s_per_item_r0:.2f} s/item > {TIMING_RULE_S:g} s")
    pk = peak_gb()
    mean_tok = sum(tok_counts) / n_items

    doc = {
        "gauge": GAUGE,
        "engine": ENGINE_NAME,
        "engine_path": ENGINE_PATH,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "n_items": n_items,
        "split": "fit",
        "item_ids": [it["id"] for it in items],
        "rotations": N_ROTATIONS,
        "forwards_per_item": N_ROTATIONS,
        "load_s": round(load_s, 3),
        "warmup_s": round(warm_s, 3),
        "process_s": round(process_s, 3),
        "s_per_item": round(s_per_item, 4),
        "s_per_pass": round(s_per_pass, 4),
        "s_per_pass_by_rotation": [round(x / n_items, 4) for x in rot_seconds],
        "s_per_item_shared_overhead": round(overhead_s / n_items, 4),
        "s_per_item_r0_only": round(s_per_item_r0, 4),
        "timing_notes": "a pass = chat-template render + think assertion + encode + one forward for one rotation; "
                        "s_per_item_r0_only = r0 pass + the per-item overhead shared by all rotations "
                        "(TOOLS block build), i.e. the measured cost of a --rotations 1 run",
        "mean_prompt_tokens": round(mean_tok, 1),
        "max_prompt_tokens": max(tok_counts),
        "n_truncated": n_trunc,
        "peak_gb": None if pk is None else round(pk, 3),
        "letter_ids": dict(zip(LETTERS, reader.letter_ids)),
        "letter_forms": reader.letter_forms,
        "n_bos_fixed_forwards": reader.n_bos_fixed,
        "template_fallback_no_enable_thinking": reader.template_fallback,
        "rule": f"> {TIMING_RULE_S:g} s per item (3 rotations) -> r0 only; still > {TIMING_RULE_S:g} s -> exam shrinks to "
                f"{DEFAULT_REDUCED_EXAM} (stratified, seed {SEED})",
        "recommended_rotations": rec_rot,
        "recommended_exam_cap": rec_cap,
        "decision": decision,
        "measured_at": now_iso(),
    }
    write_json(TIMING_PATH, doc)
    print(f"load_s {load_s:.2f}")
    print(f"warmup_s {warm_s:.2f}")
    print(f"n_items {n_items}  rotations {N_ROTATIONS}")
    print(f"s_per_item {s_per_item:.3f} ({N_ROTATIONS} forwards)  s_per_pass {s_per_pass:.3f}  "
          f"s_per_item_r0_only {s_per_item_r0:.3f}")
    print(f"mean_prompt_tokens {mean_tok:.1f}  max_prompt_tokens {max(tok_counts)}  n_truncated {n_trunc}")
    print(f"peak_gb {'n/a' if pk is None else f'{pk:.2f}'}")
    print(f"decision: {decision}")
    print(f"wrote {TIMING_PATH}")
    return 0


def mode_run(rotations_n: int, exam_cap: int | None, force: bool) -> int:
    if rotations_n not in (1, 3):
        raise SystemExit("STOP: --rotations must be 3 or 1")
    if RESULT_PATH.exists() and not force:
        try:
            existing = read_json(RESULT_PATH)
        except Exception:
            existing = {}
        if isinstance(existing, dict) and "exam" in existing:
            raise SystemExit(
                f"STOP: {RESULT_PATH} already carries an 'exam' block (the exam has been scored for this gauge); "
                f"re-running the gauge would discard it. Pass --force only for a logged, deliberate re-run."
            )
    if not FIT_PATH.exists() or not EXAM_PATH.exists():
        raise SystemExit(f"STOP: set A split missing ({FIT_PATH} / {EXAM_PATH}); run data_seta first")

    # The manifest is mandatory: it is what seals the splits (Task 1 writes both sha256 before this
    # gauge ever runs). No manifest, or a manifest without both digests, is an unsealed exam -> STOP.
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"STOP: {MANIFEST_PATH} missing — set A is not sealed; run data_seta first")
    manifest = read_json(MANIFEST_PATH)
    if not isinstance(manifest, dict):
        raise SystemExit(f"STOP: {MANIFEST_PATH} is not a JSON object")
    verify_manifest_sha(manifest, FIT_PATH, "fit.jsonl")
    verify_manifest_sha(manifest, EXAM_PATH, "exam.jsonl")

    fit = read_jsonl(FIT_PATH)
    exam = read_jsonl(EXAM_PATH)
    if not fit or not exam:
        raise SystemExit(f"STOP: empty split (fit {len(fit)} items, exam {len(exam)} items); the brief fixes them at 300/700")
    validate_items(fit, "fit")
    validate_items(exam, "exam")
    fit_ids = {it["id"] for it in fit}
    exam_ids = {it["id"] for it in exam}
    if fit_ids & exam_ids:
        raise SystemExit(f"STOP: fit and exam overlap on {len(fit_ids & exam_ids)} ids")
    if len(fit_ids) != len(fit) or len(exam_ids) != len(exam):
        raise SystemExit("STOP: duplicate ids inside a split")

    reduced_counts = None
    if exam_cap is not None:
        if exam_cap <= 0 or exam_cap > len(exam):
            raise SystemExit(f"STOP: --exam-cap {exam_cap} must be in 1..{len(exam)}")
        exam, reduced_counts = select_exam_subset(exam, exam_cap)
        print(f"exam reduced to {len(exam)} items (cap {exam_cap}, seed {SEED}): {reduced_counts}", flush=True)

    license_line = read_license_line()
    psha = prompt_sha256()
    rotations = list(range(rotations_n))

    # Pre-flight passed: this run is going ahead, so a checkpoint left by a crashed earlier --run
    # must not outlive it (downstream modules glob results/*.json and know nothing about partials).
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
        print(f"removed stale checkpoint {PARTIAL_PATH.name} from an earlier run", flush=True)

    t_start = time.perf_counter()
    reader, load_s = load_engine()
    print(f"loaded engine in {load_s:.2f} s; letter_ids {dict(zip(LETTERS, reader.letter_ids))}; "
          f"letter_forms {reader.letter_forms}; rotations {rotations_n}; prompt {PROMPT_VERSION} {psha[:12]}…",
          flush=True)
    warm_s = warmup(reader)

    def header(n_items_done: int, process_s: float, partial: bool) -> dict:
        """The spec's gauge header (timing has exactly the spec's five keys). A checkpoint carries
        "partial_of" instead of a top-level "gauge" key, so calibrate/score/readme/selftest — which
        recognise a gauge result by a string "gauge" plus an "items" list — never consume one."""
        pk = peak_gb()
        total_wall = time.perf_counter() - t_start
        doc = {}
        if partial:
            doc["partial"] = True
            doc["partial_of"] = GAUGE
            doc["n_done"] = n_items_done
            doc["n_total"] = len(fit) + len(exam)
        else:
            doc["gauge"] = GAUGE
        doc.update({
            "engine": ENGINE_NAME,
            "engine_path": ENGINE_PATH,
            "license": license_line,
            "prompt_sha256": psha,
            "prompt_version": PROMPT_VERSION,
            "rotations": rotations_n,
            "seed": SEED,
            "n_fit": len(fit),
            "n_exam": len(exam),
            "labels": list(LABELS),
            "timing": {
                "s_per_item": round(process_s / n_items_done, 4) if n_items_done else 0.0,
                "wall_s": round(total_wall, 3),
                "peak_gb": None if pk is None else round(pk, 3),
                "n_items": n_items_done,
                "forwards_per_item": rotations_n,
            },
        })
        return doc

    plan = [("fit", it) for it in fit] + [("exam", it) for it in exam]
    n_total = len(plan)
    items_out = []
    n_trunc = 0
    done = {"fit": 0, "exam": 0}
    t0 = time.perf_counter()
    for i, (split, it) in enumerate(plan, 1):
        out = reader.gauge_item(it, rotations)
        n_trunc += int(out["truncated"])
        items_out.append({
            "id": it["id"],
            "split": split,
            "label": it["label"],
            "logits_mean": out["logits_mean"],
            "logits_by_rotation": out["logits_by_rotation"],
            "probs_raw": out["probs_raw"],
            "truncated": out["truncated"],
            "prompt_tokens": out["prompt_tokens"],
        })
        done[split] += 1
        if i % PROGRESS_EVERY == 0 or i == n_total:
            print(f"progress {i}/{n_total} items (fit {done['fit']}/{len(fit)}, exam {done['exam']}/{len(exam)}, "
                  f"truncated {n_trunc}) elapsed {time.perf_counter() - t0:.1f} s", flush=True)
            clear_cache()
        if i % CHECKPOINT_EVERY == 0 and i < n_total:
            doc = header(i, time.perf_counter() - t0, partial=True)
            doc["items"] = items_out
            write_json(PARTIAL_PATH, doc)
    process_s = time.perf_counter() - t0

    # letter-bias line on the fit split only: mean over fit items of (r0 logit - mean logit) per label
    letter_bias_fit = None
    if rotations_n == 3 and fit:
        fit_rows = [row for row in items_out if row["split"] == "fit"]
        letter_bias_fit = [
            round(sum(row["logits_by_rotation"][0][k] - row["logits_mean"][k] for row in fit_rows) / len(fit_rows), 4)
            for k in range(3)
        ]

    if exam_cap is not None:
        manifest["exam_reduced"] = True
        manifest["exam_cap"] = exam_cap
        manifest["exam_reduced_seed"] = SEED
        manifest["exam_reduced_counts"] = {"total": len(exam), "by_label": reduced_counts}
        manifest["exam_reduced_ids"] = [it["id"] for it in exam]
        write_json(MANIFEST_PATH, manifest)
        print(f"manifest rewritten: exam_reduced=true exam_cap={exam_cap} kept={len(exam)}", flush=True)

    # results/gauge1-logit.json: exactly the spec's schema (header + items), nothing else.
    doc = header(n_total, process_s, partial=False)
    doc["items"] = items_out
    write_json(RESULT_PATH, doc)
    try:
        PARTIAL_PATH.unlink(missing_ok=True)
    except Exception:
        pass

    # results/gauge1-provenance.json: diagnostics for ops/STATUS.md. No "items" list, so none of the
    # downstream results/*.json globs treats it as a gauge result.
    provenance = {
        "gauge": GAUGE,
        "result_file": RESULT_PATH.name,
        "engine": ENGINE_NAME,
        "engine_path": ENGINE_PATH,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": psha,
        "rotations": rotations_n,
        "letters": list(LETTERS),
        "letter_ids": dict(zip(LETTERS, reader.letter_ids)),
        "letter_forms": reader.letter_forms,
        "rotation_maps": ROTATION_MAPS,
        "think_forms_checked": [list(f) for f in THINK_FORMS],
        "template_fallback_no_enable_thinking": reader.template_fallback,
        "n_bos_fixed_forwards": reader.n_bos_fixed,
        "tools_cap_tokens": TOOLS_CAP_TOKENS,
        "n_truncated": n_trunc,
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in items_out) / n_total, 1),
        "max_prompt_tokens": max(r["prompt_tokens"] for r in items_out),
        "n_fit": len(fit),
        "n_exam": len(exam),
        "fit_by_label": label_counts(fit),
        "exam_by_label": label_counts(exam),
        "exam_reduced": exam_cap is not None,
        "exam_cap": exam_cap,
        "letter_bias_fit_r0_minus_mean": letter_bias_fit,
        "timing": {
            **doc["timing"],
            "load_s": round(load_s, 3),
            "warmup_s": round(warm_s, 3),
            "process_s": round(process_s, 3),
        },
        "mlx_version": getattr(mx, "__version__", None),
        "run_at": now_iso(),
    }
    write_json(PROVENANCE_PATH, provenance)

    t = provenance["timing"]
    print(f"done: n_fit {len(fit)} n_exam {len(exam)} rotations {rotations_n} truncated {n_trunc}")
    print(f"timing: s_per_item {t['s_per_item']} wall_s {t['wall_s']} load_s {t['load_s']} "
          f"peak_gb {t['peak_gb']} forwards_per_item {t['forwards_per_item']}")
    if letter_bias_fit is not None:
        print(f"letter-bias (fit, r0 - mean, label order): {letter_bias_fit}")
    print(f"wrote {RESULT_PATH}")
    print(f"wrote {PROVENANCE_PATH}")
    return 0


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m nikasha.gauge_logit",
        description="gauge 1: restricted-logit read of the 12B text tower on set A (never prints item text)",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt-sha", action="store_true",
                      help="print prompt version, sha256 and template strings (no model, no data)")
    mode.add_argument("--timing", type=int, metavar="N",
                      help="time the first N fit items x 3 rotations; write results/timing-gauge1.json")
    mode.add_argument("--run", action="store_true",
                      help="read all fit then all exam items; write results/gauge1-logit.json")
    ap.add_argument("--rotations", type=int, choices=(3, 1), default=None,
                    help="--run only: rotations (3 = r0,r1,r2, the default; 1 = r0 only)")
    ap.add_argument("--exam-cap", type=int, default=None, metavar="N",
                    help="--run only: stratified exam subset of N ids (seed SEED); manifest records exam_reduced")
    ap.add_argument("--force", action="store_true",
                    help="--run only: overwrite a results file that already carries an exam block")
    args = ap.parse_args(argv)

    # The --run-only options are never silently ignored under another mode.
    if not args.run and (args.rotations is not None or args.exam_cap is not None or args.force):
        ap.error("--rotations/--exam-cap/--force apply to --run only")

    if args.prompt_sha:
        return mode_prompt_sha()
    if args.timing is not None:
        if args.timing <= 0:
            raise SystemExit("STOP: --timing N must be positive")
        return mode_timing(args.timing)
    rotations_n = N_ROTATIONS if args.rotations is None else args.rotations
    return mode_run(rotations_n, args.exam_cap, args.force)


if __name__ == "__main__":
    sys.exit(main())
