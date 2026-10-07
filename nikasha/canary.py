"""canary.py — Task 4, KU3: is BFCL in the engine's pretraining?

Fit split only (brief rule 8). Twenty fit requests (seed SEED) are cut at their whitespace
midpoint; the first half is fed to the engine as a PLAIN completion (no chat template) and
24 tokens are greedy-decoded. A verbatim hit is a generated sequence that shares a prefix of
at least 6 tokens with the true second half. Twins repeat the test on a perturbed prefix
(every ASCII digit swapped for one of the nine other digits, every capitalised word after the
first swapped for a placeholder from a fixed list) and are compared against the ORIGINAL
second half. Rule: contamination-flagged if verbatim >= 4/20 and twins <= 1/20.

Twin construction, fixed so it can be re-implemented from this text: words are whitespace-
separated; a word after the first is "capitalised" iff its FIRST character is uppercase
(leading punctuation such as a quote mark is not skipped); such a word is replaced whole, so
any digits inside it draw nothing from the rng; the placeholder cycle restarts at the first
entry for every prefix; a single random.Random(SEED) supplies every digit draw across the 20
prefixes, consumed in sample order. A prefix with no digit and no capitalised word after the
first yields a twin identical to itself (recorded as twin_identical / n_twin_identical).

Writes results/canary.json with ids, counts and token/word lengths only. Prints the two rates
and the flag. Never prints the text of any set-A item.

Run from the repo root:  $PY -m nikasha.canary
"""
from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timezone

from . import (
    ENGINE_NAME,
    ENGINE_PATH,
    FIT_PATH,
    RESULTS_DIR,
    SEED,
    read_jsonl,
    write_json,
)

N_ITEMS = 20
GEN_TOKENS = 24
MIN_PREFIX_TOKENS = 6
FLAG_MIN_VERBATIM = 4
FLAG_MAX_TWINS = 1
RULE = "contamination-flagged if verbatim >= 4/20 and twins <= 1/20"
DIGITS = "0123456789"
PLACEHOLDERS = [
    "Alder", "Birch", "Cedar", "Dunmore", "Elston", "Farrow",
    "Garnet", "Haldane", "Ingram", "Juniper", "Kestrel", "Linden",
]
OUT_PATH = RESULTS_DIR / "canary.json"


# ----------------------------------------------------------------------------- text handling

def split_request(request: str) -> tuple[str, str]:
    """Cut the request at its whitespace midpoint: (prefix, suffix)."""
    words = request.split()
    half = len(words) // 2
    return " ".join(words[:half]), " ".join(words[half:])


def make_twin(prefix: str, rng: random.Random) -> tuple[str, int, int]:
    """Perturb a prefix: every ASCII digit -> rng.choice of the nine other digits; every
    whitespace-separated word after the first whose first character is uppercase -> the next
    placeholder (cycling, restarting at the first placeholder for each prefix), keeping the
    word's trailing punctuation. A name-replaced word is replaced whole, so digits inside it
    consume no rng draw; only word[0] is tested, so a name behind leading punctuation (e.g. a
    quote mark) is left in place. Returns (twin_prefix, n_digits_swapped, n_words_swapped)."""
    out: list[str] = []
    n_digits = 0
    n_names = 0
    k = 0
    for i, word in enumerate(prefix.split()):
        if i > 0 and word[0].isupper():
            j = len(word)
            while j > 0 and not word[j - 1].isalnum():
                j -= 1
            out.append(PLACEHOLDERS[k % len(PLACEHOLDERS)] + word[j:])
            k += 1
            n_names += 1
        else:
            chars: list[str] = []
            for c in word:
                if c in DIGITS:
                    chars.append(rng.choice(DIGITS.replace(c, "")))
                    n_digits += 1
                else:
                    chars.append(c)
            out.append("".join(chars))
    return " ".join(out), n_digits, n_names


def shared_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ----------------------------------------------------------------------------- tokenizer / engine

def encode_prefix(tok, prefix: str) -> list[int]:
    """Plain completion prompt: tok.encode adds BOS exactly once (recon fact)."""
    ids = [int(t) for t in tok.encode(prefix)]
    if not ids:
        bos = getattr(tok, "bos_token_id", None)
        if bos is not None:
            ids = [int(bos)]
    return ids


def encode_suffix(tok, suffix: str) -> list[int]:
    """True-continuation ids, no special tokens, with the leading space the completion would carry."""
    text = " " + suffix
    try:
        return [int(t) for t in tok._tokenizer.encode(text, add_special_tokens=False)]
    except (AttributeError, TypeError):
        ids = [int(t) for t in tok.encode(text)]
        bos = getattr(tok, "bos_token_id", None)
        if ids and bos is not None and ids[0] == int(bos):
            ids = ids[1:]
        return ids


def greedy_generate(model, ids: list[int], n_tokens: int) -> list[int]:
    """Greedy-decode n_tokens after ids. Uses mlx_lm's generate_step, whose default sampler is
    argmax over the log-probabilities (mlx-lm 0.31.3 generate.py: `sampler or (lambda x:
    mx.argmax(x, axis=-1))`). Falls back to a cache-free manual argmax loop if the import fails."""
    import mlx.core as mx

    if not ids:
        return []
    try:
        from mlx_lm.generate import generate_step
    except ImportError:  # pragma: no cover - defensive
        generate_step = None

    out: list[int] = []
    if generate_step is not None:
        for tok_id, _logprobs in generate_step(mx.array(ids), model, max_tokens=n_tokens):
            out.append(int(tok_id))
            if len(out) >= n_tokens:
                break
    else:
        toks = list(ids)
        for _ in range(n_tokens):
            logits = model(mx.array(toks)[None])[0, -1]
            nxt = int(mx.argmax(logits).item())
            toks.append(nxt)
            out.append(nxt)
    mx.clear_cache()
    return out


def _peak_gb() -> float | None:
    try:
        import mlx.core as mx

        if hasattr(mx, "get_peak_memory"):
            return round(mx.get_peak_memory() / 1e9, 3)
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            return round(mx.metal.get_peak_memory() / 1e9, 3)
    except Exception:  # pragma: no cover - defensive
        pass
    return None


# ----------------------------------------------------------------------------- main

def prepare_items() -> list[dict]:
    """Sample the 20 fit items and build prefix / suffix / twin for each. No engine involved."""
    fit = read_jsonl(FIT_PATH)
    items = random.Random(SEED).sample(fit, N_ITEMS)
    rng = random.Random(SEED)
    prepared = []
    for it in items:
        prefix, suffix = split_request(it["request"])
        twin, n_digits, n_names = make_twin(prefix, rng)
        prepared.append({
            "id": it["id"],
            "prefix": prefix,
            "suffix": suffix,
            "twin_prefix": twin,
            "twin_identical": bool(twin == prefix),
            "prefix_words": len(prefix.split()),
            "suffix_words": len(suffix.split()),
            "digits_swapped": n_digits,
            "words_swapped": n_names,
        })
    ids = [p["id"] for p in prepared]
    assert len(ids) == N_ITEMS and len(set(ids)) == N_ITEMS, "canary sample must be 20 distinct ids"
    return prepared


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="nikasha.canary",
        description="KU3 canary: verbatim-completion test on 20 fit items plus digit/name twins. "
                    "Prints the two rates and the flag; writes results/canary.json.",
    )
    ap.parse_args()

    prepared = prepare_items()

    from mlx_lm import load

    t0 = time.time()
    model, tok = load(ENGINE_PATH)
    load_s = time.time() - t0

    rows = []
    t1 = time.time()
    for p in prepared:
        true_ids = encode_suffix(tok, p["suffix"])

        prefix_ids = encode_prefix(tok, p["prefix"])
        gen = greedy_generate(model, prefix_ids, GEN_TOKENS)
        shared = shared_prefix_len(gen, true_ids)

        twin_ids = encode_prefix(tok, p["twin_prefix"])
        twin_gen = greedy_generate(model, twin_ids, GEN_TOKENS)
        twin_shared = shared_prefix_len(twin_gen, true_ids)

        rows.append({
            "id": p["id"],
            "verbatim_hit": bool(shared >= MIN_PREFIX_TOKENS),
            "twin_hit": bool(twin_shared >= MIN_PREFIX_TOKENS),
            "shared_prefix_tokens": int(shared),
            "twin_shared_prefix_tokens": int(twin_shared),
            "twin_identical": p["twin_identical"],
            "prefix_tokens": len(prefix_ids),
            "twin_prefix_tokens": len(twin_ids),
            "suffix_tokens": len(true_ids),
            "generated_tokens": len(gen),
            "twin_generated_tokens": len(twin_gen),
            "prefix_words": p["prefix_words"],
            "suffix_words": p["suffix_words"],
            "digits_swapped": p["digits_swapped"],
            "words_swapped": p["words_swapped"],
        })
    gen_s = time.time() - t1

    verbatim_hits = sum(1 for r in rows if r["verbatim_hit"])
    twin_hits = sum(1 for r in rows if r["twin_hit"])
    n_twin_identical = sum(1 for r in rows if r["twin_identical"])
    flagged = bool(verbatim_hits >= FLAG_MIN_VERBATIM and twin_hits <= FLAG_MAX_TWINS)
    verbatim_rate = f"{verbatim_hits}/{N_ITEMS}"
    twin_rate = f"{twin_hits}/{N_ITEMS}"

    out = {
        "engine": ENGINE_NAME,
        "engine_path": ENGINE_PATH,
        "split": "fit",
        "n": N_ITEMS,
        "seed": SEED,
        "gen_tokens": GEN_TOKENS,
        "min_prefix_tokens": MIN_PREFIX_TOKENS,
        "verbatim_hits": verbatim_hits,
        "twin_hits": twin_hits,
        "verbatim_rate": verbatim_rate,
        "twin_rate": twin_rate,
        "rule": RULE,
        "flagged": flagged,
        "n_twin_identical": n_twin_identical,
        "method": {
            "completion": "plain, no chat template; tok.encode(prefix) with BOS; greedy argmax via mlx_lm generate_step",
            "suffix_ids": "tok._tokenizer.encode(' ' + suffix, add_special_tokens=False)",
            "twin": (
                "whitespace-split words; every ASCII digit -> another digit via one random.Random(SEED) "
                "consumed across the 20 prefixes in sample order; every word after the first whose FIRST "
                "character is uppercase (leading punctuation not skipped) -> next entry of the fixed "
                "placeholder list, cycle restarting at the first entry for each prefix, trailing punctuation "
                "kept; a name-replaced word is replaced whole, so digits inside it draw nothing; a prefix "
                "with no digit and no such word yields a twin identical to itself (twin_identical)"
            ),
            "placeholders": PLACEHOLDERS,
        },
        "timing": {
            "load_s": round(load_s, 3),
            "wall_s": round(gen_s, 3),
            "s_per_item": round(gen_s / N_ITEMS, 3),
            "generations": 2 * N_ITEMS,
            "peak_gb": _peak_gb(),
        },
        "ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "items": rows,
    }
    write_json(OUT_PATH, out)

    print(f"verbatim {verbatim_rate}  twins {twin_rate}  flagged {flagged}")


if __name__ == "__main__":
    main()
