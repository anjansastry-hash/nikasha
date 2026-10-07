"""KU2 — hidden-state shape check for gauge ③ (Task 5 of the Gate 3 brief).

One fit prompt (the FIRST item of fit.jsonl, prompt v1, rotation r0) goes through the engine's
inner model. The prompt is rendered by gauge ①'s own helpers in nikasha.gauge_logit (TOOLS block
cap, system/user text, chat template, think-block assertion, single-BOS token stream), so it is
the prompt gauge ① sees for the same item. The inner module is found by inspection of the loaded
object (never by patching mlx-lm): the first of `language_model.model`, `model`, `language_model`
that is callable on the token array and returns a rank-3 array whose last axis is the text hidden
size. We assert the hidden state has shape (1, T, hidden), apply the model's own head to the LAST
POSITION h[:, -1] (tied embedding `as_linear` or `lm_head`, then the final logit softcap when the
class has one), and compare with `model(toks)[0, -1]`; pass iff max |diff| <= 1e-2 (8-bit engine).

Writes results/ku2-hidden.json. Prints only attr_path, hidden shape, max_abs_diff and PASS/FAIL.
No set-A text is ever printed — ids, counts and numbers only.

Run from the repo root (no arguments):  $PY -m nikasha.ku2_hidden
"""
from __future__ import annotations

import argparse
import importlib
import sys
from datetime import datetime, timezone

from nikasha import ENGINE_NAME, ENGINE_PATH, FIT_PATH, RESULTS_DIR, read_jsonl, write_json
from nikasha.gauge_logit import (
    PROMPT_VERSION,
    Reader,
    assert_think_empty,
    prompt_sha256,
    system_text,
    user_text,
)

TOLERANCE = 1e-2
OUT_PATH = RESULTS_DIR / "ku2-hidden.json"
INNER_CANDIDATES = ("language_model.model", "model", "language_model")
ROTATION = 0  # r0: A=no-call, B=one-call, C=multi-call
RENDER_SOURCE = "nikasha.gauge_logit (Reader.build_tools_block + system_text/user_text + Reader.render)"


# --------------------------------------------------------------------------------------------
# Prompt v1 through gauge ①'s own code — no local copy of the renderer
# --------------------------------------------------------------------------------------------
def render_prompt(reader: Reader, item: dict, rotation: int = ROTATION) -> tuple:
    """Gauge ①'s v1 prompt for one item under one rotation. Returns (prompt, truncated)."""
    tools, truncated = reader.build_tools_block(item.get("functions") or [])
    msgs = [
        {"role": "system", "content": system_text(rotation)},
        {"role": "user", "content": user_text(tools, str(item.get("request", "")))},
    ]
    prompt = reader.render(msgs)
    assert_think_empty(prompt)
    return prompt, truncated


# --------------------------------------------------------------------------------------------
# Model inspection
# --------------------------------------------------------------------------------------------
def _resolve(root, path: str):
    """Follow a dotted attribute path. Returns (object, parent) or (None, None)."""
    obj, parent = root, None
    for part in path.split("."):
        parent = obj
        obj = getattr(obj, part, None)
        if obj is None:
            return None, None
    return obj, parent


def find_hidden_size(model):
    for owner in (getattr(model, "language_model", None), model):
        if owner is None:
            continue
        hs = getattr(getattr(owner, "args", None), "hidden_size", None)
        if isinstance(hs, int) and hs > 0:
            return hs, f"{'language_model.' if owner is not model else ''}args.hidden_size"
    tc = getattr(getattr(model, "args", None), "text_config", None)
    if isinstance(tc, dict) and isinstance(tc.get("hidden_size"), int):
        return tc["hidden_size"], "args.text_config['hidden_size']"
    return None, None


def discover_inner(model, x, hidden: int, mx):
    """First candidate (in INNER_CANDIDATES order) that is callable on x and returns a rank-3
    array with last axis == hidden. Returns (inner, parent, attr_path, h, tried)."""
    tried = []
    for path in INNER_CANDIDATES:
        cand, parent = _resolve(model, path)
        if cand is None:
            tried.append({"attr_path": path, "present": False})
            continue
        if not callable(cand):
            tried.append({"attr_path": path, "present": True, "callable": False})
            continue
        try:
            out = cand(x)
            mx.eval(out)
            shape = [int(s) for s in out.shape]
            entry = {"attr_path": path, "present": True, "callable": True, "ndim": int(out.ndim),
                     "shape": shape, "dtype": str(out.dtype)}
            tried.append(entry)
            if out.ndim == 3 and shape[-1] == hidden:
                entry["selected"] = True
                return cand, parent, path, out, tried
        except Exception as e:  # noqa: BLE001 — recorded, never printed with item text
            tried.append({"attr_path": path, "present": True, "callable": True, "error": type(e).__name__})
    return None, None, None, None, tried


def _lookup(owner, name):
    v = getattr(owner, name, None)
    if v is None:
        v = getattr(getattr(owner, "args", None), name, None)
    return v


def make_head(inner, text_model, mx):
    """The model's own head: tied embedding `as_linear` or `lm_head`, then the final logit softcap
    when the class defines one. Returns (head_fn, description, info)."""
    tie = _lookup(text_model, "tie_word_embeddings")
    if tie is None:
        tie = getattr(text_model, "lm_head", None) is None
    tie = bool(tie)
    cap = _lookup(text_model, "final_logit_softcapping")
    cap = float(cap) if cap is not None else None

    if tie:
        emb = getattr(inner, "embed_tokens", None)
        if emb is None or not hasattr(emb, "as_linear"):
            raise AttributeError("tied head requested but inner.embed_tokens.as_linear is missing")
        proj, proj_name = emb.as_linear, "embed_tokens.as_linear"
    else:
        lm_head = getattr(text_model, "lm_head", None)
        if lm_head is None:
            raise AttributeError("untied head requested but text model has no lm_head")
        proj, proj_name = lm_head, "lm_head"

    softcap_impl = None
    softcap = None
    if cap is not None:
        mod = sys.modules.get(type(text_model).__module__)
        own = getattr(mod, "logit_softcap", None) if mod is not None else None
        if callable(own):
            softcap, softcap_impl = (lambda z: own(cap, z)), f"{type(text_model).__module__}.logit_softcap"
        else:
            softcap, softcap_impl = (lambda z: cap * mx.tanh(z / cap)), "cap * tanh(x / cap)"

    def head(z):
        y = proj(z)
        if softcap is not None:
            y2 = softcap(y)
            # sanity: same shape and bounded by the cap; otherwise fall back to the plain formula
            if tuple(y2.shape) != tuple(y.shape) or float(mx.max(mx.abs(y2))) > cap * (1 + 1e-3):
                y2 = cap * mx.tanh(y / cap)
            y = y2
        return y

    desc = proj_name + (f" + logit_softcap({cap})" if cap is not None else "")
    info = {"tie_word_embeddings": tie, "final_logit_softcapping": cap, "softcap_impl": softcap_impl}
    return head, desc, info


def _versions():
    # the top-level `mlx` package has no __version__ (namespace package); it lives on mlx.core
    out = {}
    for key, modname in (("mlx", "mlx.core"), ("mlx_lm", "mlx_lm")):
        try:
            out[key] = str(importlib.import_module(modname).__version__)
        except Exception:
            out[key] = None
    return out


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _emit(rec: dict) -> int:
    """Write the JSON and print the four permitted lines.

    Exit 0 when the measurement was produced — a recorded FAIL is a measurement, not a crash, and
    readme.py prints PASS/FAIL from the JSON (so `make all` continues to readme/selftest).
    Exit 2 when the check could not be completed (rec["error"] set, max_abs_diff null)."""
    write_json(OUT_PATH, rec)
    print(f"attr_path={rec.get('attr_path')}")
    print(f"hidden_shape={rec.get('hidden_shape')}")
    print(f"max_abs_diff={rec.get('max_abs_diff')}")
    print("PASS" if rec.get("pass") else "FAIL")
    return 0 if rec.get("max_abs_diff") is not None else 2


# --------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="nikasha.ku2_hidden",
        description="KU2: hidden-state shape check on the first fit item (gauge ③ prerequisite). "
                    "Takes no arguments: reads data/seta/fit.jsonl, writes results/ku2-hidden.json.",
    )
    ap.parse_args(argv)

    try:
        rows = read_jsonl(FIT_PATH)
    except OSError as e:
        print(f"ERROR: cannot read fit split ({type(e).__name__})", file=sys.stderr)
        return 2
    if not rows:
        print("ERROR: fit split is empty", file=sys.stderr)
        return 2
    item = rows[0]
    item_id = item.get("id")

    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(ENGINE_PATH)

    rec = {
        "engine": ENGINE_NAME,
        "engine_path": ENGINE_PATH,
        "attr_path": None,
        "head": None,
        "hidden_shape": None,
        "hidden_size": None,
        "max_abs_diff": None,
        "tolerance": TOLERANCE,
        "pass": False,
        "item_id": item_id,
        "prompt_tokens": None,
        "argmax_match": None,
        "split": "fit",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "rotation": ROTATION,
        "render_source": RENDER_SOURCE,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "versions": _versions(),
        "checked_at": _now(),
    }

    # 1. prompt -> tokens, through gauge ①'s own code path (Reader asserts each letter is a single
    #    token, exactly as gauge ① does; the token stream is the one read_letters() would see)
    try:
        reader = Reader(model, tok)
        prompt, truncated = render_prompt(reader, item, ROTATION)
        toks = reader._ensure_single_bos(list(tok.encode(prompt)))
    except (Exception, SystemExit) as e:  # noqa: BLE001 — type name only, never item text
        rec["error"] = f"prompt render failed: {type(e).__name__}"
        return _emit(rec)
    rec.update({
        "prompt_tokens": len(toks),
        "truncated": truncated,
        "tokenization": "tok.encode(prompt) then Reader._ensure_single_bos (gauge ①'s read path)",
        "n_bos_fixed": reader.n_bos_fixed,
        "template_fallback": reader.template_fallback,
    })
    x = mx.array(toks)[None]

    # 2. hidden size from the text args
    hidden, hidden_src = find_hidden_size(model)
    rec["hidden_size"] = hidden
    rec["hidden_size_source"] = hidden_src
    if hidden is None:
        rec["error"] = "hidden_size not found on language_model.args / args"
        return _emit(rec)

    # 3. inner module by inspection
    inner, text_model, attr_path, h, tried = discover_inner(model, x, hidden, mx)
    rec["candidates"] = tried
    rec["attr_path"] = attr_path
    if inner is None:
        rec["error"] = "no candidate returned a rank-3 array with last axis == hidden_size"
        return _emit(rec)
    shape = [int(s) for s in h.shape]
    rec["hidden_shape"] = shape
    rec["hidden_dtype"] = str(h.dtype)
    rec["text_model_class"] = f"{type(text_model).__module__}.{type(text_model).__name__}"
    if shape != [1, len(toks), hidden]:
        rec["error"] = f"hidden shape {shape} != [1, {len(toks)}, {hidden}]"
        return _emit(rec)

    # 4. the model's own head
    try:
        head, head_desc, head_info = make_head(inner, text_model, mx)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"head construction failed: {type(e).__name__}"
        return _emit(rec)
    rec["head"] = head_desc
    rec.update(head_info)

    # 5. reference logits from the full model, last position, float32
    ref = model(x)[0, -1].astype(mx.float32)
    mx.eval(ref)
    rec["vocab_size"] = int(ref.shape[-1])

    # 6. THE check (brief, Task 5): the head applied to the LAST POSITION, h[:, -1] of shape
    #    (1, hidden) — the quantity gauge ③ will consume. This feeds max_abs_diff and pass.
    lp_raw = head(h[:, -1])
    rec["logits_dtype"] = str(lp_raw.dtype)
    lp = lp_raw[0].astype(mx.float32)
    diff = float(mx.max(mx.abs(lp - ref)))
    rec["max_abs_diff"] = diff
    rec["argmax_match"] = bool(int(mx.argmax(lp)) == int(mx.argmax(ref)))
    rec["head_input"] = "h[:, -1] (1, hidden)"
    rec["pass"] = bool(diff <= TOLERANCE) and shape == [1, len(toks), hidden]

    # 7. Diagnostic only (never used for pass): the head over the whole hidden state with the last
    #    position sliced after the head — the model's own forward decomposed into two calls, so it
    #    reproduces the reference to the bit when attr_path and the head are right.
    try:
        full = head(h)[0, -1].astype(mx.float32)
        rec["full_h"] = {
            "head_input": "h (1, T, hidden) through the head, position -1 taken after the head",
            "max_abs_diff": float(mx.max(mx.abs(full - ref))),
            "argmax_match": bool(int(mx.argmax(full)) == int(mx.argmax(ref))),
            "note": "diagnostic; not used for pass",
        }
    except Exception as e:  # noqa: BLE001
        rec["full_h"] = {"head_input": "h (1, T, hidden)", "error": type(e).__name__}

    return _emit(rec)


if __name__ == "__main__":
    sys.exit(main())
