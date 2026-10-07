"""data_seta — build set A (BFCL -> three labels, n = 1,000, 300 fit / 700 exam).

Run from the repo root:  $PY -m nikasha.data_seta [--force]

Prints only counts and the two sha256s. Never prints the text of any item. After this module
has written data/seta/exam.jsonl, that file is opened only by score.py (brief, rule 4).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import random
import re
import sys
from pathlib import Path

from nikasha import (
    BFCL_DIR,
    DATA_DIR,
    EXAM_PATH,
    FIT_PATH,
    LABELS,
    MANIFEST_PATH,
    SEED,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)

# ---------------------------------------------------------------------------------------------
# Fixed design (brief, Task 1)
# ---------------------------------------------------------------------------------------------

# Brief's table, in table order. Order matters: it fixes the rng stream and the output order.
BRIEF_FILES: dict[str, list[str]] = {
    "no-call": ["BFCL_v3_irrelevance.json", "BFCL_v3_live_irrelevance.json"],
    "one-call": [
        "BFCL_v3_simple.json",
        "BFCL_v3_multiple.json",
        "BFCL_v3_live_simple.json",
        "BFCL_v3_live_multiple.json",
    ],
    "multi-call": [
        "BFCL_v3_parallel.json",
        "BFCL_v3_parallel_multiple.json",
        "BFCL_v3_live_parallel.json",
        "BFCL_v3_live_parallel_multiple.json",
    ],
}
EXCLUDED_PATTERNS = [
    "BFCL_v3_exec_*.json",
    "BFCL_v3_chatable.json",
    "BFCL_v3_java.json",
    "BFCL_v3_javascript.json",
    "BFCL_v3_sql.json",
    "BFCL_v3_rest.json",
    "BFCL_v3_multi_turn_*.json",
    "BFCL_v3_live_relevance.json",
]
TARGET_BY_LABEL = {"no-call": 334, "one-call": 333, "multi-call": 333}
FIT_PER_LABEL = 100
MIN_PER_SOURCE_FILE = 10
LABEL_RULE = {
    "no-call": "irrelevance files have no possible_answer ground truth; every item is labelled by category",
    "one-call": "len(possible_answer ground_truth) == 1 (joined on id); items with count != 1 are dropped",
    "multi-call": "len(possible_answer ground_truth) >= 2 (joined on id); items with count < 2 are dropped",
}
REASON_CONTRADICTS = "ground-truth call count contradicts the file's category"
REASON_NO_GT = "no possible_answer row for the item's id (or ground_truth not a list/dict)"
REASON_EMPTY_REQUEST = "turn 0 has no user message text"
REASON_DUPLICATE_ID = "id already seen earlier in the same file (first occurrence kept)"

# ---------------------------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------------------------


def _category(brief_name: str) -> str:
    """'BFCL_v3_live_simple.json' -> 'live_simple'."""
    m = re.fullmatch(r"BFCL(?:_v\d+)?_(.+)\.json", brief_name)
    return m.group(1) if m else Path(brief_name).stem


def _is_excluded(name: str) -> bool:
    """True when `name` matches one of the brief's exclusion globs (any 'BFCL[_vN]_' prefix)."""
    cat = _category(name)
    return any(
        fnmatch.fnmatchcase(name, pat) or fnmatch.fnmatchcase(cat, _category(pat))
        for pat in EXCLUDED_PATTERNS
    )


def resolve_files(bfcl_dir: Path) -> tuple[dict[str, str | None], dict[str, str], list[str]]:
    """Map every brief name to the file actually present (or None), recording substitutions.

    A substitute must match the brief file's category exactly (any 'BFCL[_vN]_' prefix), so e.g.
    'exec_simple' or 'live_relevance' can never stand in for 'simple' or 'live_irrelevance'.
    """
    present = sorted(p.name for p in bfcl_dir.iterdir() if p.is_file() and p.suffix == ".json")
    resolved: dict[str, str | None] = {}
    substitutions: dict[str, str] = {}
    for label_files in BRIEF_FILES.values():
        for brief_name in label_files:
            if brief_name in present:
                resolved[brief_name] = brief_name
                continue
            cat = re.escape(_category(brief_name))
            pat = re.compile(rf"^BFCL(?:_v\d+)?_{cat}\.json$")
            cands = [n for n in present if pat.match(n)]
            if cands:
                resolved[brief_name] = cands[0]
                substitutions[brief_name] = cands[0]
            else:
                resolved[brief_name] = None
    return resolved, substitutions, present


# ---------------------------------------------------------------------------------------------
# Item normalisation
# ---------------------------------------------------------------------------------------------


def _content_text(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        for k in ("text", "content"):
            if isinstance(c.get(k), str):
                return c[k]
        return ""
    if isinstance(c, list):
        return "\n".join(t for t in (_content_text(x) for x in c) if t)
    return str(c)


def extract_request(item: dict) -> str:
    """Turn 0's user messages, contents joined by '\\n'. Tolerates a turn that is a single dict."""
    q = item.get("question")
    if not isinstance(q, list) or not q:
        return ""
    turn0 = q[0]
    if isinstance(turn0, dict):
        turn0 = [turn0]
    if not isinstance(turn0, list):
        return ""
    parts = []
    for msg in turn0:
        if isinstance(msg, dict) and msg.get("role") == "user":
            t = _content_text(msg.get("content"))
            if t:  # a user message with no text contributes no (empty) line
                parts.append(t)
    return "\n".join(parts)


def extract_functions(item: dict) -> list[dict]:
    fns = item.get("function")
    if isinstance(fns, dict):
        fns = [fns]
    if not isinstance(fns, list):
        return []
    out = []
    for fn in fns:
        if not isinstance(fn, dict):
            continue
        params = fn.get("parameters")
        props = params.get("properties") if isinstance(params, dict) else None
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "params": list(props.keys()) if isinstance(props, dict) else [],
            }
        )
    return out


def call_count(gt_row: dict | None) -> int | None:
    if gt_row is None:
        return None
    gt = gt_row.get("ground_truth")
    if isinstance(gt, list):
        return len(gt)
    if isinstance(gt, dict):  # a single call object
        return 1
    return None


def _count_ok(label: str, n: int) -> bool:
    if label == "one-call":
        return n == 1
    if label == "multi-call":
        return n >= 2
    return True


def load_source(bfcl_dir: Path, actual_name: str, label: str) -> dict:
    """Read one source file, label its items, and return valid items (file order) plus counts."""
    rows = read_jsonl(bfcl_dir / actual_name)
    pa_path = bfcl_dir / "possible_answer" / actual_name
    gt_by_id: dict = {}
    n_gt_rows = 0
    n_gt_dup = 0  # duplicate ids inside the possible_answer file (first row kept)
    has_pa = pa_path.exists()
    if has_pa:
        pa_rows = read_jsonl(pa_path)
        n_gt_rows = len(pa_rows)
        for r in pa_rows:
            gt_by_id.setdefault(r.get("id"), r)
        n_gt_dup = n_gt_rows - len(gt_by_id)

    valid: list[dict] = []
    drops = {"contradicts_category": 0, "no_ground_truth": 0, "empty_request": 0, "duplicate_id": 0}
    seen: set = set()
    for item in rows:
        iid = item.get("id")
        if iid in seen:
            drops["duplicate_id"] += 1
            continue
        seen.add(iid)
        if label != "no-call":
            n = call_count(gt_by_id.get(iid))
            if n is None:
                drops["no_ground_truth"] += 1
                continue
            if not _count_ok(label, n):
                drops["contradicts_category"] += 1
                continue
        request = extract_request(item)
        if not request.strip():
            drops["empty_request"] += 1
            continue
        valid.append(
            {
                "id": iid,
                "source_file": actual_name,
                "label": label,
                "request": request,
                "functions": extract_functions(item),
            }
        )
    return {
        "items": valid,
        "n_items": len(rows),
        "n_ground_truth_rows": n_gt_rows if has_pa else None,
        "n_ground_truth_duplicate_ids": n_gt_dup if has_pa else None,
        "possible_answer_file": f"possible_answer/{actual_name}" if has_pa else None,
        "drops": drops,
    }


# ---------------------------------------------------------------------------------------------
# Allocation (proportional to n_valid, floor MIN_PER_SOURCE_FILE, largest-remainder rounding)
# ---------------------------------------------------------------------------------------------


def _largest_remainder(raw: dict[str, float], total: int, order: list[str]) -> dict[str, int]:
    base = {f: int(raw[f]) for f in order}
    rem = total - sum(base.values())
    by_frac = sorted(order, key=lambda f: -(raw[f] - base[f]))  # stable: ties keep file order
    for f in by_frac[:rem]:
        base[f] += 1
    return base


def allocate(n_valid: dict[str, int], target: int, floor: int, order: list[str]) -> tuple[dict[str, int], dict]:
    """Per-file sample sizes for one label. Returns (alloc, note-dict for the manifest)."""
    files = [f for f in order if n_valid.get(f, 0) > 0]
    floors = {f: min(floor, n_valid[f]) for f in files}
    note: dict = {"target": target, "floor": floor, "floors_scaled": False, "method": "proportional-to-n_valid, floor per non-empty file, largest-remainder rounding"}
    if sum(n_valid[f] for f in files) < target:
        raise SystemExit(f"STOP: label target {target} exceeds total valid items {sum(n_valid[f] for f in files)}")
    lower = dict(floors)  # lower bound checked below; relaxed to 0 when the floors must be scaled
    if sum(floors.values()) > target:
        tot = sum(floors.values())
        raw = {f: target * floors[f] / tot for f in files}
        alloc = _largest_remainder(raw, target, files)
        lower = {f: 0 for f in files}
        note["floors_scaled"] = True
        note["floors_scaled_note"] = f"floors alone sum to {tot} > target {target}; floors scaled proportionally"
    else:
        fixed: dict[str, int] = {}
        free = list(files)
        remaining = target
        while free:
            tot = sum(n_valid[f] for f in free)
            raw = {f: remaining * n_valid[f] / tot for f in free}
            pinned: tuple[str, int] | None = None
            for f in free:
                if raw[f] < floors[f]:
                    pinned = (f, floors[f])
                    break
                if raw[f] > n_valid[f]:
                    pinned = (f, n_valid[f])
                    break
            if pinned is None:
                break
            # One pin per pass: the shares of the remaining files are recomputed from the new
            # `remaining`/`tot` before the next file is judged.
            f, k = pinned
            fixed[f] = k
            remaining -= k
            free.remove(f)
        if free:
            alloc = {**fixed, **_largest_remainder(raw, remaining, free)}
        else:
            if remaining != 0:
                raise SystemExit(f"STOP: allocation cannot place {remaining} items (every file capped)")
            alloc = dict(fixed)
    alloc = {f: alloc[f] for f in files}
    assert sum(alloc.values()) == target, (alloc, target)
    assert all(lower[f] <= alloc[f] <= n_valid[f] for f in files), (alloc, lower, n_valid)
    return alloc, note


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------


def bfcl_revision(bfcl_dir: Path) -> str | None:
    """The Hugging Face dataset commit the local copy was downloaded at: line 1 of every
    .cache/huggingface/download/*.metadata that `hf download --local-dir` writes. None if absent or mixed."""
    revs = {f.read_text(encoding="utf-8").split("\n", 1)[0].strip()
            for f in (bfcl_dir / ".cache" / "huggingface" / "download").glob("*.metadata")}
    return revs.pop() if len(revs) == 1 else None


def build(bfcl_dir: Path) -> tuple[list[dict], list[dict], dict]:
    resolved, substitutions, present = resolve_files(bfcl_dir)
    used = {a for a in resolved.values() if a}
    unused = [n for n in present if n not in used]
    excluded_present = [n for n in unused if _is_excluded(n)]  # the brief's exclusions, by name
    unexpected = [n for n in unused if not _is_excluded(n)]  # unused and not covered by the brief

    sources: dict[str, dict] = {}  # brief_name -> load_source(...) result (+ label, actual)
    for label, names in BRIEF_FILES.items():
        for brief_name in names:
            actual = resolved[brief_name]
            if actual is None:
                sources[brief_name] = {
                    "items": [], "n_items": 0, "n_ground_truth_rows": None, "n_ground_truth_duplicate_ids": None,
                    "possible_answer_file": None,
                    "drops": {"contradicts_category": 0, "no_ground_truth": 0, "empty_request": 0, "duplicate_id": 0},
                    "label": label, "actual": None,
                }
                continue
            src = load_source(bfcl_dir, actual, label)
            src["label"] = label
            src["actual"] = actual
            sources[brief_name] = src

    # --- sampling: one rng, labels in LABELS order, files in table order
    rng = random.Random(SEED)
    sampled_by_label: dict[str, list[dict]] = {}
    alloc_by_label: dict[str, dict] = {}
    n_sampled_by_file: dict[str, int] = {}
    for label in LABELS:
        order = BRIEF_FILES[label]
        n_valid = {b: len(sources[b]["items"]) for b in order}
        alloc, note = allocate(n_valid, TARGET_BY_LABEL[label], MIN_PER_SOURCE_FILE, order)
        alloc_by_label[label] = {"note": note, "by_file": {}}
        picked: list[dict] = []
        for b in order:
            k = alloc.get(b, 0)
            n_sampled_by_file[b] = k
            alloc_by_label[label]["by_file"][sources[b]["actual"] or b] = k
            if k:
                picked.extend(rng.sample(sources[b]["items"], k))
        sampled_by_label[label] = picked

    # --- split: a second rng with the same seed; shuffle ids within label, first 100 = fit
    rng2 = random.Random(SEED)
    fit_rows: list[dict] = []
    exam_rows: list[dict] = []
    for label in LABELS:
        rows = sampled_by_label[label]
        by_id = {r["id"]: r for r in rows}
        assert len(by_id) == len(rows), f"duplicate ids within label {label}"
        ids = [r["id"] for r in rows]
        rng2.shuffle(ids)
        fit_rows.extend(by_id[i] for i in ids[:FIT_PER_LABEL])
        exam_rows.extend(by_id[i] for i in ids[FIT_PER_LABEL:])

    # --- assertions
    fit_ids = {r["id"] for r in fit_rows}
    exam_ids = {r["id"] for r in exam_rows}
    assert len(fit_ids) == len(fit_rows) and len(exam_ids) == len(exam_rows), "duplicate ids across labels"
    assert not (fit_ids & exam_ids), "fit ids and exam ids overlap"
    n_total = sum(TARGET_BY_LABEL.values())
    assert len(fit_rows) + len(exam_rows) == n_total == 1000, (len(fit_rows), len(exam_rows))
    assert len(fit_rows) == FIT_PER_LABEL * len(LABELS) == 300, len(fit_rows)
    assert len(exam_rows) == n_total - 300 == 700, len(exam_rows)
    for label in LABELS:
        assert sum(r["label"] == label for r in fit_rows) == FIT_PER_LABEL, label
        assert sum(r["label"] == label for r in exam_rows) == TARGET_BY_LABEL[label] - FIT_PER_LABEL, label
    for r in fit_rows + exam_rows:
        assert set(r) == {"id", "source_file", "label", "request", "functions"}, sorted(r)
        assert r["label"] in LABELS

    # --- manifest
    def _n_fit(actual: str) -> int:
        return sum(r["source_file"] == actual for r in fit_rows)

    def _n_exam(actual: str) -> int:
        return sum(r["source_file"] == actual for r in exam_rows)

    files: dict[str, dict] = {}
    missing: list[str] = []
    drop_by_file = {k: {} for k in ("contradicts_category", "no_ground_truth", "empty_request", "duplicate_id")}
    for b, src in sources.items():
        actual = src["actual"]
        key = actual or b
        if actual is None:
            missing.append(b)
        d = src["drops"]
        for k, v in d.items():
            if v:
                drop_by_file[k][key] = v
        files[key] = {
            "label": src["label"],
            "brief_name": b,
            "present": actual is not None,
            "possible_answer_file": src["possible_answer_file"],
            "n_items": src["n_items"],
            "n_ground_truth_rows": src["n_ground_truth_rows"],
            "n_ground_truth_duplicate_ids": src["n_ground_truth_duplicate_ids"],
            "n_valid": len(src["items"]),
            "n_dropped": d["contradicts_category"],
            "n_dropped_all": src["n_items"] - len(src["items"]),  # == n_items - n_valid, all four reasons
            "n_no_ground_truth": d["no_ground_truth"],
            "n_empty_request": d["empty_request"],
            "n_duplicate_id": d["duplicate_id"],
            "n_sampled": n_sampled_by_file.get(b, 0),
            "n_fit": _n_fit(actual) if actual else 0,
            "n_exam": _n_exam(actual) if actual else 0,
        }

    def _by_label(rows: list[dict]) -> dict[str, int]:
        return {lab: sum(r["label"] == lab for r in rows) for lab in LABELS}

    manifest = {
        "name": "set A",
        "source": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "bfcl_revision": bfcl_revision(bfcl_dir),
        "source_license": "Apache-2.0",
        "bfcl_dir": str(bfcl_dir),
        "labels": list(LABELS),
        "seed": SEED,
        "label_rule": LABEL_RULE,
        "files": files,
        "dropped": {
            "total": sum(drop_by_file["contradicts_category"].values()),
            "by_file": drop_by_file["contradicts_category"],
            "reason": REASON_CONTRADICTS,
            "no_ground_truth": {"total": sum(drop_by_file["no_ground_truth"].values()), "by_file": drop_by_file["no_ground_truth"], "reason": REASON_NO_GT},
            "empty_request": {"total": sum(drop_by_file["empty_request"].values()), "by_file": drop_by_file["empty_request"], "reason": REASON_EMPTY_REQUEST},
            "duplicate_id": {"total": sum(drop_by_file["duplicate_id"].values()), "by_file": drop_by_file["duplicate_id"], "reason": REASON_DUPLICATE_ID},
            # sum over all four reasons == sum of files[*].n_dropped_all == Σ(n_items - n_valid)
            "all_reasons_total": sum(sum(drop_by_file[k].values()) for k in drop_by_file),
        },
        "excluded_patterns": EXCLUDED_PATTERNS,
        "excluded_files": excluded_present,
        "unexpected_files": unexpected,
        "name_substitutions": substitutions,
        "missing_files": missing,
        "allocation": alloc_by_label,
        "counts": {
            "total": len(fit_rows) + len(exam_rows),
            "by_label": _by_label(fit_rows + exam_rows),
            "fit": {"total": len(fit_rows), "by_label": _by_label(fit_rows)},
            "exam": {"total": len(exam_rows), "by_label": _by_label(exam_rows)},
        },
        "targets": {"by_label": TARGET_BY_LABEL, "fit_per_label": FIT_PER_LABEL},
        "min_per_source_file": MIN_PER_SOURCE_FILE,
        "sampling": "random.Random(SEED); labels in LABELS order, files in brief-table order; rng.sample(valid items in file order, k)",
        "split": "random.Random(SEED) (fresh); per label shuffle sampled ids, first 100 fit, rest exam; rows written in that order",
        "sha256": {},  # filled after the files are written
        "exam_reduced": False,
        "built_at": None,
    }
    return fit_rows, exam_rows, manifest


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build set A (BFCL -> no-call / one-call / multi-call; 300 fit / 700 exam).")
    ap.add_argument("--force", action="store_true", help="overwrite an existing exam.jsonl (re-seals the manifest)")
    args = ap.parse_args(argv)

    bfcl_dir = Path(BFCL_DIR)
    if not bfcl_dir.is_dir():
        print(f"STOP: BFCL dir not found: {bfcl_dir}", file=sys.stderr)
        return 2
    if EXAM_PATH.exists() and not args.force:
        print(f"refusing to overwrite existing {EXAM_PATH} (sealed); pass --force to rebuild", file=sys.stderr)
        return 1

    fit_rows, exam_rows, manifest = build(bfcl_dir)

    # Loud, not silent: drops the brief does not name, ground-truth oddities and stray files.
    # (ids/counts/file names only — never item text.)
    d = manifest["dropped"]
    for kind in ("empty_request", "duplicate_id"):
        if d[kind]["total"]:
            print(
                f"WARNING: {d[kind]['total']} item(s) dropped for {kind} (not a brief rule; this "
                f"shifts the sample stream); per file: {d[kind]['by_file']}",
                file=sys.stderr,
            )
    gt_dups = {n: f["n_ground_truth_duplicate_ids"] for n, f in manifest["files"].items() if f["n_ground_truth_duplicate_ids"]}
    if gt_dups:
        print(f"WARNING: duplicate ids inside possible_answer (first row kept); per file: {gt_dups}", file=sys.stderr)
    if manifest["unexpected_files"]:
        print(
            f"WARNING: unused .json files in {bfcl_dir} not covered by the brief's exclusions: "
            f"{manifest['unexpected_files']}",
            file=sys.stderr,
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(FIT_PATH, fit_rows)
    write_jsonl(EXAM_PATH, exam_rows)
    manifest["sha256"] = {"fit.jsonl": sha256_file(FIT_PATH), "exam.jsonl": sha256_file(EXAM_PATH)}
    manifest["built_at"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_json(MANIFEST_PATH, manifest)

    c = manifest["counts"]
    print(f"set A total={c['total']} by_label={c['by_label']}")
    print(f"fit  total={c['fit']['total']} by_label={c['fit']['by_label']}")
    print(f"exam total={c['exam']['total']} by_label={c['exam']['by_label']}")
    d = manifest["dropped"]
    print(
        f"dropped: contradicts_category={d['total']} no_ground_truth={d['no_ground_truth']['total']} "
        f"empty_request={d['empty_request']['total']} duplicate_id={d['duplicate_id']['total']}"
    )
    if manifest["name_substitutions"]:
        print(f"name_substitutions={manifest['name_substitutions']}")
    if manifest["missing_files"]:
        print(f"missing_files={manifest['missing_files']}")
    for name, f in manifest["files"].items():
        print(
            f"  {name}: label={f['label']} n_items={f['n_items']} n_valid={f['n_valid']} "
            f"n_dropped={f['n_dropped']} n_no_gt={f['n_no_ground_truth']} "
            f"n_sampled={f['n_sampled']} n_fit={f['n_fit']} n_exam={f['n_exam']}"
        )
    print(f"sha256 fit.jsonl  {manifest['sha256']['fit.jsonl']}")
    print(f"sha256 exam.jsonl {manifest['sha256']['exam.jsonl']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
