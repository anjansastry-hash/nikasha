# PREREG — Gate 3, gauge ① on set A (pre-registration)

Filed 2026-10-07, before any exam read. Scores never change this file; the only later edit is filling
the set-A counts from `data/seta/manifest.json` after Task 1 and before the first exam read.

## Sets
- **Set A**, built from the BFCL v3 dataset (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`, Apache-2.0) by `nikasha/data_seta.py`.
  Three labels from the ground-truth call count (no-call: irrelevance files; one-call: exactly one ground-truth
  call; multi-call: two or more). Excluded: `exec_*`, `chatable`, `java`, `javascript`, `sql`, `rest`,
  `multi_turn_*`, `live_relevance`. Items whose ground-truth count contradicts their file's category are dropped and counted.
- Target size n = 1,000, stratified 334 / 333 / 333, seed 20261007, at least 10 items from every non-empty source file.
- **Fit split**: 300 (100 per label). **Exam split**: 700 (234 / 233 / 233). Seed 20261007. `fit ∩ exam = ∅`.
- Counts as built (from `data/seta/manifest.json`, filled 2026-10-07 before any exam read): total 1000 (334 / 333 / 333); fit 300 (100 / 100 / 100); exam 700 (234 / 233 / 233). Dropped for category contradiction: 0; dropped for missing ground truth: 1. Per source file (sampled): BFCL_v3_irrelevance.json: 71 (fit 16, exam 55); BFCL_v3_live_irrelevance.json: 263 (fit 84, exam 179); BFCL_v3_simple.json: 70 (fit 15, exam 55); BFCL_v3_multiple.json: 35 (fit 12, exam 23); BFCL_v3_live_simple.json: 45 (fit 20, exam 25); BFCL_v3_live_multiple.json: 183 (fit 53, exam 130); BFCL_v3_parallel.json: 152 (fit 51, exam 101); BFCL_v3_parallel_multiple.json: 151 (fit 47, exam 104); BFCL_v3_live_parallel.json: 12 (fit 1, exam 11); BFCL_v3_live_parallel_multiple.json: 18 (fit 1, exam 17). sha256 fit.jsonl `a6246a54afa40d9d3a529d5e48f590dc6f6f753cb0ee7b24cc2e18cb3d44575c`, exam.jsonl `cb0fa1f71cd363703122ba9c50ba326700ee6f7394fc5085297855aa6ae7758e`. exam_reduced: False.
- `exam.jsonl` is opened only by programs: the gauge programs (to compute each item's output) and `score.py`
  (to score). It is never printed, grepped, or read by a person or for design; the fit split is the only design set.
  Its sha256 is in the manifest and re-checked by `score.py` and `selftest.py`.

## Labels (fixed order everywhere)
`["no-call", "one-call", "multi-call"]`

## Gauge ① — restricted-logit read
- Engine: `~/mlx-models/gemma-4-12b-8bit-text` (MLX 8-bit text tower of Gemma 4 12B; license line in `cards/`). mlx-lm 0.31.3, never patched.
- Method: the prompt names the options as single letters; each letter is asserted to be exactly one token; the rendered
  prompt is asserted to have no open or non-empty thought block (rendered with `enable_thinking=False`); one forward pass,
  logits at the last position restricted to the three letter ids; nothing is generated.
- Rotations: three cyclic letter assignments (r0: A=no-call, B=one-call, C=multi-call; r1, r2 shifted). The gauge's
  raw output per item is the mean over rotations of the restricted logits aligned to label order; its probability vector
  is the softmax of that mean at T = 1. Per-rotation logits are stored; r0 vs mean is the letter-bias line.
- Prompt v1 is fixed before the first fit read; its sha256 is recorded in `ops/PROMPTS.md`. Prompt wording, temperature
  and thresholds are designed on the fit split only and never changed after an exam read.
- Timing rule (fixed in advance): if 50 fit items × 3 rotations exceed 5 s per item, rotations drop to 1; if still
  over 5 s per item, the exam shrinks to 300 (100 per label, seed 20261007) and the manifest records it.

## Constant baseline
Two rows in the same schema: `uniform` = [1/3, 1/3, 1/3]; `majority-of-fit` = one-hot on the fit split's most common label (ties → no-call). T = 1.

## Calibration and abstention (fit split only)
- One scalar temperature T per gauge, minimising fit NLL of softmax(logits_mean / T); grid 0.05–10 (log-spaced, 200 points) then local refine.
- Per-class thresholds τ[k]: ask rate minimal subject to selective accuracy ≥ **SA_TARGET = 0.95** on the fit split
  (global τ grid 0.34–0.99 step 0.01, then per-class coordinate descent in steps of 0.01). The global τ is also recorded.
- Abstain rule: pred = argmax(p_cal); answer iff p_cal[pred] ≥ τ[pred], else ask.

## Metrics (exam, each with a 95% percentile bootstrap CI, 1,000 resamples, seed 20261007)
accuracy (argmax, no abstain) · macro-F1 · NLL before and after T · ECE (15 equal-width bins) before and after T ·
ask rate at the fixed thresholds · selective accuracy achieved at those thresholds · AUROC of max p_cal against
correctness · selective-accuracy-vs-ask-rate curve over a global τ (50 points). Two rows whose CIs overlap on a metric are
marked "within noise". "Share of the gap recovered" is not computed (needs a trained head; Gate 4) and is left absent.

## Seeds
20261007 everywhere: sampling, split, bootstrap, canary, exam reduction (if ever triggered).

## Canary (KU3) and hidden-state check (KU2)
- Canary on 20 fit items: verbatim-completion rate and digit/name-twin rate; contamination-flagged if verbatim ≥ 4/20 and twins ≤ 1/20.
  The flag only changes the README wording from "zero-shot read" to "read".
- KU2: the last hidden state of the engine has shape (1, T, hidden) and the model's own head applied to it reproduces the logits within max-abs 1e-2.

## Publication rule
**Every column that is run is published; no column is dropped after scoring.**
README.md is generated by `nikasha/readme.py` from `results/*.json` and `data/seta/manifest.json`; no number is written by hand.
