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

## Amendment 1 (Gate 4)

Filed 2026-10-07 06:45 ET, before any exam read of brief 10 (rule 12: amend, never rewrite). Everything above is
unchanged and Gate 3 numbers are never recomputed. Before this amendment, gauge ③ and the labels-needed heads were
run on the fit split only (features of the 300 fit items into a scratch file, head and labels-needed dry runs; all
logged in `ops/COMMAND_LOG.md`), and the Jev subsample ids below were drawn from `exam.jsonl` ids and labels only.
No exam feature, Jev call or exam score of any Gate 4 gauge exists.

### New gauge versions
- **gauge3-probe v1** (gauge ③: frozen 12B hidden state + linear head), `nikasha/gauge_probe.py`.
  - Features: for every set-A item, prompt v1 (sha256 `f45f503ecce61805817028a7abca55e14bc4c50da0e4f3fb2b08ccfef78641d4`)
    at rotation r0, rendered by gauge ①'s own code; one forward pass of the inner text model at the KU2 path
    `language_model.model`; the final-normed hidden state (`Gemma4TextModel.norm`) at the last position, cast to
    float32 (dimension 3840). Stored in `results/gauge3-features.npy` + `results/gauge3-features.index.json` with
    its sha256 (committed if ≤ 50 MB, else git-ignored with the sha256 recorded). Per item, the model's own tied head
    applied to that vector and restricted to the letter ids must reproduce gauge ①'s stored r0 logits (stop if off
    by more than 1 logit; one bf16 ulp at these magnitudes is 0.125).
  - Head: StandardScaler + multinomial LogisticRegression in one pipeline (scikit-learn 1.9.1; L2 via
    `l1_ratio = 0`, lbfgs, `class_weight = None`, `max_iter = 20000`), so standardisation statistics come from
    fit-split training rows only. C by 5-fold `StratifiedKFold(shuffle=True, random_state=20261007)` on the 300 fit
    items over `logspace(-4, 2, 13)`; criterion: pooled out-of-fold NLL of softmax(decision values); first minimum
    (smallest C) on ties.
  - Output `results/gauge3-probe.json` in the gauge schema: fit-split `logits_mean` = out-of-fold decision values at
    the chosen C (the same folds), flagged `oof: true`; exam `logits_mean` = decision values of the head refit on all
    300 fit items; `rotations: 1`; `C`.
  - Calibration and scoring exactly as gauge ①: `calibrate.py` (T on the fit out-of-fold logits; per-class τ at
    SA 0.95 on the fit out-of-fold probabilities), `score.py`, the same metrics, bootstrap (1,000 resamples, seed
    20261007) and README rules. Scored once.
- **labels-needed v1** (head variants of gauge3-probe v1 on the same features), `gauge_probe.py labels-needed`
  then `score.py --labels-needed`.
  - n ∈ {24, 48, 96, 192, 300} fit labels, n/3 per label; draws k = 0, 1, 2, 3, 4 with seed 20261007 + k
    (n = 300: one draw, the whole fit split, k = 0). Subsample: one `numpy.random.default_rng(seed)` per draw; per
    label in label order, `choice(that label's fit rows, n/3, replace=False)`; rows kept in fit order.
  - Per variant: C by the same CV (`StratifiedKFold(shuffle=True, random_state=seed)`, 3-fold if n < 96, else
    5-fold; same grid and criterion); T and per-class τ fitted by `calibrate.py`'s own code on the variant's
    out-of-fold predictions within its subsample; head refit on the n rows; exam decision values. Each of the 21
    variants is scored once on the 700-item exam. (The n = 300 variant repeats gauge3-probe v1's fit-side procedure.)
  - Reported per n: mean and min–max over draws of exam accuracy and of exam ask rate @ SA 0.95 (at the variant's
    own τ), and a paired 95% percentile bootstrap CI of the draw-mean accuracy and ask rate (the 1,000 exam
    resamples of seed 20261007, the same index draws for every variant).
  - "Smallest n within noise of gauge ①'s or better": the smallest n whose draw-mean ask-rate CI has its lower end
    ≤ the upper end of gauge ①'s exam ask-rate CI (overlap = within noise; entirely below = better); "none" if no n
    qualifies. Outputs `results/labels-needed.json` and `results/labels-needed.png`.
- **jev jev-v1** (external, hosted), `nikasha/gauge_jev.py`; README row "Jev (external, hosted, same 300 items)".
  - Subsample: the 300 exam ids listed below (100 per label), drawn with gauge ①'s pre-registered reducer
    `gauge_logit.select_exam_subset(exam, 300)` (`random.Random(20261007)`); file `data/seta/jev-subsample.json`,
    sha256 `69e7937fe6b81f6c4f507c5445e51a87c6b198c0f5aeaf60b81b73d17d6eb187`.
  - Endpoint `POST https://openrouter.ai/api/alpha/decisions`, model pinned `typesafe/jev-1.13` (the dated id of
    each response is recorded). Request template jev-v1, sha256
    `1e5da8521bf778f6fc292044507325a05eb567116c4d1433816ab52e56192213` (in `ops/PROMPTS.md`): one `choice` question
    `tool_calls` whose instructions are prompt v1's system sentence without "Answer with a single letter." and whose
    criteria are the three plain phrases of prompt v1, each mapped to itself, in label order; `state` =
    {"tools": prompt v1's TOOLS block, "request": the item's request}. Schema read from the openrouter.ai docs before
    any call (URLs in `ops/COMMAND_LOG.md` and `results/jev.json`).
  - Mapping: `answers.tool_calls.probabilities` → probability vector in label order, normalised to sum 1 (raw sum
    kept); a missing option key counts 0; keys outside the three phrases make the response invalid. A response
    without `probabilities` → one-hot on its `choice`, `confidence: none` (the `confidence` field measures how
    concentrated the distribution is; it is stored verbatim and never used as a probability). If no response carries
    probabilities, the ask rate is "n/a (no confidence returned)". Prediction = argmax of the vector (ties → lower
    label index); disagreements with the returned `choice` are counted.
  - An exam-only subsample has no fit-split outputs, so nothing is fitted for Jev: T = 1 (fixed, as for the constant
    baselines) and no thresholds — its ask rate and selective accuracy at fitted thresholds are
    "n/a (no fit-split outputs)". Accuracy, macro-F1, NLL, ECE, AUROC and the global-τ curve are reported on its
    successful items, with the same bootstrap.
  - Calls: sequential; up to 3 retries per item (backoff 2, 4, 8 s) on timeouts, connection errors, HTTP 408, 409,
    425, 429, 5xx, 524, 529 or an unusable 200 body; HTTP 400 and 413 fail the item; HTTP 401, 402, 403, 404 abort
    the run. Spend cap: no call after the 400th. Every raw response body is stored verbatim in
    `results/jev-raw/<id>.json`. Items still failing are listed as `failed` and counted in the README, never dropped
    silently. The key is read from `~/.openrouter-env` inside the calling command and never printed, logged or committed.
  - Like-for-like group: on the subsample ids where Jev succeeded (exam.jsonl order), gauges ① and ③ are scored with
    their fit-split T and τ, and Jev as above; the same bootstrap seed and item order make the three rows' resamples
    paired. In this group only, all three rows also report the unfitted gate "answer iff max p ≥ 0.95" (each gauge's
    own final probabilities; no threshold fitted): ask rate and selective accuracy with CIs, labelled unfitted.

### Share of the gap (the analogue of the per-category threshold rung — never "replication")
On the 700-item exam, with ask rate @ SA 0.95 as the metric:
`share = (ask_①(global τ) − ask_①(per-class τ)) / (ask_①(global τ) − ask_③(per-class τ))`,
where ask_①(global τ) answers iff max p_cal ≥ gauge ①'s fit-chosen global τ and the per-class terms use each gauge's
fit-chosen per-class τ. CI: the same 1,000 resamples (seed 20261007, exam.jsonl order), paired — each resample
recomputes all three ask rates on the same items; percentile CI over the resamples with a nonzero denominator. If the
denominator's 95% CI includes 0 or lies entirely below 0, the README prints "undefined: the probe does not beat the
zero-shot read" instead of a number. ask_①(per-class τ) recomputed here must equal the Gate 3 exam block's value.

### Exam reads under this amendment
gauge3-probe v1 once; each of the 21 labels-needed variants once; Jev once (one call per item, retries aside); the
like-for-like group once; the share of the gap once. README rows: uniform, majority-of-fit, gauge ① logit read,
gauge ③ probe, Jev (external, hosted, same 300 items); then the 300-item like-for-like group; the labels-needed
figure; the share-of-gap line. **Every column that is run is published; no column is dropped after scoring.**

### Jev subsample ids (300, exam.jsonl order)
```
irrelevance_163, live_irrelevance_142-13-8, live_irrelevance_503-148-0, live_irrelevance_17-2-5, irrelevance_161, live_irrelevance_13-2-1, live_irrelevance_712-236-0, live_irrelevance_416-95-0, live_irrelevance_493-142-4, live_irrelevance_686-223-1, live_irrelevance_605-193-7, live_irrelevance_823-316-0, live_irrelevance_469-130-1, live_irrelevance_64-2-52, live_irrelevance_563-172-4, irrelevance_116, live_irrelevance_57-2-45, live_irrelevance_258-56-1, live_irrelevance_661-211-1, live_irrelevance_711-235-0, irrelevance_26, live_irrelevance_32-2-20, live_irrelevance_338-80-1, irrelevance_187, live_irrelevance_726-242-0, live_irrelevance_622-196-3, live_irrelevance_560-172-1, live_irrelevance_362-81-23, irrelevance_10, live_irrelevance_806-305-10, live_irrelevance_732-247-0, live_irrelevance_410-91-1, live_irrelevance_478-135-0, live_irrelevance_516-154-1, live_irrelevance_94-2-82, live_irrelevance_485-139-0, live_irrelevance_499-145-0, live_irrelevance_253-53-1, live_irrelevance_368-81-29, live_irrelevance_357-81-18, live_irrelevance_1-0-1, live_irrelevance_169-23-1, live_irrelevance_477-134-1, irrelevance_73, live_irrelevance_748-259-0, live_irrelevance_60-2-48, irrelevance_90, live_irrelevance_762-272-1, irrelevance_86, irrelevance_155, live_irrelevance_248-49-0, live_irrelevance_59-2-47, live_irrelevance_407-89-0, live_irrelevance_20-2-8, irrelevance_168, live_irrelevance_302-76-0, live_irrelevance_610-194-3, live_irrelevance_680-220-0, irrelevance_221, live_irrelevance_404-86-0, irrelevance_175, irrelevance_12, live_irrelevance_472-132-0, live_irrelevance_573-179-2, live_irrelevance_5-0-5, live_irrelevance_191-32-4, live_irrelevance_632-201-0, irrelevance_210, live_irrelevance_220-34-9, live_irrelevance_133-12-0, live_irrelevance_48-2-36, live_irrelevance_607-194-0, live_irrelevance_98-2-86, live_irrelevance_657-209-0, live_irrelevance_86-2-74, live_irrelevance_229-36-0, live_irrelevance_746-257-0, live_irrelevance_214-34-3, live_irrelevance_356-81-17, live_irrelevance_490-142-1, irrelevance_48, live_irrelevance_58-2-46, irrelevance_130, live_irrelevance_597-192-0, irrelevance_71, irrelevance_142, live_irrelevance_256-55-0, live_irrelevance_683-221-1, live_irrelevance_238-43-0, live_irrelevance_821-314-5, irrelevance_4, live_irrelevance_286-68-0, live_irrelevance_873-358-0, irrelevance_29, live_irrelevance_353-81-14, live_irrelevance_616-195-4, live_irrelevance_391-81-52, irrelevance_82, live_irrelevance_74-2-62, live_irrelevance_177-29-0, simple_174, live_multiple_1018-247-0, multiple_126, simple_240, live_multiple_246-110-0, simple_304, live_multiple_34-11-0, live_multiple_497-148-7, simple_211, simple_112, live_multiple_288-129-3, multiple_98, live_simple_68-32-0, multiple_13, live_multiple_418-141-7, live_multiple_685-164-1, live_multiple_94-41-1, simple_201, live_multiple_254-118-0, simple_104, live_multiple_747-169-2, simple_301, live_multiple_562-155-2, live_multiple_463-145-14, live_multiple_810-176-1, live_multiple_143-55-1, live_multiple_528-151-4, live_multiple_74-34-0, multiple_112, live_simple_100-59-1, live_multiple_412-141-1, simple_364, live_multiple_800-175-6, live_simple_115-71-0, multiple_26, live_multiple_437-142-3, live_multiple_1007-236-0, live_multiple_658-162-0, simple_300, live_multiple_501-148-11, live_multiple_491-148-1, live_multiple_182-77-0, simple_213, live_simple_176-102-0, live_simple_59-28-0, live_simple_144-95-1, live_multiple_595-158-1, live_multiple_238-106-4, simple_4, live_multiple_526-151-2, simple_62, multiple_7, live_simple_189-114-0, live_multiple_934-191-22, live_multiple_393-138-1, live_multiple_162-63-1, simple_257, live_multiple_135-51-1, simple_200, live_multiple_229-103-0, simple_292, live_simple_1-1-0, multiple_174, simple_32, live_simple_101-60-0, live_multiple_510-149-7, live_multiple_173-71-2, simple_244, live_multiple_56-22-3, live_multiple_134-51-0, live_multiple_667-162-9, simple_145, live_multiple_429-141-18, multiple_149, live_multiple_89-39-0, simple_212, live_multiple_1036-263-1, live_multiple_987-218-0, simple_128, simple_23, simple_268, live_multiple_79-36-0, live_multiple_732-167-3, live_multiple_130-50-2, live_multiple_950-199-1, simple_345, simple_158, live_multiple_15-4-7, live_simple_44-18-0, live_multiple_391-137-9, live_multiple_531-151-7, live_multiple_240-107-1, multiple_155, live_multiple_71-31-0, live_multiple_578-156-4, simple_31, multiple_189, live_multiple_432-141-21, live_multiple_87-38-4, simple_369, parallel_multiple_46, parallel_167, parallel_16, parallel_multiple_138, parallel_multiple_28, parallel_multiple_52, parallel_95, parallel_multiple_124, parallel_multiple_148, live_parallel_12-8-0, parallel_169, parallel_multiple_69, parallel_198, parallel_multiple_179, parallel_multiple_163, parallel_multiple_95, parallel_80, live_parallel_multiple_8-7-0, parallel_multiple_77, parallel_multiple_176, live_parallel_0-0-0, live_parallel_multiple_11-10-0, parallel_41, parallel_multiple_18, live_parallel_13-9-0, live_parallel_multiple_2-2-0, live_parallel_multiple_19-16-1, parallel_173, parallel_multiple_14, parallel_125, parallel_109, parallel_multiple_189, parallel_multiple_162, parallel_186, parallel_multiple_80, parallel_98, parallel_multiple_87, parallel_multiple_121, parallel_multiple_180, parallel_183, parallel_multiple_192, parallel_6, parallel_multiple_99, live_parallel_multiple_5-4-0, parallel_multiple_6, parallel_37, parallel_104, parallel_31, parallel_189, parallel_144, parallel_72, parallel_76, parallel_multiple_134, parallel_73, parallel_45, parallel_91, parallel_22, parallel_114, parallel_multiple_112, parallel_multiple_93, parallel_multiple_150, live_parallel_multiple_7-6-0, parallel_177, parallel_69, parallel_40, parallel_62, parallel_172, parallel_multiple_94, parallel_multiple_115, parallel_multiple_41, parallel_20, parallel_111, parallel_multiple_23, parallel_multiple_183, parallel_multiple_116, live_parallel_multiple_4-3-0, parallel_194, parallel_multiple_174, parallel_118, parallel_100, parallel_49, parallel_137, parallel_multiple_153, parallel_multiple_167, parallel_multiple_136, parallel_74, parallel_110, parallel_multiple_79, parallel_multiple_61, parallel_multiple_50, live_parallel_multiple_3-2-1, parallel_multiple_24, parallel_175, parallel_multiple_152, live_parallel_10-6-0, parallel_25, parallel_92, parallel_134, parallel_multiple_76, parallel_160
```
