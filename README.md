# nikasha

One sealed exam, many decision gauges. A gauge is any program that, for one item, returns a probability vector over the fixed label list (`no-call`, `one-call`, `multi-call`): one entry per label, every entry between zero and one inclusive, and the entries summing to one within a tolerance of one part in a million. Each gauge is its own process and writes `results/<gauge>.json`; the harness (`calibrate.py`, `score.py`, `readme.py`) reads that JSON and never imports an engine.

## Set A

set A: built from `gorilla-llm/Berkeley-Function-Calling-Leaderboard` (license Apache-2.0). Three labels, in fixed order:

- `no-call` — irrelevance files have no possible_answer ground truth; every item is labelled by category
- `one-call` — len(possible_answer ground_truth) == 1 (joined on id); items with count != 1 are dropped
- `multi-call` — len(possible_answer ground_truth) >= 2 (joined on id); items with count < 2 are dropped

| | total | no-call | one-call | multi-call |
|---|---|---|---|---|
| set A | 1000 | 334 | 333 | 333 |
| fit | 300 | 100 | 100 | 100 |
| exam | 700 | 234 | 233 | 233 |

Dropped for a category contradiction: 0 items — ground-truth call count contradicts the file's category.  
Dropped for missing ground truth: 1 item (BFCL_v3_live_multiple.json: 1) — no possible_answer row for the item's id (or ground_truth not a list/dict).  
Sampling draws at least 10 items from every non-empty source file, so live and non-live items both appear. Seed 20261007 for sampling and split.

Canary (KU3): verbatim completions 0/20, twins 0/20 — not flagged; gauge ① is reported as a zero-shot read.

## Results

| gauge | n_exam | accuracy [CI] | ECE raw → cal | ask rate @ SA 0.95 [CI] | selective accuracy achieved [CI] | AUROC [CI] |
|---|---|---|---|---|---|---|
| uniform | 700 | 33.4 [30.1, 37.3] † | 0.001 † → 0.001 † | 100.0 [100.0, 100.0] † | — | 0.500 [0.500, 0.500] † |
| majority-of-fit | 700 | 33.4 [30.1, 37.3] † | 0.666 → 0.666 | 100.0 [100.0, 100.0] † | — | 0.500 [0.500, 0.500] † |
| gauge ① logit read | 700 | 88.4 [85.9, 90.7] † | 0.106 † → 0.041 † | 13.4 [11.0, 16.0] | 95.2 [93.4, 96.9] † | 0.836 [0.789, 0.881] † |
| gauge ③ probe | 700 | 93.0 [91.1, 94.9] | 0.039 † → 0.030 † | 7.3 [5.4, 9.2] | 95.8 [94.3, 97.2] † | 0.904 [0.866, 0.936] † |
| Jev (external, hosted, same 300 items) | 300 | 86.0 [81.7, 90.0] † | 0.072 † → 0.072 † | n/a (no fit-split outputs) | n/a (no fit-split outputs) | 0.839 [0.768, 0.899] † |

† within noise: 95% CIs overlap with another row.

A constant uniform predictor is calibrated by construction (confidence 1/3, accuracy ≈ 1/3), so its ECE overlaps any well-calibrated gauge.

95% percentile bootstrap CIs, 1000 resamples, seed 20261007; ECE with 15 equal-width bins; selective-accuracy target 0.95; thresholds fitted on the fit split only.

Letter bias of gauge ① logit read (fit split, rotation r0 minus the mean over rotations, logits per label in label order): [+0.36, +0.79, −1.63]; averaging over three letter orders removes this bias.

![selective accuracy vs ask rate](results/curve.png)

Hidden-state check (KU2): last hidden state shape [1, 156, 3840], head reproduces logits within max-abs 0.0 — PASS.

## Like-for-like: the same 300 exam items

Gauges ① and ③ (their fit-split T and thresholds) and Jev on the same 300 exam items (the Jev subsample, 0 Jev failures removed), one bootstrap order, so the rows' resamples are paired. The last two columns are the unfitted gate "answer iff max p ≥ 0.95" on each gauge's own final probabilities — no threshold fitted, reported for this group only.

| gauge | n | accuracy [CI] | ECE raw → cal | ask rate @ SA 0.95 [CI] | selective accuracy achieved [CI] | AUROC [CI] | unfitted gate: ask rate [CI] | unfitted gate: selective accuracy [CI] |
|---|---|---|---|---|---|---|---|---|
| gauge ① logit read | 300 | 88.3 [84.7, 91.7] † | 0.113 † → 0.061 † | 13.3 [9.3, 17.0] † | 95.8 [93.2, 98.0] † | 0.827 [0.746, 0.893] † | 62.7 [57.0, 67.3] | 98.2 [95.5, 100.0] † |
| gauge ③ probe | 300 | 93.7 [90.7, 96.0] † | 0.041 † → 0.032 † | 8.0 [5.0, 11.3] † | 96.7 [94.6, 98.6] † | 0.889 [0.815, 0.954] † | 29.7 [24.3, 35.0] † | 98.6 [96.7, 100.0] † |
| Jev (external, hosted, same 300 items) | 300 | 86.0 [81.7, 90.0] † | 0.072 † → 0.072 † | n/a (no fit-split outputs) | n/a (no fit-split outputs) | 0.839 [0.768, 0.899] † | 38.0 [32.7, 43.7] † | 96.8 [94.1, 99.0] † |

† within noise: 95% CIs overlap with another row.

Jev: 300 of 300 subsample items answered, 0 failed (listed in `results/jev.json`); 304 calls including 4 retries (cap 400); every response carried per-option probabilities, taken as its probability vector. Nothing is fitted for an exam-only subsample: T = 1 and no thresholds, so its ask rate at fitted thresholds is n/a. Raw responses: `results/jev-raw/`.

## Labels needed (gauge ③ head)

The gauge ③ head retrained on n fit labels (equal per class), 21 variants in all; each variant's T and per-class thresholds are fitted on its own out-of-fold fit predictions, then the 700-item exam is scored once per variant. Mean over draws (min–max) and the paired 95% bootstrap CI of the draw mean (1000 exam resamples, seed 20261007; the CI does not include draw-to-draw variation, the min–max does).

| fit labels n | draws | exam accuracy | ask rate @ SA 0.95 | selective accuracy achieved | draws meeting the SA target | within noise of gauge ① or better |
|---|---|---|---|---|---|---|
| 24 | 5 | 90.6 (89.0–91.4) [88.6, 92.5] | 8.5 (0.3–21.6) [7.2, 9.7] | 92.0 (89.2–94.1) | 0/5 | yes |
| 48 | 5 | 89.8 (89.0–90.1) [87.7, 91.7] | 1.2 (0.0–3.6) [0.8, 1.6] | 90.3 (89.0–92.2) | 0/5 | yes |
| 96 | 5 | 91.0 (90.3–92.0) [89.0, 92.9] | 14.6 (1.9–55.1) [13.5, 15.7] | 92.3 (89.5–94.9) | 0/5 | yes |
| 192 | 5 | 92.1 (91.6–92.6) [90.3, 94.0] | 5.9 (0.9–11.1) [4.7, 7.2] | 94.6 (92.2–96.8) | 2/5 | yes |
| 300 | 1 | 93.0 [91.1, 94.9] | 7.3 [5.4, 9.2] | 95.8 | 1/1 | yes |

Smallest n whose mean ask rate is within noise of gauge ①'s or better (pre-registered rule; gauge ① 13.4 [11.0, 16.0]): **24**. Read it with the selective-accuracy column: at n = 24, 48, 96, 192 the draw-mean selective accuracy achieved on the exam is below the 0.95 target (gauge ①: 95.2 [93.4, 96.9]), so thresholds fitted on that few out-of-fold predictions under-ask — a lower ask rate there is not an improvement at equal selective accuracy.

![labels needed by the gauge ③ head](results/labels-needed.png)

## Engines and licenses

- gemma-4-12b-8bit-text: license: apache-2.0 — license_link: https://ai.google.dev/gemma/docs/gemma_4_license
- jev-1.13: license: proprietary — no published weights ("Jev is a proprietary model and as such there are no published weights and no paper.")

## Promised, not claimed

promised, not claimed: embeddings gauge, set B, set C from the measured week

Generated by nikasha/readme.py from results/*.json and data/seta/manifest.json — no number in this file is written by hand.
