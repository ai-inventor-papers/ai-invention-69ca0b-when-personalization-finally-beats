# Iteration-2 Evaluation: pre-registered cold-start screen + confirm round (EXECUTED)

This artifact EXECUTES the pre-registered 9-step cold-start heuristic wide-screen against
the now-published shared measurement pool, and runs the pre-registered **confirm round**
for the first time. It is a repaired/deepened fork of the iteration-1 protocol
(`gen_art_evaluation_1/eval.py`), which was left **INCONCLUSIVE** because the pool had not
yet been published. All four iteration-1 failure modes are fixed and disclosed in the
output preamble (F1-F4).

## Inputs (read-only)

- **EXPERIMENT** measurements (art_SxhGX25ryIj7):
  `/ai-inventor/aii_data/runs/run_5D4WD4vgZZMJ/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/out/`
  - `method_out/method_out_*.json` - 46 parts, 215 catalogs, 521,106 examples
  - `method_out_manifest.json` - part<->catalog map
  - `aggregates_by_catalog_heuristic_k.csv`, `catalog_diagnostics.csv`, `provenance.jsonl`
- **DATASET** pool (art_ptSx3clYW5Mo): `/.../gen_art_dataset_1/processed/*.json` (pool
  synthetic `family_params`, used for the candidate-D turnover flag and the plug-in MI).

## How to run

```bash
uv run eval.py                  # full 215-catalog screen + confirm round (~3 min on 4 CPUs)
uv run eval.py --selftest       # quick code-correctness smoke test on an inline pool
uv run eval.py --pool-dir <dir> # point at a different manifest+parts directory
```

CPU-only; zero paid API spend (no OpenRouter calls in this artifact).

## The four fixes (disclosed in eval_out.json metadata.run_preamble.fixes_applied)

- **F1** MERGE-ALL-PARTS loading: reads `method_out_manifest.json`, loads EVERY part
  under `method_out/`, merges by `catalog_id` (asserts no duplicate across parts).
- **F2** POPULATE CATALOG META: populates `sales_hhi`, `attr_entropy`, `mean_hist`,
  `headroom`, `n_items`, `n_users`, `fold` by precedence (a) part dataset metadata,
  (b) per-example `metadata_*`, (c) `catalog_diagnostics.csv`, (d) `provenance.jsonl`
  (authoritative for n_items/n_users/fold).
- **F3** `n_items <= 100` filter for the plain-vs-banded small-catalog check (reproduces
  the reviewer 98), with BOTH denominators (`<100` -> 50 and `<=100` -> 98) reported.
- **F4** `MIN_USERS_CELL = 50` guard excludes under-powered bootstrap cells from
  statistical claims.

## Headline results (see eval_out.json -> metadata / metrics_agg)

- **Survivor: B (NOISE) / popularity wins.** No candidate (A content/hybrid, C active
  elicitation, D recency-windowed) exceeds the mechanical gate: median-over-catalogs
  screen NDCG@5 gain >= +0.02 AND 95% CI excludes 0 in >= 50% of catalogs, AND not
  dominated. Median screen gains: A -0.002, C -0.071, D +0.004. Winner identity is B
  under all five bootstrap seed bases (agreement 1.0).
- **R1 PASS**: 18,275 (catalog, heuristic, k) cells recomputed from raw ranks; 0
  mismatches vs aggregates CSV (max abs diff 5e-6 = CSV storage precision).
- **Reviewer counts reproduced exactly**: plain pop_global beats pop_category at k=0 in
  **75/98** and pop_price in **73/98** small catalogs (n_items<=100); the small set is
  **98**, and iteration-1's "144" was in fact `uci_retail_1_lights`, a **144-ITEM
  catalog**, not a 144-catalog set.
- **Crossover k\*** is `inf` for 95.6% of screen catalogs (content never overtakes
  popularity in this generative family) -> the pre-registered k\*-vs-stat correlations
  are INESTIMABLE (honest-degeneracy rule) and the pre-registered continuous SURROGATE
  is reported instead: content-family best gain at k=8 **FALLS with sales HHI**
  (Spearman rho -0.74, p~0.000), rises weakly with |V| and attr-entropy.
- **Decision rule as a rule**: LOO (leave-one-family-out) R2 of headroom = **0.914**,
  driven almost entirely by sales normalized entropy (pearson 0.93); but **0 of 203**
  screen synthetic catalogs reach the +0.02 headroom build threshold (max +0.0025), so
  the +0.02 don't-build boundary is DEGENERATE: always-popularity (always don't-build)
  trivially achieves accuracy 1.0 and no non-trivial rule is learnable in range.
- **MI ceiling**: generator-MC median I(history; next) = 0.086 nats over the 192 inline
  synthetic catalogs (range 0..1.81), bounding the maximum estimable headroom.
- **Confirm round (never-touched evidence)**: no candidate sustains a majority
  CI-excluding gain on CONFIRM user-log halves / confirm-fold synthetic / real catalogs ->
  the NOISE/popularity verdict **holds out-of-sample**. Real Olist cats show A gain
  +0.007 (CI-excluding 0%, 2 supported), UCI `_lights` flagged too-few-users.

## Output files

| file | purpose |
|---|---|
| `eval_out.json` | **the deliverable**, validated `exp_eval_sol_out` (schema PASS) |
| `full_/mini_/preview_eval_out.json` | size-optimized variants for quick inspection |
| `logs/run.log`, `logs/full_run*.log` | loguru run logs |
| `pyproject.toml`, `.venv/` | uv environment |

Validation:

```bash
SKILL_DIR=/ai-inventor/.claude/skills/aii-json
$SKILL_DIR/../.ability_client_venv/bin/python \
  $SKILL_DIR/scripts/aii_json_validate_schema.py \
  --format exp_eval_sol_out \
  --file <abs path>/eval_out.json
```

## Reproducibility / determinism

Seeds fixed and disclosed: per-catalog bootstrap seed = 20240601 + sha1(catalog_id)[:8]
mod 100000; B=3000 for STEP-5/STEP-10 headline CIs, B=1500 for the noise report and seed
sweep only (recorded). All ranks loaded from the shared pool (no model re-runs); the
confirm half is never used for any screen-side choice (fit / reference / survivor / seed /
threshold). `eval.py` is deterministic given the shared pool.