# Cold-start heuristic wide-screen: pre-registered survivor selection (evaluation artifact)

**Task.** Apply the pre-registered wide-screen selection rule to the shared per-user
measurements produced by this iteration's parallel EXPERIMENT artifact, mechanically pick
ONE survivor among four candidate mechanisms, run the screen-half signed checks of the
main hypothesis, fit a tiny diagnostic decision rule, plug-in estimate the
history→next-item mutual-information ceiling, and emit the iteration-2 confirmation
protocol. The CONFIRM half is never touched.

## Outcome (this iteration): INCONCLUSIVE — shared pool absent at evaluation time

Per the artifact plan's STEP 0 *coverage handling*, the SCREEN is run **only on the shared
measurements** produced by the parallel EXPERIMENT (`method_out.json`). At evaluation time:

- The EXPERIMENT workspace (`../gen_art_experiment_1/`) had **empty** `out/` and `data/`
  — its per-user `(catalog, heuristic, k)` rank measurements had **not been published**.
- The DATASET workspace (`../gen_art_dataset_1/`) was still acquiring real catalogs
  (e.g. `raw/olist`, `raw/OnlineRetail*.xlsx`), with `out/` and `processed/` empty.
- No `method_out.json` / compatible pool existed anywhere on the run filesystem.

**Therefore this artifact did NOT fabricate or substitute its own measurements.** It
emits `eval_out.json` as a fully-schema-valid **skeleton** recording the
inconclusive/uncovered status, the coverage report, and the complete pre-registered
iteration-2 confirmation protocol. When the EXPERIMENT publishes its measurements, the
identical pipeline (`eval.py`, no code changes) loads them and runs the full 9-step screen.

## Files

| file | purpose |
|------|---------|
| `eval.py` | Full pre-registered evaluation (Steps 1–10). Re-runnable; discovers the pool, runs the full screen if present, else emits the skeleton. |
| `eval_out.json` | **Primary deliverable.** Skeleton (uncovered status) + iteration-2 confirmation protocol. Validates against `exp_eval_sol_out`. |
| `mini_eval_out.json`, `preview_eval_out.json` | Truncated inspection variants (aii-json). |
| `selftest/selftest_report.json` | **Code-correctness self-test only** (`eval.py --selftest`) on an inline synthetic pool. Never a screen result; do not cite as measurements. |
| `logs/run.log` | Run log. |
| `pyproject.toml` | Dependencies (numpy, scipy, scikit-learn, loguru). |

## Reproduce

```bash
uv venv .venv --python=3.12
uv pip install --python=.venv/bin/python -e .
.venv/bin/python eval.py            # skeleton (or full screen if the pool exists)
.venv/bin/python eval.py --selftest # exercise the analysis machinery on a tiny inline pool
```

## What the screen will do when the pool arrives

1. **Metrics** — per-user NDCG@5 (primary), Recall@5 / NDCG@10 (secondary) from each
   heuristic's rank `r` of the held-out next item: `NDCG@k = 1/log2(r+1)` if `r <= k`
   else 0 (single relevant item); `r = inf` = genuine miss (counted 0).
2. **Reference** — best popularity-family heuristic per catalog (global MostPop,
   category-banded POP, price-band POP; NOT recency-windowed POP) by mean NDCG@5 over
   `k in {0,1,2,3,5,8}` on the screen half.
3. **Aggregation** — per-`(catalog,heuristic,k)` means with 95% stratified-bootstrap CIs
   over users, paired at the user level; fixed disclosed seeds
   `base_seed=20240601 + sha1(catalog_id)[:8] % 100000`; `B=2000` resamples.
4. **Selection rule (mechanical)** — target `k`: A in `{1,2,3}`, C in `{0}`, D in
   `{0,1,2,3,5,8}`; representative gain = max-over-target-k gain. A candidate survives iff
   median-over-catalogs gain `>= 0.02` **and** its 95% CI excludes 0 in `>= 50%` of
   catalogs **and** it is not dominated by another survivor in `>= 50%` of catalogs.
   Multiple → largest median gain; none → NOISE (B) formally wins; ties reported.
5. **Candidate B (noise)** — within-noise fraction of pairwise median gains and
   seed-to-seed variance of winner identity under 4 alternate disclosed seed bases.
6. **Candidate A (diagnostic)** — crossover `k*`, signed Spearman checks vs `|V|`,
   attribute entropy, and sales HHI, and plain-vs-banded popularity in small catalogs.
7. **Decision rule** — depth-`<=2` tree over `{sales_hhi, attr_entropy, mean_hist}`
   (+ headroom), leave-one-family-out accuracy vs always-popularity / always-hybrid.
8. **MI ceiling** — coarse plug-in `I(history; next_item)` (positive-bias caveat).
9. **Confirmation protocol** — written, never executed; confirm half untouched.

All bootstrap seeds are fixed and disclosed; CPU-only; no paid API calls.