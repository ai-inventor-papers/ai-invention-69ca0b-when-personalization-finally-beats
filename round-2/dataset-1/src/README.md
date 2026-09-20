# Iteration-2 dataset artifact — additional small e-commerce purchase logs

**Status: no additional real catalog accepted** (honest empty result; the search
log is the corpus contribution). See `rejections.json`, `search_log.md`, and
`full_data_out.json`.

## What this artifact set out to do
Widen the real-catalog **confirm** evidence (fold=confirm, confirm-round only)
for the cold-start heuristic diagnostic by acquiring genuinely small
(`<= 1000`-item) e-commerce purchase logs with item metadata (category + price),
restricted to category subtrees / product-category subsets, standardized to the
iteration-1 common schema (`catalog_schema.json`, copied from the shared pool at
`3_invention_loop/iter_1/gen_art/gen_art_dataset_1/catalog_schema.json`).

## Outcome
None of the freely-downloadable candidates met the pre-registered acceptance bar
(n_items 150-900; nominal category + price-band per item; timestamped per-user
purchase logs; `>= ~100` users with `>= 4` purchases; redistribution-friendly
license; `< 300MB`). The bar was **not** lowered.

Concise reasons:
- **Retailrocket** — full raw files were acquired (Git-LFS mirror
  `sabin74/Retailrocket-Recommender-System`, sha256-verified), but even the best
  accessible restriction gives only **96 users with `>= 4` purchases** (83 with
  `>= 5`) across a 900-item catalog, below the `~100` floor; the source has
  **no price** field. (Canonical ~115k-transaction events are Kaggle-gated.)
- **Dunnhumby** — gated (Kaggle login / dunnhumby gated form); only tiny ungated
  samples exist; no raw mirror with committed data found.
- **Boutique/marketplace sweep** (HF + web, ~50 terms, 30+ families previewed) —
  zero candidates combined per-user purchased logs + item category + price in a
  freely-downloadable, properly-licensed, <=1000-item form (coffee-shop = no user
  id; Northwind = not a structured per-user log; rest = OCR/text/video/catalog).

## Deliverables
| File | What it is |
|---|---|
| `catalog_schema.json` | iteration-1 common schema (the contract; unchanged) |
| `search_log.md` | exhaustive landscape search log (sources, attempts, counts) |
| `rejections.json` | machine-readable rejection records (source/attempt/criterion/counts) |
| `explore_rr.py` | reproducible Retailrocket analysis (event types, category tree, per-subtree counts) |
| `build_catalogs.py` | restriction + acceptance check (emits catalogs only if they pass; none did) |
| `raw/fetch_rr_lfs.py` | reproducible Git-LFS download of the Retailrocket raw files |
| `data.py` | emits `full_data_out.json` (64 rejection-record examples, `n_accepted: 0`) |
| `full_data_out.json` (+`mini_`/`preview_`) | exp_sel_data_out-validated aggregate, `n_accepted: 0` |

## Where the (empty) catalog pool is published
This artifact's workspace IS the shared pool location the iteration-2 evaluation
discovers. There are **no new catalogs** in `processed/` (0 files) and
`full_data_out.json` records `n_accepted: 0`. Any confirm-round discovery therefore
finds no additional real catalogs, exactly as the evaluation is instructed to
handle ("dynamically re-discover the pool; never wait for them"). No catalogs
were fabricated or accepted on a lowered bar.

## Two-level split (reviewer MAJOR-4)
No catalogs were produced, so there is no fold assignment to document for new
data; the convention would have been `fold=confirm` for every real catalog plus a
within-catalog leave-last-out user-log split (`partition_meta.split_levels` in the
schema), as in iteration-1. Existing iteration-1 confirm catalogs are unchanged.

## Reproducibility / MAJOR-1 discipline
Every count in `search_log.md`/`rejections.json` traces to a named raw file and a
row filter (e.g., Retailrocket: `events.csv` `event=='transaction'` filter;
`item_properties_part1/2.csv` `property=='categoryid'`), with raw-file SHA-256 in
`rejections.json`. Scripts are re-runnable with `uv run` / `.venv/bin/python`.