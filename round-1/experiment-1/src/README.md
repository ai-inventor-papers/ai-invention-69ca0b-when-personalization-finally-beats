# Cold-start heuristic sweep across small e-commerce catalogs

CPU-only sweep of 16 lightweight cold-start recommendation heuristics on a
factorial grid of synthetic small-catalog families **plus every usable real
catalog found in the iteration's shared pool** (UCI Online Retail II x2
restricted, Olist Brazilian e-commerce x2 restricted, 19 synthetic families
produced by the DATASET artifact).  For each catalog, each user, each history
length `k in {0,1,2,3,5,8}` and each heuristic, the experiment records the
**rank r of the user's true next purchase** and the **top-50 ranking** — the
complete signal from which NDCG@5/10, Recall@5/10 and bootstrap CIs can be
recomputed without re-running any model.

## Files

| file | purpose |
|---|---|
| `method.py` | main entry: corpus acquisition, worker pool, orchestration, provenance |
| `catalog_pool.py` | shared-pool discovery + normalization adapter (accepts several producer schemes) |
| `synthetic.py` | latent-intent catalog generator + 192-config factorial grid + MI ceiling (MC) |
| `heuristics.py` | the 16 disclosed heuristics + prior-art landscape |
| `eval_metrics.py` | temporal leave-one-out-at-k protocol, rank collection, diagnostics, active elicitation |
| `output.py` | schema-conformant `method_out_*.json` parts, aggregates CSV, diagnostics CSV |
| `checks.py` | unit/logic checks C1–C8 (see below) |
| `out/method_out/method_out_*.json` | **the deliverable**: one part per chunk of catalogs; each part self-contained `{"metadata": ..., "datasets": [...]}` |
| `out/aggregates_by_catalog_heuristic_k.csv` | per (catalog, heuristic, k): n, mean rank, NDCG@5/10, Recall@5/10 |
| `out/catalog_diagnostics.csv` | per catalog: HHI, normalized entropy, attr entropy, mean history length, cross-fitted headroom |
| `out/provenance.jsonl` | per catalog: origin, source path, fold, n_items, n_users, wall_s, cells |

## Reproduce

```bash
uv run method.py --mode full            # 192-catalog synthetic grid + all pool catalogs
uv run method.py --mode corners         # 10 grid-corner catalogs (quick check)
uv run method.py --mode smoke           # 1 tiny catalog end-to-end
uv run method.py --checks               # unit/logic checks C1-C8
uv run method.py --mode full --max-catalogs 30 --outdir out_small
```

Deterministic: every catalog is seeded; byte-identical outputs on re-run (C7).

## Exact (catalog, heuristic, k) → method_out.json mapping

For catalog `C` (dataset id = `metadata_catalog`), user `u`, history length
`k` and heuristic `h` with sanitized key `hk = h.replace(".", "_")`:

- the example is the row with `metadata_catalog == C`, `metadata_user == u`,
  `metadata_k == k` inside the part whose `datasets[].dataset == C`;
- `metadata_rank_<hk>` = integer `r`: rank (1-based) of `metadata_true` (the
  user's actual next purchase) in the heuristic's ranking, ties broken by item
  index ascending;
- `predict_<hk>` = JSON string of the top-50 ranked item ids (truncated to
  `|V|` when the catalog has fewer than 50 items);
- `input` = `{"catalog","user","k","history"}`; `output` = `metadata_true`.

Metrics need no model re-runs:

```
NDCG@cut = 1/log2(r+1) if r <= cut else 0
Recall@cut = 1 if r <= cut else 0
```

Bootstrap/user resampling: resample `metadata_user` values within a
(catalog, k) cell and recompute means of these closed forms.

### Not-applicable cells

- `content_knn_{1,3,5}`, `last_item_nbhd`, `pop_scaled_content`,
  `co_purchase`: undefined at `k=0` → `metadata_rank_* = ""`, no `predict_*`.
- `active_elic2` (Bayesian forced-choice elicitation): defined only at `k=0`
  → `""` at `k>0`.  It is reported at k=0 only; a k>0 comparison belongs to
  the unknowns of the experiment.
- `lambda_hybrid_*` at k=0 degenerate to `(1-lam)*normalized_popularity`
  (no content signal yet) — they are reported, marked degenerate in the
  metadata's `degenerate_cells` note.

## Protocol (leak-free temporal leave-one-out-at-k)

- Screen half = positions `1..floor(0.8L)`; confirm half = the rest.
- All fitted parameters (popularity counts, category/price shares, recency
  decays, TF-IDF, co-occurrence) use **only screen-half events**, and for each
  cell `(u,k)` the user's own events at positions `1..k+1` (context + ground
  truth) are excluded (L1O-pop).  Confirm events never enter any estimate
  (C8 verifies this).
- Ground truth for cell `(u,k)` = the purchase at position `k+1`.
- For `k>=1` the user's own context items are filtered from every ranking
  (uniform across heuristics).
- Tuning discipline: every heuristic parameter is fixed & disclosed a priori,
  or chosen only on screen-half users via the first-j-predict-j+1 inner split
  (headroom diagnostic).

## Heuristics (16, all disclosed a priori)

Popularity family: `pop_global`, `recency_pop_{0.1,0.5,1.0}` (half-lives as
quantiles of the screen span), `pop_category`, `pop_price` (banded
popularity, gamma=1).  Content-kNN family: `content_knn_{1,3,5}`,
`last_item_nbhd`, `pop_scaled_content` (beta=0.5).  Association:
`co_purchase` (max-lift).  Hybrids: `lambda_hybrid_{0.25,0.5,0.75}`
(B2P-style).  Elicitation: `active_elic2` (2 forced-choice questions, greedy
expected-entropy-reduction over the category posterior; pairs enumerated
exactly for |V|<=250, else a fixed 20k-pair subset).

Prior art: Ji et al. SIGIR 2020 (popularity baseline), Chaimalas et al.
RecSys 2023 (B2P hybrids), content-based textbook remedies, Agrawal &
Srikant VLDB 1994 (association rules), Golovin & Krause ICML 2010 (adaptive
submodularity — no canonical cold-start elicitation paper: the gap this sweep
characterizes).  Full citations live in `metadata.prior_art`.

## Diagnostics per catalog (metadata_* on every example)

`metadata_sales_hhi`, `metadata_sales_norm_entropy` (screen-half shares),
`metadata_attr_entropy` (category/price/tag distribution entropy),
`metadata_mean_hist`, `metadata_headroom` (cross-fitted NDCG@5 gain of
content_knn_3 over the best popularity-family heuristic on the screen inner
split), `metadata_mi_ceiling` (MC estimate of I(history; next) per k,
**inline synthetic catalogs only**, in `metadata_mi_ceiling` per example and
`diagnostics.mi_ceiling` per catalog).

## Corpus provenance

- `source_name == "pool"` catalogs come from
  `gen_art_dataset_1/processed/*.json` (23 usable: 4 real restricted
  catalogs + 19 synthetic families); pool files that cannot be parsed are
  listed in `metadata.pool_files_skipped` with reasons (e.g. the two full
  UCI versions whose histories reference items outside their 1000-item
  universe — the `_lights` restricted variants are the usable forms).
- `source_name == "inline_fallback"` catalogs are the experiment's own
  192-catalog factorial grid (|V| in {20,100,500,1000} x LOW/HIGH attribute
  entropy x zipf 0.5/1.5 x stickiness 0.2/0.6 x mean history 4/10 x seeds
  0..2), generated by `synthetic.py` from the latent-intent model.
- Pool catalogs with >1000 users are subsampled to a fixed RandomState(0)
  sample of 1000, recorded in their family_params.

## Output format notes

- Parts validate against the `exp_gen_sol_out` aii-json schema
  (`additionalProperties == false` at the example level).
- Heuristic names containing `.` are sanitized to `_` in
  `metadata_rank_*`/`predict_*` keys (schema restriction); the original
  names are in `metadata.heuristic_inventory` and the aggregates CSV.
- Pool catalogs whose item ids exceed 12 chars are re-encoded to short
  aliases `p0..pN` inside `input`/`output`/`predict_*`/`metadata_true`; the
  lossless alias→original mapping is in `metadata.metadata_id_mappings`.
- Cells with fewer than `metadata_n_valid` ≥ `min_examples` users are marked
  `metadata_supported: false` and must be excluded from statistical
  significance claims downstream.

## Runtime / logs

- Full run (`logs/full2.log`): **391.7 s total wall**, 215 catalogs OK,
  38 parts, 521,106 examples, ~3.5 GB.  This run's parts carry
  `metadata.total_wall_seconds ≈ 274` (compute + pool-load phase; the value
  stamped at part-flush now covers compute + serialization — the patched
  `write_parts` stamps the cumulative wall at flush, so re-runs report the
  full wall in every part).
- `summary/` at the workspace root holds the compact artifacts:
  `aggregates_by_catalog_heuristic_k.csv`, `catalog_diagnostics.csv`,
  `method_out_manifest.json` (part → catalog mapping).

## Check results (checks.py, C1–C8)

C1 rank↔metric recomputation PASS · C2 k=0 popularity≈Bayes-optimal
(corr 0.9994) PASS · C3 content-kNN category separation PASS · C4 recency vs
all-time trend discrimination PASS · C5 NDCG@10≥NDCG@5 PASS · C6
crossover-direction **checkpoint (non-gating)**: in our latent-intent
synthetic, content-kNN does NOT overtake pure popularity at any k — the model
separates the category signal from within-category choice (which is
popularity-driven), so the sweep maps where personalization *can* and
*cannot* win rather than assuming it does · C7 determinism PASS · C8 no-leak
PASS.