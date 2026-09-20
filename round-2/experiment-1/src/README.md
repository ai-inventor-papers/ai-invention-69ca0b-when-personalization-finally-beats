# Iteration-2 ablation: does item-level appeal let personalization win?

**Question.** Iteration-1 established that in small e-commerce catalogs (|V| <= 1000) the
lightweight 16-heuristic cold-start family **cannot overtake popularity** within the
k <= 8 cold-start window, even when users have strong category-level latent intent.
Iteration-2 asks whether that null survives when **items themselves carry genuine,
recoverable per-item latent appeal inside a category** — i.e. when "personalization"
has something real to work with, controlled by an informativeness parameter `phi`.

**Answer (headline).** No — within the tested regime, the null *hardens*. Even at
`phi = 1` (users have strong per-item preferences inside categories; item features
carry the exact latent signal; tag encoding verified at agreement = 1.0):

* content-kNN / lambda-hybrid **never statistically overtake** popularity: 0 of 108
  cells at any inner k in {1,2,3,5,8} (`ablation_headroom_by_phi.csv`,
  row filter `overtake_any == True`), and the min phi at which a crossover enters
  k <= 8 is **none** in all 12 (size x entropy x zipf) regions
  (`null_vs_extended.csv`, column `min_phi_eff_overtake`).
* Mean headroom *narrows* with phi (phi=0: **-0.1673** -> phi=1: **-0.1519**,
  `headline_summary.json`) and the movement is **7/24 CI-dominant** at phi=1 — but
  the decomposition shows it is ~90% *popularity family degrading* (best-pop drops
  -0.0136 NDCG@5), not content-kNN improving (+0.0017); content-kNN stays pinned at
  ~0.03 NDCG@5 at k=8 while pop_global falls from 0.137 to 0.122
  (`bootstrap_headroom_CIs.csv`).
* The k=5 label "winner" (best mean NDCG@5) shifts in a few regions at phi=1
  (`lambda_hybrid_0.25` wins 5/24 cells at k=5 vs 2/24 at phi=0), but no win is
  above bootstrap noise against pop_global at any k.
* Diagnostic robustness: the paper's "+0.02 don't-build" boundary never fires at
  any phi <= 1 (max headroom_full across all 108 cells is -0.0714, at
  `phi_V500_EHIGH_z1.5_p0.5_s2`; the phi=1 max is -0.0743), so the don't-build
  guidance never fails within the tested strength range
  (`diagnostic_robustness.csv`).

Interpretation: the headroom ceiling is a **cold-start regime effect, not a feature
leak / feature-strength artifact**. Even with perfect latent tags (C10:
latent-tag agreement = 1.0, i.e. the tag-leak diagnostic probe *is* the deployed
content-kNN input), k <= 8 context cannot monetize within-category item appeal.

---

## 1. Data / porting discipline (maximal comparability with iteration-1)

| module | status |
|---|---|
| `heuristics.py` (all 16) | byte-identical to iteration-1 |
| `eval_metrics.py` | byte-identical to iteration-1 |
| `checks.py` (C1-C8) | byte-identical to iteration-1 |
| `synthetic.py` | byte-identical to iteration-1 |
| `catalog_pool.py` | ported; only pool-roots / comments adapted |
| `output.py` | ported; `PART_SIZE_MB` 85 -> 40 |

Reused verbatim: NDCG@5/10 & Recall@5/10 closed forms (`NDCG_cut = 1/log2(r+1)`
if `r <= cut` else 0), `K_VALUES = {0,1,2,3,5,8}`, `K_CUT = 50`,
`MIN_EXAMPLES = 50`, temporal user-level 80/20 screen/confirm split
(screen = positions 1..floor(0.8L)), leave-one-out popularity counts
(L1O-pop: the cell user's own events at positions 1..k+1 are excluded from every
fitted count), TF-IDF item features (one-hot category + price band + tag TF-IDF
built on screen events).

**Gate that proves the port is faithful** (T4, run before any phi>0 evidence is
interpreted — `out/null_replication_comparison.csv`,
`out/null_replication_gate.json`):
* all 19 iteration-1 pool synthetic families regenerated at phi=0 are
  **sha256 byte-identical** to the published pools
  (`out/null_replication_bytecheck.csv`, `byte_identical == True` for 19/19);
* per-(catalog, heuristic, k) NDCG@5 recomputed from the ported eval equals the
  published iteration-1 aggregates to the published file's print precision:
  1955/1955 cells satisfy `published == round(mine, 5)` and
  `pass_print5` (max |delta| = 4.996e-6 = the 5-dp quantization floor);
* headroom matches the published `catalog_diagnostics.csv` to **~1e-17**
  (23/23 catalogs, |delta| <= 1e-3 gate;
  `null_replication_comparison.csv` rows with `heuristic == "HEADROOM"`).
  (The plan's "NDCG@5 within 1e-6" strict tolerance is reported as `pass_1e6`
  (573/1955); it is unattainable against a 5-dp published file by construction —
  the print floor is 5e-6 — so the effective gate is `pass_print5`, documented in
  the `basis` field of `null_replication_gate.json`.)

Additionally check C8 (no-leak) re-verified with the new generator: screen counts
and k=0 pop_global ranks are unchanged after scrambling confirm items (confirm
events never enter fits).

## 2. Generator extension (`generator_phi.py`)

Ported generator machinery (`make_items`, `make_turnover`, `w_at`,
`sample_histories_legacy`, `build_family_legacy`, `family_grid`) is a verbatim
port of iteration-1 `generator.py`; the phi=0 path is the legacy sampler, so
identical (config, seed) produce byte-identical logs.

**Exact phi-interpolation formula** (per user u, item v in category c, time t):

    s_{u,v}    = z_v . p_{u,c}  -  mean_{v' in c}( z_{v'} . p_{u,c} )     (within-category mean-zero centering)
    P_{u,v}(t) = w_v(t) * exp(beta * theta_u[cat_v]) * exp(phi * s_{u,v}) normalized over items

with `w_v(t) = base_w_v * turnover(t)` (dynamic drift/spikes as iteration-1;
`beta = 2.0`, `dynamic = True`).  At phi=0 this is exactly the iteration-1 null
`P(v) prop exp(beta*theta_u[cat_v]) * w_v(t)`.

**Latent structure** (drawn ONCE per (config, seed), shared across all phi of the
cell — indexed by `latent_sample_seed = seed*104729 + 3011`):
* per-category latent dimension `D_c = 4`; item factor `z_v ~ N(0,I)` in R^4,
  fixed across phi; per-user per-category preference `p_{u,c} ~ N(0,I)`;
* **unit-normed factors** (`z_v <- z_v/||z_v||`): measured design decision — raw
  factors give items with large norm capture within-category mass proportional to
  factor magnitude (not direction), which broke the marginal control (gate failed
  9/9 sampled cells at phi=1; `meas_gate.py`).  Sign(z_v) unchanged, so the tag
  encoding is identical;
* **tag encoding** (the feature interface for content heuristics):
  tag `lz_d` (d in 0..D_c-1) present on item v **iff** `z_v[d] > 0`, plus
  `N_DISTRACTOR_TAGS = 2` tags `lx_0`,`lx_1` with fixed random presence (prob 0.3)
  NOT tied to z_v (C10: latent-tag agreement = 1.0000; distractor max
  |corr with z_v| = 0.149).  z_v itself never appears in `family_params` or any
  fitted parameter.

**Marginal control (popularity prior held across phi).**
The mean-zero centering makes each user's within-category modulation
marginal-preserving to second order; the unit-normed factors make every item's
preference signal identically distributed.  Per (config, seed) cell and phi level,
a **marginals-match quality gate** compares the phi catalog's screen-half per-item
empirical shares to the phi=0 catalog's (same items by construction):

| metric | tolerance | measured max (108 cells) |
|---|---|---|
| KS (max |CDF diff|) | <= 0.05 | 0.0396 |
| max absolute per-item share deviation | <= 0.02 | 0.0188 |
| |delta HHI| | <= 0.02 | 0.01744 |
| |delta normalized demand entropy| | <= 0.03 | 0.0267 |

Gate ladder (fallback plan 1), measured + disclosed:
1. sample at the nominal phi;
2. T1 tighten: per-user per-category phi=0 mass anchoring;
3. **effective-strength capping**: binary-search the largest `phi_eff <= phi`
   that passes, resample at `phi_eff`, record `phi_effective` / `phi_capped`
   in `family_params` and `out/marginals_match_gate.csv`.

Result: **108/108 cells PASS** (`marginals_match_gate.csv`, `gate_pass == True`);
21 cells are capped (`phi_capped == True`), mean phi_effective at nominal 1.0 is
0.749 (`headline_summary.json`).  All downstream analysis uses `phi_effective` for
diagnostic-robustness grouping and reports `phi_nominal` for design cells.

## 3. Grid (`ablation_grid.py`)

**Main factorial (96 cells):** |V| in {20,100,500} x entropy {LOW=2 attrs,
HIGH=30 attrs} x Zipf s {0.5, 1.5} x phi {0, 0.25, 0.5, 1.0} x seed {0,1}.
**Seed-2 (12 cells):** phi=0.5 at every (|V|, entropy, zipf) corner.
Total **108 main-grid catalogs** + **19 null-replication** + **4 real pool
catalogs** (olist x2, UCI-lights x2, copied verbatim from the iteration-1 pool) =
131 evaluated catalogs.

Hold-across-phi controls (not grid axes): `mean_hist = 10`, `alpha = 0.3`
(sticky), `beta = 2.0`, `n_users` in {600, 1000, 1500} by |V| (every
(catalog, k) cell stays well above MIN_EXAMPLES=50), dynamic turnover,
horizon 365 days.

Cell id scheme: `phi_V{size}_E{entropy}_z{zipf}_p{phi}_s{seed}`
(`out/grid_manifest.json`: 108 main + 19 null, 0 gate failures).

## 4. Evaluation

Port of iteration-1 `eval_metrics.evaluate_catalog` run unchanged per catalog:
CatRuntime screen-fitted parameters (g_counts, category/price shares, recency
lambdas, co-occurrence, IDF), per k in {0,1,2,3,5,8} with context = positions
1..k (k+1 <= L), L1O-pop counts, then rank r + top-50 for each of the 16
heuristics (`content_knn_{1,3,5}`, `last_item_nbhd`, `pop_scaled_content`,
`co_purchase` NA at k=0; `active_elic2` only at k=0; `lambda_hybrid_{0.25,0.5,
0.75}` degenerate at k=0 -> (1-lam)*norm_pop).  Outputs per-user rank +
top-50 under the iteration-1 `exp_gen_sol_out` schema (stat heuristics keys in
`metadata_rank_*` / `predict_*` with '.' sanitized to '_'; e.g.
`metadata_rank_lambda_hybrid_0_25`).

Emissions: `out/method_out/method_out_NNNN.json` (103 parts, all <= 61 MB,
all validated against `exp_gen_sol_out` with aii-json — 103/103 PASS),
`out/aggregates_by_catalog_heuristic_k.csv` (12,576 rows: 131 catalogs x
heuristics x ks), `out/catalog_diagnostics.csv` (131 rows),
`out/provenance.jsonl` (131 per-catalog lines + generate/analyze summary lines).

## 5. Post-processing / headline deliverables (`analyze_ablation.py`)

Headline metric (mirrors iteration-1):

    headroom(cell) = mean over k in {1,2,3,5,8} of
        [ NDCG@5(content_knn_3) - NDCG@5(best popularity-family heuristic) ]

with best-pop chosen on **screen-inner** cells (first-j-predict-j+1 discipline;
confirm half never selects anything): `headroom` is evaluated on inner cells only;
`headroom_full` on all valid cells (primary ablation metric).  Winner per (cell,k)
= best mean NDCG@5 full cell.  Overtake ("does a crossover enter k<=8") = strict
bootstrap-CI dominance of `content_knn_3` or any `lambda_hybrid_*` over
`pop_global`: `heuristic_CI_lo > pop_global_CI_hi`.

**Bootstrap CIs:** user-level stratified bootstrap, n_boot = 1000,
`rng = default_rng(7 + seed*104729 + k)` within (catalog, k); vectorized block
resampling (Bb = 100).  Reported: 95% CI (2.5/97.5 percentiles) on per-(cell) mean
NDCG@5 per (k, heuristic) (`bootstrap_headroom_CIs.csv`), on headroom /
headroom_full and on `delta headroom(phi - phi=0)` per cell
(`ablation_headroom_by_phi.csv`, columns `headroom_lo/hi`, `delta_lo/hi`);
dominance flagged only when the CI excludes 0 (`delta_dominant`).

## 6. Results — every headline count and its source

All counts are reproduced by the queries in `results_headline_checks.py`-free
Provenance section below (plain pandas filters over the named CSVs).

### 6.1 Null-replication gate (faithfulness before phi>0 evidence)
| check | count | source / filter |
|---|---|---|
| pool families byte-identical to iteration-1 | **19/19** | `null_replication_bytecheck.csv`, `byte_identical == True` |
| NDCG@5 cells equal published to 5-dp print precision | **1955/1955** | `null_replication_comparison.csv`, `pass_print5 == True` |
| headroom within 1e-3 of published | **23/23** | `null_replication_comparison.csv`, `heuristic == "HEADROOM"`, `pass_1e6 == True` |
| overall gate | **True** | `null_replication_gate.json`, `gate` |

### 6.2 Marginals-match gate (popularity prior held)
| check | count | source / filter |
|---|---|---|
| main-grid cells passing the gate | **108/108** | `marginals_match_gate.csv`, `gate_pass == True` |
| capped cells (phi_eff < phi) | **21** | `marginals_match_gate.csv`, `phi_capped == True` |
| mean phi_eff at nominal phi = 1.0 | **0.749** | `headline_summary.json`, `mean_phi_effective_by_nominal["1.0"]` |

### 6.3 Headroom per phi (the core ablation)
| phi (nominal) | mean headroom_full | mean headroom (inner) | mean phi_eff |
|---|---|---|---|
| 0.0 | **-0.16733** | -0.16625 | 0.000 |
| 0.25 | **-0.16492** | -0.16382 | 0.250 |
| 0.5 | **-0.16557** | -0.16458 | 0.487 |
| 1.0 | **-0.15192** | -0.15117 | 0.749 |

Source: `headline_summary.json` (`headroom_full_mean_by_nominal_phi` /
`headroom_inner_mean_by_nominal_phi` / `mean_phi_effective_by_nominal`), derived
from `ablation_headroom_by_phi.csv` (108 rows, grouped by `phi_nominal`).

Region-level means: `null_vs_extended.csv` (12 region rows, `headroom_full_phi*`
columns).  The strongest phi=1 cells are the less-concentrated Zipf s=1.5 corner:
e.g. `phi_V500_EHIGH_z1.5_p1_s0` headroom_full **-0.0743** CI [-0.0817, -0.0671]
(`ablation_headroom_by_phi.csv` row with that `catalog_id`).

### 6.4 Overtake / crossover analysis (does the winner flip inside k <= 8?)
| count | value | source / filter |
|---|---|---|
| cells where content/hybrid CI-dominates pop_global at any inner k | **0 / 108** | `ablation_headroom_by_phi.csv`, `overtake_any == True` |
| regions (size x entropy x zipf) with any phi>0 overtake | **0 / 12** | `null_vs_extended.csv`, `content_or_hybrid_overtakes_at_phi_gt0 == True` |
| min phi_eff at which a crossover first enters k<=8 | **none** (all regions) | `null_vs_extended.csv`, `min_phi_eff_overtake` |
| k=5 winner phi=0 -> phi=1 (mode over cells) | pop_global (17/24) -> pop_global (14/24); lambda_hybrid_0.25 2/24 -> 5/24 | `ablation_headroom_by_phi.csv` `winner_per_k`, `headline_summary.json`; counts re-derived in `results_headline_checks.py` |
| k=8 winner phi=0 -> phi=1 | pop_global (10/24) -> pop_global (8/24), recency_pop_0.1 8/24 at both | same |

### 6.5 Bootstrap CIs on the phi-movement (category level)
| count | value | source / filter |
|---|---|---|
| phi=1 cells with CI-dominant positive delta vs their phi=0 twin | **7 / 24** | `ablation_headroom_by_phi.csv`, `phi_nominal == 1.0 and delta_dominant == True` |
| regions containing the 7 dominant cells | 100/500 x {LOW,HIGH} x zipf=1.5 (4 regions; n=2/2/2/1) | `null_vs_extended.csv`, `n_cells_delta_dominant` |
| mean |delta headroom_full| (phi1-phi0, paired 24 cells) | **+0.01535** | derive: `ablation_headroom_by_phi.csv` delta columns; archived below |
| decomposition: d content_knn_3 (paired) | +0.00173 (71% cells >0) | `bootstrap_headroom_CIs.csv` (ndcg5 means per cell/k), twin-paired by (n_items, entropy, zipf_s, seed) |
| decomposition: d best popularity-family | **-0.01362** (88% cells <0) | same pair |
| at k=8 (N=24 per phi): content_knn_3 mean NDCG@5 phi0->phi1 | 0.0314 -> 0.0293 | `bootstrap_headroom_CIs.csv`, `heuristic == "content_knn_3" and k == 8` |
| at k=8: pop_global mean NDCG@5 phi0->phi1 | 0.1374 -> 0.1223 | same filter, `heuristic == "pop_global"` |

The decomposition is computed by `results_headline_checks.py` (archived below);
the CSV values it consumes are `bootstrap_headroom_CIs.csv` + 
`ablation_headroom_by_phi.csv`.

### 6.6 Null-vs-extended table (reviewer MAJOR-3)
`out/null_vs_extended.csv`, 12 rows = (|V| x entropy x zipf).  Columns:
`headroom_full_phi{0,0.25,0.5,1.0}` (means), `winner_k5_phi0`/`winner_k5_phi1`,
`content_or_hybrid_overtakes_at_phi_gt0`,
`min_phi_eff_overtake`, `delta_hf_phi1_minus_phi0`, `n_cells_delta_dominant`.
Every region rows as above: headroom rises with phi (mean delta global
+0.0154) but never approaches 0; no region develops a content/hybrid overtake.
The published null "content-kNN never overtakes popularity" **holds at all
phi levels**.

### 6.7 Diagnostic-robustness check (does the don't-build rule survive phi?)
`out/diagnostic_robustness.csv` (22 phi_effective levels):
* labels: `label_build = headroom_full > +0.02` (the paper's don't-build
  boundary).  **0 of 108 cells** ever cross +0.02 at any phi <= 1
  (max headroom_full over all cells = -0.0714, `phi_V500_EHIGH_z1.5_p0.5_s2`;
  the phi=1 max is -0.0743), so the boundary never fires and the don't-build
  guidance never fails (`first_failure_phi` = empty everywhere).
* HHI-threshold rule (threshold = accuracy-optimal on phi=0, `_best_hhi_threshold`,
  tau = fitted value) accuracy per phi_eff: 0.958 (phi=0) ... 1.000 (phi>0);
  always-popularity baseline accuracy: 1.000 (trivially, since no build label
  exists); size-only rule: 0.000-0.333.  LR classifier skipped: the phi=0 label
  set is single-class (all don't-build), reported honestly as empty `acc_lr`.
* Caveat reported in the README-facing summary: within phi <= 1 the
  classification test is degenerate (no build-worthy catalog appears); the
  boundary is monotone-robust but its resolution is untested above phi=1.

## 7. Reproducing

```bash
uv run method.py --checks            # C1-C12 (ported C1-C8 + new C9-C12)
uv run method.py --generate          # 108 main-grid catalogs + 19 null-rep + gate CSVs  (~17 min)
uv run method.py --evaluate --mode full   # eval all 131 catalogs -> method_out parts, payloads (~8 min)
uv run analyze_ablation.py           # headroom/bootstrap/null-vs-extended/diag-robustness (~1 min)
```

Environment: `.venv` (uv, Python 3.12); hardware measured via cgroup
(4 CPUs, 32 GB); `set_limits()` pins BLAS threads to 1 and caps RSS;
`ProcessPoolExecutor(mp_context=spawn)` with min(CPUs, 4) workers for the eval.

## 8. File inventory

| path | contents |
|---|---|
| `generator_phi.py` | extended generator (phi formula, latent structure, tags, gate ladder, legacy port) |
| `ablation_grid.py` | grid spec + cell-id parse |
| `method.py` | orchestration (generate/evaluate/write-only) |
| `analyze_ablation.py` | post-processing (metrics, bootstrap, tables, diagnostics) |
| `checks.py`, `checks_phi.py`, `meas_gate.py` | C1-C8 port + C9-C12 + factor-norm measurement |
| `catalog_pool.py`, `eval_metrics.py`, `heuristics.py`, `synthetic.py`, `output.py` | ported eval stack |
| `data/main_grid/*.json` (108) | phi-aware catalogs (pool schema) |
| `data/null_replication/*.json` (23) | 19 byte-identical null families + 4 real catalogs |
| `out/method_out/method_out_NNNN.json` (103) | per-user rank + top-50 (exp_gen_sol_out) |
| `out/aggregates_by_catalog_heuristic_k.csv` | per-(catalog, heuristic, k) NDCG/Recall |
| `out/catalog_diagnostics.csv` | sales_hhi, norm_entropy, attr_entropy, headroom per catalog |
| `out/ablation_headroom_by_phi.csv` | **headline** per-cell headroom + CIs + delta + overtake + gate metrics |
| `out/bootstrap_headroom_CIs.csv` | per-(cell, k, heuristic) NDCG@5 mean + 95% CI |
| `out/null_vs_extended.csv` | region-level MAJOR-3 table |
| `out/null_replication_comparison.csv`, `out/null_replication_gate.json` | null-replication gate |
| `out/marginals_match_gate.csv`, `out/null_replication_bytecheck.csv` | generation gates |
| `out/diagnostic_robustness.csv` | don't-build boundary robustness per phi_eff |
| `out/headline_summary.json` | machine-readable headline numbers |
| `out/provenance.jsonl` | per-catalog + phase provenance |
| `out/payloads/*.pkl` | rank-array cache (re-derivable from parts; input to analyze) |
| `results_headline_checks.py` | the exact filters behind every 6.x count |
| `mini_method_out.json`, `preview_method_out.json` | mini (131 datasets x 3 examples) and preview (3 datasets x 3 examples, 200-char truncation) variants of the method output, both exp_gen_sol_out-validated |
| `method_out.json`, `full_method_out.json` | single-file views (63.7 MB, 7,099 complete verbatim examples of the first 3 datasets: olist_all, olist_furniture_decor, phi_V100_EHIGH_z0.5_p0.25_s0; identical, hard-linked).  The complete output is too large for one file under the 100 MB GitHub limit, so these single views hold the verbatim front slice of the output and explicit metadata pointing at the complete split parts |
| `method_out/`, `full_method_out/`, `out/method_out/` | the COMPLETE method output (5.35 GB, 131 datasets, 632,262 examples) stored as 103 split parts each <= 61 MB (100 MB-per-file limit; the three views are hard links to the same inodes, i.e. one physical copy; reconstruction + provenance in `full_method_out/full_method_out_manifest.json`; `merge_and_split_outputs.py` rebuilds the split views from the canonical `out/method_out/` parts and verifies with `verify_merged_output.py`) |

## 9. Provenance / headline discipline

Every headline count in Section 6 is a deterministic filter over a named CSV in
this workspace (column + value given per row).  Generation parameters incl.
`phi`, `phi_effective`, `phi_capped`, `latent_D_c`, `n_distractor_tags`,
`latent_sample_seed`, `tag_encoding` are recorded in each catalog's
`family_params` (available in `out/payloads/*.pkl` `_extra.family_params_json`
and in the emitted JSON parts).  `out/provenance.jsonl` records per-catalog
eval status (131 ok) plus phase summaries.  No phi>0 claim in this README
predates the Section 6.1 gate; the tags given to content heuristics are the
exact `sign(z_v)` encoding with zero tag noise (C10), so the "tag-leak" probe of
the fallback plan is unnecessary — the feature interface is already perfect.