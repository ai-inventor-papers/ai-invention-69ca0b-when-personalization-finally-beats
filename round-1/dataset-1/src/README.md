# Small e-commerce catalog screen corpus

Shared evidence corpus for **cold-start recommendation heuristic screening**
(popularity, banded popularity, content-kNN, hybrids) in the small-catalog
regime (`|V| ≤ ~1000 items`). Every catalog is written to ONE common schema so
that every heuristic reads identical evidence. The corpus is **RAW only**: it
stores items, purchase logs, forced-choice probes, generation params and fold
metadata — it contains **no metrics, no fitted models, no derived statistics**
(HHI / entropy / headroom / MI are the EXPERIMENT artifact's job).

Deliverable: `out/data_out.json` in the `exp_sel_data_out` shape, 25 catalogs.

---

## Contents

- `generator.py` — synthetic latent-intent family generator.
- `process_real.py` — converts the real public sources to the common schema.
- `assemble.py` — validates catalogs and builds `out/data_out.json`.
- `catalog_schema.json` / `catalog_schema.md` — the single common schema.
- `processed/` — one `<catalog_id>.json` per catalog (25 files).
- `raw/` — ephemeral downloads (not part of the final artifact).
- `out/data_out.json` (+ `full_/mini_/preview_` variants) — final corpus.

## Common schema (per catalog)

```
{catalog_id, origin(real|synthetic), source_name,
 family_params,                 # generation/restriction inputs (metadata)
 items[   {item_id, attrs(one-hot), category, price_band(0..4), attribute_diversity}],
 user_logs[{user_id, ordered_history[item_id...], heldout_item, timestamps[epoch_ms...]}],
 forced_choice_probes[{user_id, item_a, item_b, choice(0|1), ...}],
 fold(screen|confirm), partition_meta}
```

- `ordered_history` is chronological; the next (leave-last-out) purchase is
  `heldout_item`. The cold-start k-grid `{0,1,2,3,5,8}` arises by truncating
  `ordered_history` to its first `k` items (used at the group level by the
  experiment step).

## Fold assignment (deterministic)

- **ALL real catalogs → `confirm`** (scarce; reserved for iteration-2
  confirmation of the diagnostic).
- **Synthetic families → `sha256(catalog_id)[0] % 2`** (`0`→confirm,
  `1`→screen), giving a stable ~half / half split (11 screen / 8 confirm).

---

## Real catalogs (fold = confirm)

| catalog_id | source | restriction rule | items / users |
|---|---|---|---|
| `uci_retail_1` | UCI Online Retail (I) | top-N items by purchase-line count, ≤1000 | ~800 |
| `uci_retail_1_lights` | UCI Online Retail (I) | description-keyword subtree `lights` | ≤1000 |
| `uci_retail_2` | UCI Online Retail II | top-N items by purchase-line count, ≤1000 | ~1000 |
| `uci_retail_2_lights` | UCI Online Retail II | description-keyword subtree `lights` | ≤1000 |
| `olist_all` | Olist Brazilian E-Commerce | top-N products by purchase count, ≤1000 (min 4 purchases/user) | ~1000 |
| `olist_furniture_decor` | Olist | product-category subtree `furniture_decor` (min 4 purchases/user) | ≤1000 |

Provenance:

- **UCI Online Retail (I)** — University of California Irvine ML Repository,
  dataset 352 (also "Online Retail"). Real UK non-store online retailer,
  transactions 2010-12-01 → 2011-12-09. Public / academic use; no auth.
- **UCI Online Retail II** — UCI dataset 502. Same retailer, 2009-12 →
  2011-12. Public / academic use; no auth.
- **Olist Brazilian E-Commerce** — Kaggle `olistbr/brazilian-ecommerce`,
  100k orders 2016-2018. Mirror used: HF `aviahYadler/Olist_Ecommerce_Dataset`
  (auth-free). License per Kaggle page.

Per-source processing (see `process_real.py`):

- UCI: keep positive-quantity purchase lines; per-item median `UnitPrice` →
  0..4 price band; coarse nominal category derived from description keywords
  (`lights/candles/holders/christmas/bags/boxes/vases/mugs/clocks/bears/other`).
- Olist: join orders ↔ order_items ↔ products (+EN category translation);
  category = English product category; price band from per-product median price.
- Every user_logs entry is timestamp-sorted and requires ≥ 5 purchases (so
  `k ∈ {0,1,2,3}` is computable for all, and `k ∈ {5,8}` for the subset of
  longer-history users). `forced_choice_probes` for real data are
  **revealed-preference** pairs: `item_a` = user's most recent purchase,
  `item_b` = a popular item the user never bought, `choice = 1` (documented in
  `prob_note`). No model probability is attached to real probes.

## Synthetic families (fold = screen/confirm)

Latent-intent generator (`generator.py`), ~19 families spanning the factorial
design as a main-effects sweep around a reference cell
(`|V|=200, Zipf s=1.0, K=6 attrs, alpha=0.3 sticky, mean_hist=14, dynamic`):

Catalog assets: item `i` gets a nominal one-hot attribute (`cat_k`), a price
band, and a Zipf popularity weight `base_i ∝ rank_i^(-1/s)`.

Demand / turnover: `w_i(t) = base_i · exp(drift_i · frac(t))` when dynamic
(`drift_i ~ N(0,1.5)`, 10% "trending" items boosted) else `w_i(t) = base_i`.

Users: `theta_u ~ Dirichlet(alpha·1_K)` (small `alpha` → peaked/sticky users;
large `alpha` → homogeneous). Purchase probability is a softmax/Bradley-Terry
blend `P_i(t) ∝ exp(beta · theta_u[cat_i]) · w_i(t)`. History length
`L ~ max(3, Poisson(mean_hist))`; the `(L+1)`-th draw is `heldout_item`
(temporal leave-last-out). Timestamps advance with the draw fraction over the
`horizon_days` window.

Forced-choice response model: `p_A = P(A)/(P(A)+P(B))`,
`answer ~ Bernoulli((1-eps)·p_A + eps/2)` with `eps = probe_noise`. Emitted as
`forced_choice_probes` with `ground_truth_p_a`.

Every generation parameter is recorded verbatim in `family_params` per catalog,
including the `seed`, so the whole grid is reproducible byte-for-byte.

## Running

```bash
uv venv .venv --python=3.12 && source .venv/bin/activate
uv pip install pandas numpy loguru openpyxl requests
python generator.py      # -> processed/synth_*.json      (19 families)
python process_real.py   # -> processed/uci_*_*.json, olist_*.json  (6 real)
python assemble.py       # -> out/data_out.json           (25 catalogs)
```