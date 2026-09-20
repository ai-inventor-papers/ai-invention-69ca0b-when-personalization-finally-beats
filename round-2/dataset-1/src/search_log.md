# Acquisition search log — additional small e-commerce purchase logs

Deliverable per `gen_plan_dataset_1`: acquire and standardize genuinely small
(`<= 1000`-item) e-commerce purchase logs with item metadata (category + price)
to widen the real-catalog **confirm** evidence, **or** document how thin the
public small-catalog data landscape is. **Outcome: 0 additional catalogs were
accepted against the pre-registered bar.** This log records every source tried
and the exact reason each fails. Every count traces to a named file + filter.

## Acceptance bar (all must hold)
1. restricted `n_items <= 1000` (target 150-900);
2. usable nominal **category** per item AND **price-band** per item (category
   required; price may be honestly absent but never the sole evidence);
3. timestamped **per-user** purchase logs;
4. `>= ~100` users each with `>= 4` purchases (so the `k-grid {0,1,2,3,5,8}` is
   non-degenerate);
5. license permits redistribution;
6. post-restriction size well under 300MB.

The bar was **not** lowered to inflate counts (plan Phase 5 rule).

---

## A. Retailrocket (CIKM 2016) — designated category-tree source
- **Acquired**: full raw files from the public Git-LFS mirror
  `sabin74/Retailrocket-Recommender-System` (batch LFS download; sha256-verifiable):
  `events.csv` (sha256 `3745aa83238b1e6d...`, 94,237,913 B, 2,756,101 rows),
  `item_properties_part1.csv` (`30aad5aeca58b2dc...`, 484,315,749 B),
  `item_properties_part2.csv` (`d5e7d1a91dc4...`, 408,929,907 B),
  `category_tree.csv` (14,454 B, 1669 nodes, 25 top-level roots). events mirror also
  cross-checked against `FRIEDparrot/RecommendSystem_practice` (identical 94,237,913 B).
- **Schema found**: `events.csv` = `timestamp,visitorid,event,itemid,transactionid` where
  `event` is a label in {`view`,`addtocart`,`transaction`}; `item_properties*.csv` =
  `timestamp,itemid,property,value` with `property=='categoryid'` giving a flat category id
  (20,275,902 property rows total).
- **Filter used** (`build_catalogs.py`): keep `event=='transaction'` only (22,457 rows); keep
  items whose modal `categoryid` is in the category tree (417,053 items labelled, of which
  11,645 were ever purchased). Map item -> top-level category via upward parent-walk.
- **Restrictions tried** (`build_catalogs.py`, MIN_PURCHASES in {4,5}):
  - `retailrocket_top900` (top-900 items by purchase-line count across all categories):
    900 items, **users_ge4 = 96, users_ge5 = 83**.
  - per-subtree top-900 (largest top-level categories): cat140 -> ge4=81/ge5=67,
    cat1532 -> 37/23, cat1600 -> 41/31, cat1482 -> 21/13, cat395 -> 40/33,
    cat653 -> 28/22, cat1224 -> 31/22, others < 10.
- **Verdict: REJECT.** The largest accessible configuration yields only **96 users with
  >= 4 purchases**, marginally below the `~100` floor, so the confirm k-grid would be
  degenerate; and the source has **no price field** (criterion 2 partial, price-less ->
  peripheral only). The canonical ~115k-transaction events of the paper version is only
  redistributed via Kaggle (login-gated), so this is the most transaction-complete freely
  obtainable copy.

## B. Dunnhumby — designated price-bearing source
- dunnhumby.com/source-files: the full 9-part "lets-get-sort-of-real" RDS (4.3GB) is served
  through **gated** Contentful buttons (gatedForm/gatedButton); only tiny ungated samples exist
  (`Data-Sample` 22,437 B, `Sample-2K-baskets`), far below the 100-user bar.
- Kaggle `frtgnn/dunnhumby-the-complete-journey` and `retailrocket/ecommerce-dataset` require
  login (no Kaggle token present in this environment).
- HF mirror `54-acme/dunnhumby_2019` (plan candidate) returns gated/404 on the Hub API.
- GitHub mirrors (`RecoHut-Datasets/dunnhumby`, and >25 `dunnhumby`-named repos) are empty or
  contain no committed `transaction_data.csv`.
- **Verdict: REJECT (not obtainable);** price-bearing real-catalog evidence is therefore
  unavailable in this environment.

## C. Boutique / marketplace / small-shop sweep (HuggingFace + web)
- ~50 broad terms searched (HF API + aii-hf-datasets + web): 'retail', 'ecommerce',
  'purchase history', 'customer transactions', 'transactions', 'supermarket', 'shop', 'store
  sales', 'coffee shop', 'bookstore', 'bakery', 'cosmetics', 'watches', 'jewelry',
  'marketplace', 'small business', 'receipts', 'loyalty', 'online retail', 'recommendation'.
- Previewed 30+ candidate families; **none** combined per-user timestamped purchases +
  item category + price in a <=1000-item, freely-downloadable, properly-licensed log.
  Representative rejects: Maven Roasters coffee-shop (category+price+timestamp but **no user
  id**); Northwind purchase orders (PDF documents, and classic Northwind is only ~77 products /
  ~91 customers); online shopping reviews (text classification); banking/fraud/crypto
  (not shop purchases); OCR receipts (images); product-catalog benchmarks (Shopify, Shopee,
  powerline, "large-scale catalogue" i.e. not purchase logs).
- Olist / UCI Online Retail mirrors were consciously skipped (already in the iteration-1 corpus).
- **Verdict: REJECT (zero qualifying).** The public, freely-downloadable landscape of genuinely
  small per-user purchase logs with item metadata is, in this environment, effectively empty.

## D. Additional workspace-found raw candidate files
Three raw candidate files were present under `raw/` (pre-downloaded):
- `ecommerce_orders.csv` (10,000 rows) — **REJECT**: no citeable source and a clear synthetic
  signature (365 distinct `order_date` values for 10,000 rows, up to 45 rows sharing one
  timestamp; sequential `customer_id` 1..2999, `product_id` 1..1000; generated addresses).
  Accepting it as real catalog evidence would violate the no-fabricated-provenance rule.
- `supermarket_germany.csv` (1,000 rows) — **REJECT**: only **6 distinct items** (the
  `Product_line` categories; no SKU column), so `n_items` is far below the 150-item floor.
- `tafeng_D11-02.zip` (91 KB) — **REJECT**: corrupt/truncated transfer (zip central directory
  missing); full Ta Feng is not small and its redistribution provenance is unverified.)

## Reproducibility
- `raw/fetch_rr_lfs.py` — LFS download of the Retailrocket files (sha-verified).
- `explore_rr.py` — event-type / category-tree / per-subtree counts.
- `build_catalogs.py` — restriction + acceptance check; emits `processed/*` only for passing
  catalogs (none passed).
- `rejections.json` — machine-readable rejection records (source/attempt/criterion/counts).
- `full_data_out.json` (+`mini_`/`preview_`) — exp_sel_data_out-validated aggregate (64 rejection
  examples) carrying `n_accepted: 0` so any downstream confirm-round discovery sees the honest result.

## Statement for the paper's corpus contribution
No additional real catalog met the pre-registered bar from freely-downloadable sources:
Retailrocket's most transaction-complete free mirror fails both the 100-user floor (96 users
with >= 4 purchases in the best configuration) and the price requirement; Dunnhumby is gated;
the boutique sweep is effectively empty. This confirms how thin the public small-catalog data
landscape is and is reported as a finding, not padded with near-miss catalogs.