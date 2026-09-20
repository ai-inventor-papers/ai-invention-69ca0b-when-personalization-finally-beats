# Common catalog schema

One **catalog** = one small e-commerce shop: a restricted subset of a real
public log, OR one synthetic latent-intent family. All catalogs share this
single JSON structure so every cold-start heuristic reads identical evidence.

> RAW catalogs only — `family_params` (generation inputs) and `fold` are
> metadata; the corpus stores **no metrics, fitted models, or derived
> statistics** (HHI / entropy / headroom / MI are computed by the EXPERIMENT
> artifact, not stored here).

## Fields

| field | type | meaning |
|---|---|---|
| `catalog_id` | string | unique id, e.g. `uci_retail_1`, `synth_v200_s0` |
| `origin` | `"real"` \| `"synthetic"` | provenance class |
| `source_name` | string | e.g. `uci_online_retail_1`, `olist_brazilian_ecommerce`, `latent_intent_model` |
| `family_params` | object | generation / restriction inputs (metadata, reproducible) |
| `items` | array | catalog universe, `\|V\| ≤ ~1000` |
| `user_logs` | array | per-user chronological purchase histories + leave-last-out heldout |
| `forced_choice_probes` | array | forced-choice Q/A for the elicitation simulator |
| `fold` | `"screen"` \| `"confirm"` | pre-assigned screen vs confirm fold |
| `partition_meta` | object | label/outcome semantics |

## item

```json
{"item_id": "i00001",
 "attrs": {"cat_lights": 1.0},   // one-hot nominal attributes (content-kNN features)
 "category": "lights",            // nominal category (popularity/stratification key)
 "price_band": 3,                 // within-catalog 0..4 band (banded-popularity key)
 "attribute_diversity": 1}        // # active attributes (structural)
```

## user_log

```json
{"user_id": "u000123",
 "ordered_history": ["i00001","i00120","i00007"],  // chronological item_ids
 "heldout_item": "i00004",                         // the NEXT purchase (leave-last-out)
 "timestamps": [1600000000000,1600086400000,...]}  // epoch ms, strictly increasing
```

The heldout purchase occurs after the last history timestamp (temporal
leave-last-out). The cold-start k-grid `k ∈ {0,1,2,3,5,8}` is obtained by
truncating `ordered_history` to its first `k` items; the experiment aggregates
over the users whose history is long enough for each k.

## forced_choice_probe

```json
{"user_id": "u000123", "item_a": "i00001", "item_b": "i00400",
 "choice": 1,                     // 1 => item_a chosen/endorsed
 "ground_truth_p_a": 0.87}        // synthetic: model Bernoulli p; real: absent
```

- **Synthetic** probes: `answer ~ Bernoulli(p_A)` with
  `p_A = P(A|θ_u)/(P(A|θ_u)+P(B|θ_u))` optionally corrupted by
  `probe_noise` (see `family_params.probe_noise`). `ground_truth_p_a` records
  the underlying probability.
- **Real** probes are **revealed preference**: `item_a` = the user's most
  recent purchase, `item_b` = a popular item the user never bought,
  `choice = 1`, `prob_note = "revealed preference (A in history, B not)"`.

## family_params (metadata, not statistics)

- **Synthetic**: `n_items`, `n_attrs` (attribute entropy level), `zipf_s`
  (sales concentration), `alpha` (stickiness/heterogeneity), `beta`,
  `mean_hist`, `n_users`, `dynamic_turnover`, `horizon_days`, `probe_noise`,
  `n_probes`, `seed`.
- **Real**: `restriction_rule` (verbatim), `min_user_purchases`,
  `forced_choice_origin`.

## fold

- Real catalogs → always `"confirm"` (reserved for iteration-2 confirmation).
- Synthetic families → `sha256(catalog_id)[0] % 2`: `0`→confirm, `1`→screen
  (deterministic ~half / half).

## partition_meta

- Real: `{partition:"heldout", holdout:"last", temporal:"true", confirm_resource:"true"}`
- Synthetic: `{partition:"generation", generator:"latent_intent", n_items_total:<|V|>}`