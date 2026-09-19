#!/usr/bin/env python3
"""Synthetic latent-intent e-commerce catalog generator.

Implements the latent-intent user model described in the artifact plan:
  * Catalog: |V| items, each with a nominal (one-hot) attribute vector over K
    attribute groups, a price band, and a time-varying popularity weight.
  * Demand:  item marginal popularity ~ Zipf(s) (sales concentration) with
    optional exponential turnover drift per item and occasional single-item
    spikes (dynamic turnover) or none (static).
  * Users:   per-user latent preference theta_u ~ Dirichlet(alpha) over the K
    attribute groups. Small alpha -> peaked/sticky; large alpha -> homogeneous.
  * Purchase:  P(user u buys item i at time t) ∝ exp(beta * <theta_u, phi_i>)
               * w_i(t)   (softmax / Bradley-Terry blend with popularity).
  * Output:   ordered history (temporal) + heldout_item (next purchase) per
              user; the cold-start k-grid arises by truncating history.
  * Forced choice probes: Bernoulli(p) answer, p = P(A)/(P(A)+P(B)) with
              optional corruption noise eps.

Emit RAW catalogs only (generation params in family_params, fold as metadata).
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

T0_MS = 1_600_000_000_000  # 2020-09-13 epoch ms anchor
DAY_MS = 86_400_000


# --------------------------------------------------------------------------- #
# RNG helpers
# --------------------------------------------------------------------------- #
def _seeded(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Catalog asset generation
# --------------------------------------------------------------------------- #
def make_items(
    rng: np.random.Generator,
    n_items: int,
    n_attrs: int,
    zipf_s: float,
    forced_zipf: bool = True,
) -> tuple[list[dict], dict]:
    """Build n_items with: one-hot attr (category), price band, base weight."""
    if forced_zipf:
        # Deterministic Zipf ranks: rank 0 dominates, exponent s.
        ranks = np.arange(1, n_items + 1)
        base_w = ranks ** (-1.0 / zipf_s)
        # assign items to the n_items of the Zipf ladder via shuffled ranks to
        # decouple item id order from popularity.
        perm = rng.permutation(n_items)
        base_w = base_w[np.argsort(np.argsort(perm))]  # keep ladder, permuted ids
    else:
        base_w = rng.zipf(zipf_s, size=n_items).astype(float)
    base_w = base_w / base_w.sum()

    # attribute (category) assignment: entropy is governed by n_attrs; use a
    # categorical prior that itself is Zipf-concentrated when n_attrs is small
    # and flatter when large, so 'entropy' level changes composition.
    cat_prior = (1.0 / np.arange(1, n_attrs + 1)) ** 1.5
    cat_prior = cat_prior / cat_prior.sum()
    cats = rng.choice(n_attrs, size=n_items, p=cat_prior)

    items = []
    for i in range(n_items):
        attrs = {f"cat_{int(cats[i])}": 1.0}
        items.append(
            {
                "item_id": f"i{i:05d}",
                "attrs": attrs,
                "category": int(cats[i]),
                "price_band": 0,  # filled below
                "attribute_diversity": 1,
                "base_weight": float(base_w[i]),
            }
        )

    # price bands: categories differ in pricing; map to 0..4 quantiles.
    cat_price_center = rng.uniform(1.0, 5.0, size=n_attrs)
    prices = np.array(
        [cat_price_center[it["category"]] * rng.lognormal(0.0, 0.6) for it in items]
    )
    edges = np.quantile(prices, [0.2, 0.4, 0.6, 0.8])
    bands = np.digitize(prices, edges)  # 0..4
    for it, b in zip(items, bands):
        it["price_band"] = int(b)

    cat_meta = {"n_attrs": n_attrs, "cat_prior_zipf_exp": 1.5}
    return items, cat_meta


# --------------------------------------------------------------------------- #
# Time-varying weights
# --------------------------------------------------------------------------- #
def make_turnover(
    rng: np.random.Generator, n_items: int, dynamic: bool, horizon_days: int
) -> dict[str, Any]:
    """Return per-item drift (log-rate) and, for dynamic mode, spike params."""
    if not dynamic:
        return {"dynamic": False, "drift": np.zeros(n_items), "spike_prob": 0.0}
    # drift ~ N(0, 1.5) on a log scale over the whole horizon; spikes rare but
    # strong for a few "trending" items.
    drift = rng.normal(0.0, 1.5, size=n_items)
    trend_mask = rng.random(n_items) < 0.1
    drift = drift + trend_mask * rng.uniform(2.0, 4.0, size=n_items)
    spike_prob = 0.0
    return {
        "dynamic": True,
        "drift": drift,
        "spike_prob": spike_prob,
        "spike_items": None,
        "horizon_days": horizon_days,
    }


def w_at(turn: dict, i: int, frac: float) -> float:
    """Popularity factor w_i(t); frac in [0,1] across the window."""
    if not turn["dynamic"]:
        return 1.0
    return float(np.exp(turn["drift"][i] * frac))


# --------------------------------------------------------------------------- #
# Sequence generation
# --------------------------------------------------------------------------- #
def sample_histories(
    items: list[dict],
    turn: dict,
    n_users: int,
    mean_hist: int,
    alpha: float,
    beta: float,
    seed: int,
    n_probes: int = 3,
    probe_noise: float = 0.0,
    frac_per_item: float = 0.0,
) -> dict:
    """Simulate user logs via the latent-intent model.

    Returns a dict with user_logs and forced_choice_probes.
    """
    rng = _seeded(seed)
    n_items = len(items)
    cats = np.array([it["category"] for it in items])
    base_w = np.array([it["base_weight"] for it in items])
    n_attrs = int(cats.max()) + 1

    # category-level popularity for fast static sampling
    cat_agg = np.zeros(n_attrs)
    cat_switch = np.zeros(n_attrs, dtype=int)  # unused, placeholder

    def cat_partition() -> list[np.ndarray]:
        return [np.where(cats == c)[0] for c in range(n_attrs)]

    partitions = cat_partition()

    logs = []
    probes = []

    # history length per user; drop users with < 3 purchases (cold-start floor)
    hist = np.maximum(3, rng.poisson(mean_hist, size=n_users))

    for u in range(n_users):
        theta = rng.dirichlet(np.full(n_attrs, alpha))  # latent preference
        uid = f"u{u:06d}"
        L = int(hist[u])
        # per-item affinity: exp(beta * theta[cat_i]) — constant per user
        affinity = np.exp(beta * theta[cats])
        frac_step = 1.0 / max(L + 1, 1)  # time fraction advance per step

        # sample one purchase at a time so time-varying weights are exact;
        # probability at step s:  (affinity_i * base_w_i * w_i(frac_s))
        history = []
        timestamps = []
        for s in range(L + 1):  # L history + 1 heldout
            frac = (s + 0.5) * frac_step
            w = np.array(
                [base_w[i] * w_at(turn, i, frac) for i in range(n_items)],
                dtype=float,
            )
            p = affinity * w
            p = p / p.sum()
            choice = int(rng.choice(n_items, p=p))
            if s < L:
                history.append(items[choice]["item_id"])
                ts = T0_MS + int((frac) * turn.get("horizon_days", 365) * DAY_MS)
                timestamps.append(ts)
            else:
                heldout = items[choice]["item_id"]

        # forced-choice probes: item in user's preferred category vs a random
        # category item, answered by the same model.
        nonempty = [c for c in range(n_attrs) if len(partitions[c]) > 0]
        pref_cat = int(nonempty[int(theta[nonempty].argmax())])
        cands = partitions[pref_cat]
        other_cats = [c for c in nonempty if c != pref_cat]
        if len(cands) >= 2 and other_cats:
            for _ in range(n_probes):
                a = int(rng.choice(cands))
                oc = int(rng.choice(other_cats))
                b = int(rng.choice(partitions[oc]))
                frac = 1.0
                wA = base_w[a] * w_at(turn, a, frac)
                wB = base_w[b] * w_at(turn, b, frac)
                pA = (affinity[a] * wA) / (
                    affinity[a] * wA + affinity[b] * wB
                )
                if probe_noise > 0:
                    pA = (1.0 - probe_noise) * pA + probe_noise * 0.5
                choice = 1 if rng.random() < pA else 0
                probes.append(
                    {
                        "user_id": uid,
                        "item_a": items[a]["item_id"],
                        "item_b": items[b]["item_id"],
                        "choice": choice,
                        "ground_truth_p_a": float(pA),
                    }
                )

        logs.append(
            {
                "user_id": uid,
                "ordered_history": history,
                "heldout_item": heldout,
                "timestamps": timestamps,
            }
        )

    # ensure timestamps sorted (they are by construction) and non-empty
    return {"user_logs": logs, "probes": probes}


# --------------------------------------------------------------------------- #
# Family assembly
# --------------------------------------------------------------------------- #
def build_family(params: dict) -> dict:
    """Generate one full synthetic catalog (family) from cell+seed params."""
    rng = _seeded(params["seed"])
    horizon_days = params.get("horizon_days", 365)

    items, cat_meta = make_items(
        rng,
        params["n_items"],
        params["n_attrs"],
        params["zipf_s"],
        forced_zipf=True,
    )
    turn = make_turnover(rng, params["n_items"], params["dynamic"], horizon_days)

    sim = sample_histories(
        items,
        turn,
        params["n_users"],
        params["mean_hist"],
        params["alpha"],
        params["beta"],
        seed=params["seed"] + 1000,
        n_probes=params.get("n_probes", 3),
        probe_noise=params.get("probe_noise", 0.0),
    )

    # fold: deterministic seeded hash -> ~half screen / half confirm
    digest = hashlib.sha256(params["catalog_id"].encode()).digest()[0]
    fold = "confirm" if (digest % 2 == 0) else "screen"

    catalog = {
        "catalog_id": params["catalog_id"],
        "origin": "synthetic",
        "source_name": "latent_intent_model",
        "family_params": {
            "n_items": params["n_items"],
            "n_attrs": params["n_attrs"],
            "attribute_diversity": cat_meta,
            "zipf_s": params["zipf_s"],
            "alpha": params["alpha"],
            "beta": params["beta"],
            "mean_hist": params["mean_hist"],
            "n_users": params["n_users"],
            "dynamic_turnover": params["dynamic"],
            "horizon_days": horizon_days,
            "probe_noise": params.get("probe_noise", 0.0),
            "n_probes": params.get("n_probes", 3),
            "seed": params["seed"],
        },
        "items": [
            {
                "item_id": it["item_id"],
                "attrs": it["attrs"],
                "category": it["category"],
                "price_band": it["price_band"],
                "attribute_diversity": it["attribute_diversity"],
            }
            for it in items
        ],
        "user_logs": sim["user_logs"],
        "forced_choice_probes": sim["probes"],
        "fold": fold,
        "partition_meta": {
            "partition": "generation",
            "generator": "latent_intent",
            "n_items_total": params["n_items"],
        },
    }
    return catalog


# --------------------------------------------------------------------------- #
# Fixed grid of 20 synthetic families spanning the factorial dimensions
# --------------------------------------------------------------------------- #
def family_grid() -> list[dict]:
    """Return the ~20 (cell, seed) family param dicts for the 25-catalog corpus.

    Designed as a main-effects sweep around a reference cell so that the
    experiment can read each dimension's factor levels from family_params:
      reference: |V|=200, Zipf s=1.0, K=6 attrs, alpha=0.3 (sticky),
                 mean_hist=14, dynamic turnover, seed 0.
    """
    ref = dict(
        n_items=200,
        zipf_s=1.0,
        n_attrs=6,
        alpha=0.3,
        mean_hist=14,
        n_users=1200,
        dynamic=True,
        beta=2.0,
        horizon_days=365,
        n_probes=3,
        probe_noise=0.0,
    )
    cells = []

    def add(name, mutate, seeds):
        for sd in seeds:
            p = dict(ref)
            p.update(mutate)
            p["seed"] = sd
            p["catalog_id"] = f"synth_{name}_s{sd}"
            cells.append(p)

    # |V| sweep (others fixed, seed 0); plus seed replicates for extremes
    add("v20", {"n_items": 20, "n_users": 600}, [0])
    add("v50", {"n_items": 50, "n_users": 600}, [0])
    add("v200", {"n_items": 200, "n_users": 1200}, [0, 1, 2, 3])
    add("v1000", {"n_items": 1000, "n_users": 2500}, [0, 1, 2])

    # Zipf sweep (at |V|=200)
    add("zipf05", {"zipf_s": 0.5}, [0])
    add("zipf15", {"zipf_s": 1.5}, [0])
    add("zipf20", {"zipf_s": 2.0}, [0, 1])

    # attribute entropy sweep
    add("attr_many", {"n_attrs": 30}, [0])
    add("attr_hetero", {"n_attrs": 90, "alpha": 1.2}, [0])

    # stickiness sweep (alpha)
    add("stick_low", {"alpha": 1.5}, [0])

    # history-length sweep
    add("hist_short", {"mean_hist": 8}, [0])
    add("hist_long", {"mean_hist": 24}, [0])

    # turnover sweep
    add("tstat", {"dynamic": False}, [0])

    return cells


def main() -> None:
    grid = family_grid()
    logger.info(f"Grid has {len(grid)} synthetic families")
    import os
    from pathlib import Path as _P
    outdir = _P(os.path.dirname(os.path.abspath(__file__))) / "processed"
    outdir.mkdir(exist_ok=True)
    for p in grid:
        cat = build_family(p)
        (outdir / f"{cat['catalog_id']}.json").write_text(
            json.dumps(cat)
        )
        logger.info(f"wrote {cat['catalog_id']}: {len(cat['items'])} items, "
                    f"{len(cat['user_logs'])} users, fold={cat['fold']}")


if __name__ == "__main__":
    from pathlib import Path
    main()