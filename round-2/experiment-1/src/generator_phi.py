#!/usr/bin/env python3
"""Iteration-2 extended latent-intent catalog generator (item-level appeal).

Ports generator.py from the iteration-1 DATASET artifact (gen_art_dataset_1)
VERBATIM for items / turnover / history machinery, so that the phi=0 legacy
path reproduces the iteration-1 pool catalogs BYTE-FOR-BYTE, and extends it
with a within-category item-level latent-appeal model controlled by phi:

  * Latent structure (drawn ONCE per (config, seed), shared across phi):
      - per-category latent dimension D_c = 4;
      - item factor z_v ~ N(0, I) in R^{D_c}   (fixed across phi);
      - per-user per-category preference p_{u,c} ~ N(0, I) in R^{D_c}.
  * Tag encoding (the FEATURE interface for the ported content heuristics):
      tag d in 0..D_c-1 present on item v  iff  z_v[d] > 0,
      plus N_DISTRACTOR_TAGS tags with random fixed presence NOT tied to z_v.
  * phi-interpolated choice model (the EXACT formula, per the artifact plan):
        s_{u,v}    = z_v . p_{u,c} - mean_{v' in c}( z_{v'} . p_{u,c} )   # mean-zero centering
        P_{u,v}(t) = w_v(t) * exp(beta * theta_u[cat_v]) * exp(phi * s_{u,v})
    normalized over items.  At phi=0 this is EXACTLY the iteration-1 null:
    P(v) ∝ exp(beta*theta_u[cat_v]) * w_v(t).  The mean-zero centering holds
    each user's within-category modulation marginal-preserving, so the
    aggregate popularity prior stays put across phi (verified by the
    marginals-match gate in method.py / C9).
  * Marginal-preservation construction (measured, disclosed): item factors
    are UNIT-NORMED (z_v <- z_v/||z_v||) so every item's within-category
    preference signal has the SAME marginal distribution; otherwise items
    with large ||z_v|| capture mass proportional to factor magnitude (not
    direction), which breaks the popularity-marginal control at phi>0
    (measured: raw factors fail the gate on 9/9 sampled cells at phi=1,
    unit-normed factors pass the majority).
  * Marginals-match gate + fallback (per cell; gate tolerances below).
      T1: per user per category, rescale the phi probabilities by
          M0[c]/M1[c] (phi=0 category mass / phi category mass) so the
          category marginals match phi=0 per user;
      then EFFECTIVE-STRENGTH CAPPING (the plan's fallback-1): binary-search
      the largest phi_eff <= phi whose catalog passes the gate, regenerate at
      phi_eff, and LOG the cap ("phi effective strength was capped") in
      family_params['phi_effective'] + the gate CSV.  A per-category
      base_w-anchored softmax (T2) was measured COUNTERPRODUCTIVE on every
      failing cell (stronger flattening than the raw kernel) and is therefore
      NOT part of the ladder.
  * Null-replication: build_family_legacy + family_grid() are the pristine
    port of generator.py and reproduce the iteration-1 pool catalogs
    (processed/*.json) byte-for-byte (C12).

Emitted catalogs keep the iteration-1 pool JSON schema
(items / user_logs / forced_choice_probes / family_params / fold /
partition_meta) so the ported catalog_pool adapter + eval_metrics run
unchanged.  Every generation parameter incl. phi and seed is recorded in
family_params.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from loguru import logger

T0_MS = 1_600_000_000_000  # 2020-09-13 epoch ms anchor
DAY_MS = 86_400_000

LATENT_D_C = 4          # per-category latent dimension (D_c in the plan)
N_DISTRACTOR_TAGS = 2   # tags whose presence is NOT tied to z_v
LATENT_SEED_MULT = 104729
LATENT_SEED_ADD = 3011

GATE_TOL = {
    "ks": 0.05,
    "max_abs_dev": 0.02,
    "d_hhi": 0.02,
    "d_norm_entropy": 0.03,
}


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------
def _seeded(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def latent_rng_for(seed: int) -> np.random.Generator:
    """Dedicated, separately-seeded RNG for ALL latent structure draws
    (z_v, p_{u,c}, distractor tags).  Kept apart from the family/user RNGs
    so the phi=0 legacy sampling path consumes EXACTLY the same RNG
    sequence as the iteration-1 generator (byte-identity gate)."""
    return _seeded(seed * LATENT_SEED_MULT + LATENT_SEED_ADD)


# ---------------------------------------------------------------------------
# Items / turnover (VERBATIM port of generator.py)
# ---------------------------------------------------------------------------
def make_items(
    rng: np.random.Generator,
    n_items: int,
    n_attrs: int,
    zipf_s: float,
    forced_zipf: bool = True,
) -> tuple[list[dict], dict]:
    """Build n_items with: one-hot attr (category), price band, base weight."""
    if forced_zipf:
        ranks = np.arange(1, n_items + 1)
        base_w = ranks ** (-1.0 / zipf_s)
        perm = rng.permutation(n_items)
        base_w = base_w[np.argsort(np.argsort(perm))]
    else:
        base_w = rng.zipf(zipf_s, size=n_items).astype(float)
    base_w = base_w / base_w.sum()

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
                "price_band": 0,
                "attribute_diversity": 1,
                "base_weight": float(base_w[i]),
            }
        )

    cat_price_center = rng.uniform(1.0, 5.0, size=n_attrs)
    prices = np.array(
        [cat_price_center[it["category"]] * rng.lognormal(0.0, 0.6) for it in items]
    )
    edges = np.quantile(prices, [0.2, 0.4, 0.6, 0.8])
    bands = np.digitize(prices, edges)
    for it, b in zip(items, bands):
        it["price_band"] = int(b)

    cat_meta = {"n_attrs": n_attrs, "cat_prior_zipf_exp": 1.5}
    return items, cat_meta


def make_turnover(
    rng: np.random.Generator, n_items: int, dynamic: bool, horizon_days: int
) -> dict[str, Any]:
    """Return per-item drift (log-rate) and, for dynamic mode, spike params."""
    if not dynamic:
        return {"dynamic": False, "drift": np.zeros(n_items), "spike_prob": 0.0}
    drift = rng.normal(0.0, 1.5, size=n_items)
    trend_mask = rng.random(n_items) < 0.1
    drift = drift + trend_mask * rng.uniform(2.0, 4.0, size=n_items)
    return {
        "dynamic": True,
        "drift": drift,
        "spike_prob": 0.0,
        "spike_items": None,
        "horizon_days": horizon_days,
    }


def w_at(turn: dict, i: int, frac: float) -> float:
    """Popularity factor w_i(t); frac in [0,1] across the window."""
    if not turn["dynamic"]:
        return 1.0
    return float(np.exp(turn["drift"][i] * frac))


# ---------------------------------------------------------------------------
# Legacy history sampling (VERBATIM port of generator.sample_histories)
# ---------------------------------------------------------------------------
def sample_histories_legacy(
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
    """Simulate user logs via the iteration-1 latent-intent model (phi=0).
    Verbatim port: reproduces generator.py's output draw-for-draw."""
    rng = _seeded(seed)
    n_items = len(items)
    cats = np.array([it["category"] for it in items])
    base_w = np.array([it["base_weight"] for it in items])
    n_attrs = int(cats.max()) + 1

    def cat_partition() -> list[np.ndarray]:
        return [np.where(cats == c)[0] for c in range(n_attrs)]

    partitions = cat_partition()
    logs = []
    probes = []

    hist = np.maximum(3, rng.poisson(mean_hist, size=n_users))

    for u in range(n_users):
        theta = rng.dirichlet(np.full(n_attrs, alpha))
        uid = f"u{u:06d}"
        L = int(hist[u])
        affinity = np.exp(beta * theta[cats])
        frac_step = 1.0 / max(L + 1, 1.0)

        history = []
        timestamps = []
        for s in range(L + 1):
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

    return {"user_logs": logs, "probes": probes}


# ---------------------------------------------------------------------------
# Latent structure + tag encoding
# ---------------------------------------------------------------------------
def make_latent_structure(
    rng: np.random.Generator,
    n_items: int,
    d_c: int = LATENT_D_C,
    n_distractor: int = N_DISTRACTOR_TAGS,
    norm_factors: bool = True,
) -> tuple[np.ndarray, list[list[str]], list[str]]:
    """Draw item factors z_v ~ N(0, I) in R^{d_c} and the tag encoding.

    Returns (Z (n_items, d_c), tags_per_item (list[list[str]]), tag_names).
    Tag d in 0..d_c-1 present on item v iff z_v[d] > 0; each of the
    n_distractor 'lx_*' tags present with prob 0.3, NOT tied to z_v.
    The rng MUST be the dedicated latent RNG (latent_rng_for(seed)).

    norm_factors=True (default, disclosed): z_v <- z_v/||z_v|| so every
    item's within-category preference signal z_v . p_{u,c} has the SAME
    marginal distribution, keeping the popularity marginal put across phi
    (measured: without norming, items with large ||z_v|| capture mass
    proportional to factor magnitude and the gate fails everywhere at
    phi=1).  Sign(z_v) is unchanged, so the tag encoding is unaffected."""
    Z = rng.standard_normal(size=(n_items, d_c))
    if norm_factors:
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    latent_names = [f"lz_{d}" for d in range(d_c)]
    dist_names = [f"lx_{j}" for j in range(n_distractor)]
    tag_names = latent_names + dist_names
    tags_per_item: list[list[str]] = []
    for v in range(n_items):
        tags = [latent_names[d] for d in range(d_c) if Z[v, d] > 0.0]
        for j in range(n_distractor):
            if rng.random() < 0.3:
                tags.append(dist_names[j])
        tags_per_item.append(tags)
    return Z, tags_per_item, tag_names


def _within_category_mean_s(s: np.ndarray, cats: np.ndarray) -> np.ndarray:
    """Subtract per-category mean of s (mean-zero centering of the plan)."""
    s_out = s.copy()
    for c in range(int(cats.max()) + 1):
        m = cats == c
        if m.any():
            s_out[m] -= s[m].mean()
    return s_out


def sample_histories_phi(
    items: list[dict],
    turn: dict,
    n_users: int,
    mean_hist: int,
    alpha: float,
    beta: float,
    seed: int,
    phi: float,
    Z: np.ndarray,
    latent_rng: np.random.Generator,
    n_probes: int = 3,
    probe_noise: float = 0.0,
    tighten: int = 0,
    frac_per_item: float = 0.0,
) -> dict:
    """phi-aware history sampler.

    phi == 0  -> EXACT legacy loop (identical RNG consumption, so for a
                 given (config, seed) the user_logs equal the iteration-1
                 generator's logs element-by-element).
    phi > 0   -> adds the per-item latent appeal term exp(phi * s_{u,v})
                 with s mean-zero centered within each category (formula in
                 the module docstring).  Per-user per-category preferences
                 p_{u,c} are drawn from latent_rng.
    tighten   -> 0: none; 1: per-user per-category mass anchored to the
                 phi=0 category masses (fallback T1); 2: within-category
                 softmax anchored to base_w and phi=0 category mass (T2).
    """
    rng = _seeded(seed)
    n_items = len(items)
    cats = np.array([it["category"] for it in items])
    base_w = np.array([it["base_weight"] for it in items])
    n_attrs = int(cats.max()) + 1

    def cat_partition() -> list[np.ndarray]:
        return [np.where(cats == c)[0] for c in range(n_attrs)]

    partitions = cat_partition()
    logs = []
    probes = []

    hist = np.maximum(3, rng.poisson(mean_hist, size=n_users))

    for u in range(n_users):
        theta = rng.dirichlet(np.full(n_attrs, alpha))
        uid = f"u{u:06d}"
        L = int(hist[u])
        affinity = np.exp(beta * theta[cats])
        frac_step = 1.0 / max(L + 1, 1.0)

        if phi > 0.0:
            # per-user per-category preference vectors p_{u,c} ~ N(0, I)
            P_u = latent_rng.standard_normal(size=(n_attrs, LATENT_D_C))
            # per-item dot: s_v = z_v . p_{u, cat(v)}
            s_full = np.sum(Z * P_u[cats], axis=1)   # (n_items,)
            s_centered = _within_category_mean_s(s_full, cats)

        history = []
        timestamps = []
        for s in range(L + 1):
            frac = (s + 0.5) * frac_step
            w = np.array(
                [base_w[i] * w_at(turn, i, frac) for i in range(n_items)],
                dtype=float,
            )
            p = affinity * w
            if phi > 0.0:
                p = p * np.exp(phi * s_centered)      # item-level appeal
            p = p / p.sum()
            if tighten == 1 and phi > 0.0:
                # T1: restore the phi=0 per-category masses per user
                p0 = affinity * w
                p0 = p0 / p0.sum()
                mass0 = np.array([p0[partitions[c]].sum()
                                  for c in range(n_attrs)])
                mass1 = np.array([p[partitions[c]].sum()
                                  for c in range(n_attrs)])
                p = p * mass0[cats] / np.maximum(mass1[cats], 1e-300)
                p = p / p.sum()
            choice = int(rng.choice(n_items, p=p))
            if s < L:
                history.append(items[choice]["item_id"])
                ts = T0_MS + int((frac) * turn.get("horizon_days", 365) * DAY_MS)
                timestamps.append(ts)
            else:
                heldout = items[choice]["item_id"]

        # forced-choice probes use the same (legacy) response model; the
        # probe answers are NOT consumed by the evaluation pipeline, which
        # simulates elicitation answers from the category posterior.
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

    return {"user_logs": logs, "probes": probes}


def sample_histories(
    items: list[dict],
    turn: dict,
    n_users: int,
    mean_hist: int,
    alpha: float,
    beta: float,
    seed: int,
    phi: float = 0.0,
    Z: np.ndarray | None = None,
    latent_rng: np.random.Generator | None = None,
    n_probes: int = 3,
    probe_noise: float = 0.0,
    tighten: int = 0,
    frac_per_item: float = 0.0,
) -> dict:
    """Dispatcher: phi=0 -> verbatim legacy loop; phi>0 -> extended loop."""
    if phi <= 0.0:
        return sample_histories_legacy(
            items, turn, n_users, mean_hist, alpha, beta, seed,
            n_probes=n_probes, probe_noise=probe_noise,
            frac_per_item=frac_per_item,
        )
    if Z is None or latent_rng is None:
        raise ValueError("phi>0 requires Z and latent_rng")
    return sample_histories_phi(
        items, turn, n_users, mean_hist, alpha, beta, seed, phi, Z, latent_rng,
        n_probes=n_probes, probe_noise=probe_noise, tighten=tighten,
        frac_per_item=frac_per_item,
    )


# ---------------------------------------------------------------------------
# Family assembly (legacy = verbatim; phi-aware = extended)
# ---------------------------------------------------------------------------
def build_family_legacy(params: dict) -> dict:
    """Pristine port of generator.build_family: byte-identical null catalogs."""
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

    sim = sample_histories_legacy(
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

    digest = hashlib.sha256(params["catalog_id"].encode()).digest()[0]
    fold = "confirm" if (digest % 2 == 0) else "screen"

    return {
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


def build_family(params: dict) -> dict:
    """phi-aware build_family (main-grid cells).

    Uses the SAME item vector as the legacy builder for identical
    (config, seed) (the popularity-marginal hold across phi), adds the latent
    structure + tag encoding, records phi and every latent knob in
    family_params.  When params['phi'] == 0 the user_logs are draw-for-draw
    identical to build_family_legacy (the dispatcher's phi==0 branch).
    """
    phi = float(params.get("phi", 0.0))
    emit_tags = bool(params.get("emit_tags", True))
    rng = _seeded(params["seed"])
    horizon_days = params.get("horizon_days", 365)
    latent_seed = params.get("latent_sample_seed") or (
        params["seed"] * LATENT_SEED_MULT + LATENT_SEED_ADD
    )
    latent_rng = _seeded(latent_seed)

    items, cat_meta = make_items(
        rng,
        params["n_items"],
        params["n_attrs"],
        params["zipf_s"],
        forced_zipf=True,
    )
    turn = make_turnover(rng, params["n_items"], params["dynamic"], horizon_days)

    # latent structure drawn ONCE per (config, seed), shared across phi;
    # unit-normed factors keep the popularity marginal put (disclosed above)
    Z, tags_per_item, tag_names = make_latent_structure(
        latent_rng, params["n_items"], LATENT_D_C, N_DISTRACTOR_TAGS,
        norm_factors=bool(params.get("norm_factors", True)),
    )
    if emit_tags:
        for it, tags in zip(items, tags_per_item):
            it["tags"] = tags

    # effective strength: nominal phi may be capped by the marginals gate
    phi_eff = float(params.get("phi_effective", phi))
    sim = sample_histories(
        items,
        turn,
        params["n_users"],
        params["mean_hist"],
        params["alpha"],
        params["beta"],
        seed=params["seed"] + 1000,
        phi=phi_eff,
        Z=Z,
        latent_rng=latent_rng,
        n_probes=params.get("n_probes", 3),
        probe_noise=params.get("probe_noise", 0.0),
        tighten=params.get("tighten", 0),
    )

    digest = hashlib.sha256(params["catalog_id"].encode()).digest()[0]
    fold = "confirm" if (digest % 2 == 0) else "screen"

    family_params = {
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
        # ---- iteration-2 additions ----
        "phi": phi,
        "phi_effective": phi_eff,
        "phi_capped": bool(phi_eff < phi - 1e-12),
        "latent_D_c": LATENT_D_C,
        "n_distractor_tags": N_DISTRACTOR_TAGS,
        "latent_sample_seed": latent_seed,
        "tag_encoding": "lz_d present iff z_v[d] > 0",
        "tag_vocab": tag_names,
        "emit_tags": emit_tags,
        "norm_factors": bool(params.get("norm_factors", True)),
        "tighten": params.get("tighten", 0),
        "item_level_formula": (
            "s = z_v . p_{u,c} - mean_{v' in c}(z_{v'} . p_{u,c}); "
            "P(v) ~ w_v(t) * exp(beta*theta_u[cat_v]) * exp(phi_eff*s), "
            "normalized; phi_eff = phi unless capped by the marginals gate"
        ),
    }
    # The raw items carry base_weight (used by the sampler); the pool schema
    # mirrors the iteration-1 emission (no base_weight in the JSON items).
    return {
        "catalog_id": params["catalog_id"],
        "origin": "synthetic",
        "source_name": "latent_intent_model",
        "family_params": family_params,
        "items": [
            {
                "item_id": it["item_id"],
                "attrs": it["attrs"],
                "category": it["category"],
                "price_band": it["price_band"],
                "attribute_diversity": it["attribute_diversity"],
            }
            | ({"tags": tags_per_item[i]} if emit_tags else {})
            for i, it in enumerate(items)
        ],
        "user_logs": sim["user_logs"],
        "forced_choice_probes": sim["probes"],
        "fold": fold,
        "partition_meta": {
            "partition": "generation",
            "generator": "latent_intent_item_appeal_phi",
            "n_items_total": params["n_items"],
        },
    }


# ---------------------------------------------------------------------------
# Marginals-match gate (per (config, seed, phi) against phi=0)
# ---------------------------------------------------------------------------
def screen_shares(cat: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-item empirical purchase shares on the SCREEN half + HHI/norm-ent.
    cat: pool-schema catalog dict.  Screen half = positions 1..floor(0.8L)
    per user (the same split the eval protocol uses)."""
    counts = np.zeros(len(cat["items"]), dtype=np.float64)
    for u in cat["user_logs"]:
        L = len(u["ordered_history"])
        s = max(0, int(L * 0.8))
        if s == 0:
            continue
        for j in range(s):
            idx = next((i for i, it in enumerate(cat["items"])
                        if it["item_id"] == u["ordered_history"][j]), -1)
            if idx >= 0:
                counts[idx] += 1.0
    total = counts.sum()
    shares = counts / total if total > 0 else np.zeros_like(counts)
    nz = shares[shares > 0]
    hhi = float((shares ** 2).sum())
    ent = float(-(nz * np.log2(nz)).sum()) if len(nz) else 0.0
    norm_ent = ent / np.log2(max(len(shares), 2.0))
    return shares, (hhi, norm_ent)


def marginals_match(cat_phi: dict, cat_phi0: dict,
                    tol: dict | None = None) -> dict:
    """Compare the phi catalog's screen-half marginal against the phi=0
    catalog's.  Returns the gate metrics + pass flag (KS <= 0.05,
    max_abs_dev <= 0.02, |dHHI| <= 0.02, |dNormEntropy| <= 0.03)."""
    tol = tol or GATE_TOL
    s1, (hhi1, ne1) = screen_shares(cat_phi)
    s0, (hhi0, ne0) = screen_shares(cat_phi0)
    # both align by item order (identical item vectors)
    n = max(len(s1), len(s0))
    pad1 = np.pad(s1, (0, n - len(s1)))
    pad0 = np.pad(s0, (0, n - len(s0)))
    ks = float(np.max(np.abs(np.cumsum(pad1) - np.cumsum(pad0))))
    max_abs = float(np.max(np.abs(pad1 - pad0)))
    d_hhi = hhi1 - hhi0
    d_ne = ne1 - ne0
    passed = bool(
        ks <= tol["ks"] and max_abs <= tol["max_abs_dev"]
        and abs(d_hhi) <= tol["d_hhi"] and abs(d_ne) <= tol["d_norm_entropy"]
    )
    return {
        "gate_KS": ks,
        "gate_max_abs_dev": max_abs,
        "dHHI": d_hhi,
        "dNormEntropy": d_ne,
        "hhi_phi": hhi1,
        "norm_entropy_phi": ne1,
        "hhi_phi0": hhi0,
        "norm_entropy_phi0": ne0,
        "pass": passed,
    }


def _gate_pass_fast(cat: dict, phi0_catalog: dict) -> tuple[bool, dict]:
    rep = marginals_match(cat, phi0_catalog)
    return bool(rep["pass"]), rep


def generate_cell_with_gate(params: dict, phi0_catalog: dict,
                            max_cap_steps: int = 14) -> tuple[dict, dict]:
    """Generate one phi>0 catalog that PASSES the marginals-match gate.

    Ladder (measured; disclosed in the module docstring):
      1. sample at the nominal phi (tighten=0); pass -> done.
      2. resample with T1 (per-user per-category phi=0 mass anchoring);
         pass -> done.
      3. EFFECTIVE-STRENGTH CAPPING (plan fallback-1): binary-search the
         largest phi_eff in (0, phi] whose catalog passes the gate, sample at
         phi_eff, and record phi_effective + phi_capped in family_params and
         in the returned gate report.

    Returns (catalog, gate_report) with gate_report['pass'] always True.
    """
    phi_nom = float(params["phi"])
    for tighten in (0, 1):
        p = dict(params)
        p["tighten"] = tighten
        p.pop("phi_effective", None)
        cat = build_family(p)
        passed, report = _gate_pass_fast(cat, phi0_catalog)
        report["attempt"] = 0 if tighten == 0 else 1
        report["tighten"] = tighten
        report["phi_effective"] = phi_nom
        if passed:
            return cat, report
        logger.warning(
            f"marginals gate FAIL at phi={phi_nom} (tighten={tighten}): "
            f"KS={report['gate_KS']:.4f} maxabs={report['gate_max_abs_dev']:.4f} "
            f"dHHI={report['dHHI']:.5f} dNE={report['dNormEntropy']:.5f}")

    # ---- effective-strength capping (binary search on phi_eff) ----
    lo, hi = 0.0, phi_nom
    best_cat = None
    best_rep = None
    for _step in range(max_cap_steps + 1):
        mid = (lo + hi) / 2.0
        p = dict(params)
        p["phi_effective"] = mid
        p["tighten"] = 1
        cat = build_family(p)
        passed, report = _gate_pass_fast(cat, phi0_catalog)
        report["phi_effective"] = mid
        if passed:
            best_cat = cat
            best_rep = report
            lo = mid
            if hi - lo < 1e-4 or mid >= phi_nom - 1e-9:
                break
        else:
            hi = mid
    if best_cat is None:
        # phi_eff -> 0 reproduces the phi0 marginal (identical draws only at
        # phi_eff=0 the sampler is the legacy loop -> shares match exactly),
        # so a passing phi_eff always exists; guard defensively.
        p = dict(params)
        p["phi_effective"] = 0.0
        best_cat = build_family(p)
        best_rep = marginals_match(best_cat, phi0_catalog)
        best_rep["phi_effective"] = 0.0
    best_rep["attempt"] = 2
    best_rep["tighten"] = 1
    best_rep["capped"] = True
    logger.warning(
        f"gate capped: phi {phi_nom} -> phi_eff={best_rep['phi_effective']:.4f} "
        f"for {params['catalog_id']} "
        f"(KS={best_rep['gate_KS']:.4f} maxabs={best_rep['gate_max_abs_dev']:.4f})")
    return best_cat, best_rep


# ---------------------------------------------------------------------------
# Null-replication family grid (VERBATIM port of generator.family_grid)
# ---------------------------------------------------------------------------
def family_grid() -> list[dict]:
    """Return the iteration-1 DATASET family_grid param dicts EXACTLY as
    generator.py defines them (null-replication targets)."""
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

    add("v20", {"n_items": 20, "n_users": 600}, [0])
    add("v50", {"n_items": 50, "n_users": 600}, [0])
    add("v200", {"n_items": 200, "n_users": 1200}, [0, 1, 2, 3])
    add("v1000", {"n_items": 1000, "n_users": 2500}, [0, 1, 2])
    add("zipf05", {"zipf_s": 0.5}, [0])
    add("zipf15", {"zipf_s": 1.5}, [0])
    add("zipf20", {"zipf_s": 2.0}, [0, 1])
    add("attr_many", {"n_attrs": 30}, [0])
    add("attr_hetero", {"n_attrs": 90, "alpha": 1.2}, [0])
    add("stick_low", {"alpha": 1.5}, [0])
    add("hist_short", {"mean_hist": 8}, [0])
    add("hist_long", {"mean_hist": 24}, [0])
    add("tstat", {"dynamic": False}, [0])
    return cells


def main() -> None:
    """Standalone: regenerate null-replication catalogs (byte-identity demo)."""
    import sys
    from pathlib import Path

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="{time:HH:mm:ss}|{level:<7}|{message}")
    outdir = Path(__file__).resolve().parent / "data" / "null_replication"
    outdir.mkdir(parents=True, exist_ok=True)
    for p in family_grid():
        cat = build_family_legacy(p)
        (outdir / f"{cat['catalog_id']}.json").write_text(json.dumps(cat))
        logger.info(f"wrote {cat['catalog_id']}")
    logger.info("null-replication families regenerated (legacy path)")


if __name__ == "__main__":
    main()