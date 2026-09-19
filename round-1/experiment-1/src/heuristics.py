#!/usr/bin/env python3
"""Cold-start heuristic library (16 disclosed variants).

Every heuristic maps a (user, k) evaluation cell to a score per candidate
item (higher = better).  Scores are computed per cell over ALL valid users
with numpy vectorization, then ranked in eval_metrics (uniform filtering of
already-purchased context items for k>=1 happens there).

Families
--------
POPULARITY: pop_global, recency_pop_{0.1,0.5,1.0}, pop_category, pop_price
CONTENT-kNN: content_knn_{1,3,5}, last_item_nbhd, pop_scaled_content
ASSOCIATION: co_purchase
HYBRID:      lambda_hybrid_{0.25,0.5,0.75}
ELICITATION: active_elic2  (k=0 only)

All fit parameters use ONLY the screen half of events (per-catalog), and
for each evaluation cell (u,k) the user's own events at positions 1..k+1
(context + ground truth) are excluded from any count ("L1O-pop", leak-free
temporal leave-one-out).  This is done inside build_* functions via the
'own_counts' inputs; every estimate used at prediction time is therefore
computed from events STRICTLY before the prediction position.
"""

from __future__ import annotations

import numpy as np

K_VALUES = (0, 1, 2, 3, 5, 8)

# ---------------------------------------------------------------------------
# Heuristic inventory (disclosed a priori, with prior-art landscape)
# ---------------------------------------------------------------------------

HEURISTIC_SPECS: list[dict] = [
    {"name": "pop_global", "family": "popularity", "params": {"L1O": "own 1..k+1 excluded"},
     "prior_art": "Ji et al., A Re-visit of the Popularity Baseline, SIGIR 2020",
     "desc": "score = screen purchase count of the item (L1O-pop)."},
    {"name": "recency_pop_0.1", "family": "popularity", "params": {"window_quantile": 0.1, "L1O": True},
     "prior_art": "Ji et al., SIGIR 2020 (recency-decayed popularity)",
     "desc": "score = sum over screen events exp(-lambda*(T_now - t)); half-life = 0.1 x screen span."},
    {"name": "recency_pop_0.5", "family": "popularity", "params": {"window_quantile": 0.5, "L1O": True},
     "prior_art": "Ji et al., SIGIR 2020 (recency-decayed popularity)",
     "desc": "score = sum over screen events exp(-lambda*(T_now - t)); half-life = 0.5 x screen span."},
    {"name": "recency_pop_1.0", "family": "popularity", "params": {"window_quantile": 1.0, "L1O": True},
     "prior_art": "Ji et al., SIGIR 2020 (all-time popularity)",
     "desc": "score = sum over screen events exp(-lambda*(T_now - t)); half-life = 1.0 x screen span (all-time)."},
    {"name": "pop_category", "family": "popularity", "params": {"gamma": 1.0, "L1O": True},
     "prior_art": "banded popularity (P3-style), cf. B2P, Chaimalas et al., RecSys 2023",
     "desc": "score = P(cat|user)^gamma * count(item); P(cat|user) from context, global shares at k=0."},
    {"name": "pop_price", "family": "popularity", "params": {"gamma": 1.0, "L1O": True},
     "prior_art": "banded popularity (P3-style), cf. B2P, RecSys 2023",
     "desc": "score = P(price_band|user)^gamma * count(item); global shares at k=0."},
    {"name": "content_knn_1", "family": "content_knn", "params": {"neighborhood": 1},
     "prior_art": "content-based cold-start textbook remedy (Basu et al. 1998; Lops et al. 2011)",
     "desc": "profile = mean of context item vectors; score = cosine(profile, item); 1-NN item expansion."},
    {"name": "content_knn_3", "family": "content_knn", "params": {"neighborhood": 3},
     "prior_art": "content-based cold-start textbook remedy",
     "desc": "as content_knn_1 but each context item expanded by its 3 nearest content neighbours."},
    {"name": "content_knn_5", "family": "content_knn", "params": {"neighborhood": 5},
     "prior_art": "content-based cold-start textbook remedy",
     "desc": "as content_knn_1 but each context item expanded by its 5 nearest content neighbours."},
    {"name": "last_item_nbhd", "family": "content_knn", "params": {"profile_items": 1},
     "prior_art": "content-based cold-start textbook remedy",
     "desc": "score = cosine to the LAST purchased item's feature vector only."},
    {"name": "pop_scaled_content", "family": "content_knn", "params": {"beta": 0.5, "L1O": True},
     "prior_art": "popularity-scaled content (standard content-based remedy)",
     "desc": "score = content_cosine * popularity^0.5 (L1O normalized popularity)."},
    {"name": "co_purchase", "family": "association", "params": {"smoothing": 1e-6, "L1O": True},
     "prior_art": "association rules: Agrawal & Srikant, VLDB 1994",
     "desc": "score = max over context h of lift(item,h)=P(item|h)/P(item); L1O co-occurrence."},
    {"name": "lambda_hybrid_0.25", "family": "hybrid", "params": {"lam": 0.25},
     "prior_art": "B2P P3: lambda*pop+(1-lambda)*content, Chaimalas et al., RecSys 2023",
     "desc": "score = lam*norm_content + (1-lam)*norm_pop; k=0 degenerates to popularity."},
    {"name": "lambda_hybrid_0.5", "family": "hybrid", "params": {"lam": 0.5},
     "prior_art": "B2P P3, Chaimalas et al., RecSys 2023",
     "desc": "score = lam*norm_content + (1-lam)*norm_pop."},
    {"name": "lambda_hybrid_0.75", "family": "hybrid", "params": {"lam": 0.75},
     "prior_art": "B2P P3, Chaimalas et al., RecSys 2023",
     "desc": "score = lam*norm_content + (1-lam)*norm_pop."},
    {"name": "active_elic2", "family": "elicitation", "params": {"n_questions": 2, "answer_noise": 0.2, "max_pairs": 20000},
     "prior_art": "Bayesian active/adaptive elicitation: Golovin & Krause, ICML 2010 (adaptive submodularity); no canonical cold-start paper = gap",
     "desc": "asks 2 forced-choice questions selected to maximize expected entropy reduction of the category posterior; k=0 only."},
]

HEURISTIC_NAMES: list[str] = [s["name"] for s in HEURISTIC_SPECS]
POPULARITY_FAMILY_NAMES: tuple[str, ...] = (
    "pop_global", "recency_pop_0.1", "recency_pop_0.5", "recency_pop_1.0",
    "pop_category", "pop_price",
)


def heuristic_defined_at_k(name: str, k: int) -> bool:
    """Whether a heuristic yields a score at history length k."""
    if name == "active_elic2":
        return k == 0
    if name in ("content_knn_1", "content_knn_3", "content_knn_5",
                "last_item_nbhd", "pop_scaled_content", "co_purchase"):
        return k >= 1
    return True


def _l1o_counts(global_counts: np.ndarray, own_ones: np.ndarray) -> np.ndarray:
    """global_counts: (|V|,). own_ones: (n_valid, |V|) int one-hot counts of
    the user's own events at positions 1..k+1.  Returns (n_valid, |V|) float64
    counts with the user's own events removed (L1O-pop)."""
    return np.maximum(global_counts[None, :].astype(np.float64) - own_ones, 0.0)


# ---------------------------------------------------------------------------
# Score builders.  Each returns (n_valid, |V|) float64 or None if NA.
# Common per-cell inputs (see eval_metrics.build_cell_inputs):
#   S   : (n_valid, |V|) float64 -- L1O popularity counts for the cell
#   Scat: (n_valid, n_cat)  -- L1O category share for the cell
#   Spr : (n_valid, n_price) -- L1O price-band share for the cell
#   ctx : (n_valid, k) int32 -- context item indices (k=0: empty)
#   last: (n_valid,) int32   -- last context item index (k>=1)
#   prof: (n_valid, n_feat)  -- content profile (mean of context item feats)
#   F   : (|V|, n_feat)      -- L2-normalized item feature matrix
#   sim : (n_valid, |V|)     -- cosine(profile, item)
# ---------------------------------------------------------------------------


def build_pop_global(alpha: float, S: np.ndarray, **_: object) -> np.ndarray:
    return S


def build_recency(name: str, R: np.ndarray, **_: object) -> np.ndarray:
    """R: (n_valid, |V|) L1O recency-decayed counts precomputed by eval_metrics."""
    _ = name
    return R


def build_pop_banded(S: np.ndarray, share: np.ndarray, band_of_item: np.ndarray) -> np.ndarray:
    """score(item) = P(band|user)^gamma * count(item) with gamma=1."""
    pw = share ** 1.0  # (n_valid, n_bands)
    return pw[:, band_of_item] * S


def build_pop_category(S: np.ndarray, Scat: np.ndarray, band_of_item: np.ndarray, **_: object) -> np.ndarray:
    return build_pop_banded(S, Scat, band_of_item)


def build_pop_price(S: np.ndarray, Spr: np.ndarray, band_of_item: np.ndarray, **_: object) -> np.ndarray:
    return build_pop_banded(S, Spr, band_of_item)


def build_last_item_nbhd(sim: np.ndarray, _last: np.ndarray, **_: object) -> np.ndarray:
    return sim


def build_content_knn(neighborhood: int, sim: np.ndarray, prof_knn: np.ndarray, F: np.ndarray, **_: object) -> np.ndarray:
    """sim: (n_valid, |V|) cosine from the plain (no-expansion) profile.
    prof_knn: (n_valid, n_feat) profile with neighborhood expansion."""
    _ = (neighborhood, sim)
    return prof_knn @ F.T


def build_pop_scaled_content(beta: float, sim: np.ndarray, S: np.ndarray, **_: object) -> np.ndarray:
    """score = content_cosine * popularity^beta (L1O popularity, min-max norm)."""
    eps = 1e-12
    lo = S.min(axis=1, keepdims=True)
    hi = S.max(axis=1, keepdims=True)
    norm_pop = (S - lo) / np.maximum(hi - lo, eps)
    return sim * np.power(norm_pop + eps, beta)


def build_co_purchase(lift: np.ndarray, **_: object) -> np.ndarray:
    """lift: (n_valid, |V|) precomputed L1O max-lift score."""
    return lift


def build_lambda_hybrid(lam: float, sim: np.ndarray, S: np.ndarray, **_: object) -> np.ndarray:
    """score = lam*norm_content + (1-lam)*norm_pop (per-user min-max norm)."""
    eps = 1e-12
    lo_s = S.min(axis=1, keepdims=True)
    hi_s = S.max(axis=1, keepdims=True)
    norm_pop = (S - lo_s) / np.maximum(hi_s - lo_s, eps)
    if sim is None:  # k=0: no content signal -> degenerate to popularity
        return (1.0 - lam) * norm_pop
    lo_c = sim.min(axis=1, keepdims=True)
    hi_c = sim.max(axis=1, keepdims=True)
    norm_c = (sim - lo_c) / np.maximum(hi_c - lo_c, eps)
    return lam * norm_c + (1.0 - lam) * norm_pop