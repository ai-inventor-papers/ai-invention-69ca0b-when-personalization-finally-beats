#!/usr/bin/env python3
"""Synthetic small-e-commerce catalog generator (latent-intent model).

For each (config, seed) a catalog is generated deterministically with
np.random.default_rng(seed). Every user draws a latent preference vector
beta_u over item categories (Dirichlet; concentration controls
heterogeneity/stickiness) plus a loyalty epsilon_u; each next purchase is
drawn i.i.d. given beta_u from

    P(i | beta_u) = (1 - eps_u) * w_i / W        (crowd / popularity arm)
                  +  eps_u * beta_u[c_i] * w_i / Z_beta   (preference arm, stickiness)

Timestamps are running order index + tiny uniform jitter so temporal
splits are well-defined and ordinal.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

# History lengths k evaluated in the sweep.
K_VALUES: tuple[int, ...] = (0, 1, 2, 3, 5, 8)
# Top-K_cut rankings stored per (user, k, heuristic).
K_CUT: int = 50
# Minimum users populating a (catalog, k) cell for it to be "supported".
MIN_EXAMPLES: int = 50
# Default users per generated catalog (adaptive; CLI-overridable).  500 keeps
# every cell (>= 50 supported users) well above the support threshold while
# bounding the JSON output size of the 192-catalog grid.
N_USERS_DEFAULT: int = 500


class Catalog:
    """Canonical in-memory catalog.

    Arrays are indexed by item-index (0..|V|-1) / user-index (0..n_users-1).
    users entries: dict(user_id: str, item_seq: np.ndarray[int], t_seq:
    np.ndarray[float], beta: np.ndarray[float] (n_cat,), epsilon: float,
    argmax_cat: int).  beta/epsilon/argmax_cat are None for real catalogs.
    """

    __slots__ = (
        "catalog_id", "origin", "source_name", "source_path", "fold",
        "family_params", "rng_seed", "item_ids", "categories", "price_bands",
        "tag_matrix", "popularity_weight", "n_cat", "n_price", "n_tags",
        "users", "notes",
    )

    def __init__(
        self,
        catalog_id: str,
        origin: str,
        source_name: str,
        fold: str,
        family_params: dict,
        rng_seed: int | None,
        item_ids: np.ndarray,
        categories: np.ndarray,
        price_bands: np.ndarray,
        tag_matrix: np.ndarray,
        popularity_weight: np.ndarray | None,
        users: list[dict],
        source_path: str | None = None,
        notes: list[str] | None = None,
    ) -> None:
        self.catalog_id = catalog_id
        self.origin = origin
        self.source_name = source_name
        self.source_path = source_path
        self.fold = fold
        self.family_params = family_params
        self.rng_seed = rng_seed
        self.item_ids = np.asarray(item_ids, dtype=object)
        self.categories = np.asarray(categories, dtype=np.int32)
        self.price_bands = np.asarray(price_bands, dtype=np.int32)
        self.tag_matrix = np.asarray(tag_matrix, dtype=bool)
        self.popularity_weight = (
            None if popularity_weight is None else np.asarray(popularity_weight, dtype=np.float64)
        )
        self.users = users
        self.n_cat = int(self.categories.max()) + 1 if len(self.categories) else 0
        self.n_price = int(self.price_bands.max()) + 1 if len(self.price_bands) else 0
        self.n_tags = int(self.tag_matrix.shape[1]) if self.tag_matrix.ndim == 2 else 0
        self.notes = notes or []

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_users(self) -> int:
        return len(self.users)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def make_config(
    n_items: int,
    entropy_level: str,
    zipf_alpha: float,
    epsilon: float,
    mean_history: int,
    seed: int,
    n_users: int = N_USERS_DEFAULT,
) -> dict:
    """One generator configuration (one catalog)."""
    assert entropy_level in ("LOW", "HIGH")
    assert n_items in (20, 100, 500, 1000)
    assert zipf_alpha in (0.5, 1.5)
    assert epsilon in (0.2, 0.6)
    assert mean_history in (4, 10)
    n_cat = 2 if entropy_level == "LOW" else 20
    n_price = 2 if entropy_level == "LOW" else 4
    n_tags = 8 if entropy_level == "LOW" else 64
    # Dirichlet concentration: sticky users => peaked, heterogeneous beta;
    # promiscuous users => concentrated near the crowd's category mix.
    conc = 0.4 if epsilon == 0.6 else 4.0
    return {
        "n_items": n_items,
        "entropy_level": entropy_level,
        "n_cat": n_cat,
        "n_price_bands": n_price,
        "n_tags": n_tags,
        "zipf_alpha": zipf_alpha,
        "epsilon": epsilon,
        "dirichlet_concentration": conc,
        "mean_history": mean_history,
        "n_users": n_users,
        "seed": seed,
        "catalog_id": f"syn_V{n_items}_E{entropy_level}_z{zipf_alpha}_e{epsilon}_m{mean_history}_s{seed}",
    }


def grid_specs(n_users: int = N_USERS_DEFAULT) -> list[dict]:
    """The full 192-catalog factorial grid."""
    specs: list[dict] = []
    for n_items in (20, 100, 500, 1000):
        for entropy_level in ("LOW", "HIGH"):
            for zipf_alpha in (0.5, 1.5):
                for epsilon in (0.2, 0.6):
                    for mean_history in (4, 10):
                        for seed in (0, 1, 2):
                            specs.append(
                                make_config(n_items, entropy_level, zipf_alpha, epsilon, mean_history, seed, n_users)
                            )
    return specs


def smoke_specs(n_users: int = N_USERS_DEFAULT) -> list[dict]:
    """Smoke test: one tiny catalog, |V|=20, LOW entropy, sticky, mu=10."""
    return [make_config(20, "LOW", 1.5, 0.6, 10, 0, n_users)]


def corner_specs(n_users: int = N_USERS_DEFAULT) -> list[dict]:
    """~10 catalogs spanning the grid corners for staged scaling."""
    raw = [
        (20, "LOW", 0.5, 0.2, 4, 0),
        (20, "HIGH", 1.5, 0.6, 10, 1),
        (100, "LOW", 1.5, 0.6, 10, 0),
        (100, "HIGH", 0.5, 0.2, 4, 2),
        (500, "LOW", 0.5, 0.6, 10, 1),
        (500, "HIGH", 1.5, 0.2, 4, 0),
        (1000, "LOW", 1.5, 0.6, 10, 0),
        (1000, "HIGH", 0.5, 0.2, 4, 2),
        (1000, "LOW", 0.5, 0.2, 4, 1),
        (1000, "HIGH", 1.5, 0.6, 10, 2),
    ]
    return [make_config(n, e, z, eps, m, s, n_users) for (n, e, z, eps, m, s) in raw]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_catalog(cfg: dict) -> Catalog:
    """Deterministically generate one catalog from a config dict."""
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["n_items"]
    n_cat = cfg["n_cat"]
    n_price = cfg["n_price_bands"]
    n_tags = cfg["n_tags"]
    alpha_w = cfg["zipf_alpha"]
    eps = cfg["epsilon"]
    conc = cfg["dirichlet_concentration"]
    mu = cfg["mean_history"]
    n_users = cfg["n_users"]

    # --- category / price prevalence over the item population ---
    q_cat = rng.dirichlet(np.ones(n_cat))
    if n_price == 2:
        q_price = np.array([0.6, 0.4])
    else:
        q_price = np.array([0.4, 0.3, 0.2, 0.1])
    q_price = q_price / q_price.sum()

    # --- items ---
    categories = rng.choice(n_cat, size=n, p=q_cat).astype(np.int32)
    price_bands = rng.choice(n_price, size=n, p=q_price).astype(np.int32)
    tag_matrix = np.zeros((n, n_tags), dtype=bool)
    for i in range(n):
        if cfg["entropy_level"] == "LOW":
            n_item_tags = int(rng.integers(1, 3))  # 1-2 tags
        else:
            n_item_tags = int(rng.integers(2, 7))  # 2-6 tags
        chosen = rng.choice(n_tags, size=n_item_tags, replace=False)
        tag_matrix[i, chosen] = True

    # Zipf popularity weights: w_i ∝ (rank+1)^(-alpha)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    w = ranks ** (-alpha_w)
    w = w / w.sum()

    # Per-category mass used for the preference-arm normalizer Z.
    w_by_cat = np.zeros(n_cat)
    for c in range(n_cat):
        w_by_cat[c] = w[categories == c].sum()
    w_by_cat = np.maximum(w_by_cat, 1e-12)

    # --- users ---
    # beta_u ~ Dirichlet(conc * q_cat): sticky (small conc) => peaked,
    # heterogeneous preferences; promiscuous (large conc) => homogeneous.
    alphas = np.maximum(conc * q_cat, 1e-6)
    item_ids = np.array([f"i{i}" for i in range(n)], dtype=object)
    users: list[dict] = []
    rng_user = np.random.default_rng(cfg["seed"] * 7919 + 13)
    for u in range(n_users):
        beta = rng_user.dirichlet(alphas)
        argmax_cat = int(np.argmax(beta))
        # Z_beta = sum_j beta[c_j] * w_j = sum_c beta[c] * w_by_cat[c]
        Z = float(np.dot(beta, w_by_cat))
        # q_i = (1-eps)*w_i/W + eps*beta[c_i]*w_i/Z
        q = (1.0 - eps) * w + eps * beta[categories] * w / max(Z, 1e-12)
        q = q / q.sum()
        # history length: 1 + Geometric (mean mu), capped at 40
        L = int(min(1 + rng_user.geometric(1.0 / mu), 40))
        # all draws i.i.d. given beta (memoryless mixture conditional on latent)
        draws = np.searchsorted(np.cumsum(q), rng_user.uniform(size=L))
        draws = np.clip(draws, 0, n - 1)
        t_seq = np.arange(L, dtype=np.float64) + rng_user.uniform(size=L)
        users.append(
            {
                "user_id": f"u{u}",
                "item_seq": draws.astype(np.int32),
                "t_seq": t_seq,
                "beta": beta.astype(np.float64),
                "epsilon": eps,
                "argmax_cat": argmax_cat,
            }
        )

    return Catalog(
        catalog_id=cfg["catalog_id"],
        origin="synthetic",
        source_name="inline_fallback",
        fold="screen",
        family_params=dict(cfg),
        rng_seed=cfg["seed"],
        item_ids=item_ids,
        categories=categories,
        price_bands=price_bands,
        tag_matrix=tag_matrix,
        popularity_weight=w,
        users=users,
    )


# ---------------------------------------------------------------------------
# Mutual-information ceiling (deterministic Monte-Carlo, disclosed)
# ---------------------------------------------------------------------------

_MC_NEXT_SAMPLES = 5000      # prior beta draws for H(next)
_MC_HIST_SAMPLES = 160       # drawn histories per k
_MC_POST_SAMPLES = 120       # posterior beta draws per history


def _q_vector(beta: np.ndarray, w: np.ndarray, categories: np.ndarray,
              w_by_cat: np.ndarray, eps: float) -> np.ndarray:
    """Item draw distribution P(next=i | beta) under the generator's model."""
    Z = float(np.dot(beta, w_by_cat))
    q = (1.0 - eps) * w + eps * beta[categories] * w / max(Z, 1e-12)
    return q / q.sum()


def mi_ceiling_mc(cfg: dict, k_values: tuple[int, ...] = K_VALUES) -> dict:
    """Deterministic MC estimate of I(history_k ; next item) per k.

    Uses the conjugacy fact that, given a length-k history, the posterior of
    beta_u is Dirichlet(conc*q_cat + category_counts(history)) (a mild
    approximation: the popularity arm's draws also carry weak category
    evidence).  Fully seeded via cfg['seed']; identical on every run.
    """
    rng = np.random.default_rng(cfg["seed"] * 104729 + 7)
    n = cfg["n_items"]
    n_cat = cfg["n_cat"]
    q_cat = rng.dirichlet(np.ones(n_cat))
    w = (np.arange(1, n + 1) ** (-cfg["zipf_alpha"]))
    w = w / w.sum()
    categories = rng.choice(n_cat, size=n, p=q_cat)
    w_by_cat = np.array([w[categories == c].sum() for c in range(n_cat)])
    w_by_cat = np.maximum(w_by_cat, 1e-12)
    alphas = np.maximum(cfg["dirichlet_concentration"] * q_cat, 1e-6)
    eps = cfg["epsilon"]

    # H(next): marginal over (beta, next)
    logp = np.zeros(n)
    for _ in range(_MC_NEXT_SAMPLES):
        beta = rng.dirichlet(alphas)
        q = _q_vector(beta, w, categories, w_by_cat, eps)
        logp += q
    p = logp / _MC_NEXT_SAMPLES
    p = np.clip(p, 1e-15, None)
    H_next = float(-(p * np.log2(p)).sum())

    out: dict[str, float] = {}
    for k in k_values:
        if k == 0:
            # No history => conditional reduces to the marginal.
            out[f"k{k}"] = 0.0
            continue
        h_cond = 0.0
        for _ in range(_MC_HIST_SAMPLES):
            beta0 = rng.dirichlet(alphas)
            # generate a length-k history from the full model
            Z0 = float(np.dot(beta0, w_by_cat))
            q0 = (1.0 - eps) * w + eps * beta0[categories] * w / max(Z0, 1e-12)
            q0 = q0 / q0.sum()
            history = np.clip(
                np.searchsorted(np.cumsum(q0), rng.uniform(size=k)), 0, n - 1
            )
            counts = np.bincount(categories[history], minlength=n_cat)
            post_alphas = alphas + counts
            acc = np.zeros(n)
            for _b in range(_MC_POST_SAMPLES):
                beta_p = rng.dirichlet(post_alphas)
                acc += _q_vector(beta_p, w, categories, w_by_cat, eps)
            p_cond = np.clip(acc / _MC_POST_SAMPLES, 1e-15, None)
            h_cond += -float((p_cond * np.log2(p_cond)).sum())
        out[f"k{k}"] = float(H_next - h_cond / _MC_HIST_SAMPLES)
    out["method"] = (
        f"deterministic MC: H(next)~{_MC_NEXT_SAMPLES} prior betas; per k: "
        f"{_MC_HIST_SAMPLES} histories x {_MC_POST_SAMPLES} posterior betas; "
        "Dirichlet-conjugate posterior approximation; seed-derived RNG"
    )
    out["H_next"] = H_next
    return out


if __name__ == "__main__":  # pragma: no cover
    from loguru import logger as _lg  # noqa
    _lg.remove()
    _lg.add(lambda m: print(m, end=""), level="INFO")
    cat = generate_catalog(smoke_specs(60)[0])
    _lg.info(f"catalog {cat.catalog_id}: {cat.n_items} items, {cat.n_users} users")
    _lg.info(f"categories: {cat.n_cat}, price bands: {cat.n_price}, tags: {cat.n_tags}")
    _lg.info(f"mean history: {np.mean([len(u['item_seq']) for u in cat.users]):.2f}")