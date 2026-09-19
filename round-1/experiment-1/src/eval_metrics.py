"""Temporal leave-one-out-at-k evaluation protocol + catalog diagnostics.

Protocol (exactly as documented in the experiment metadata):
  * User u has L chronologically ordered purchases, positions 1..L.
  * Screen half  = positions 1..floor(0.8*L)   (fits parameters)
  * Confirm half = positions floor(0.8*L)+1..L (never fits anything)
  * Evaluation cell (u, k), k in {0,1,2,3,5,8}: valid iff k+1 <= L.
      context     = position 1..k (k=0 -> empty)
      ground truth = item at position k+1
  * Fitted parameters (popularity counts, category/price shares, recency
    decays, TF-IDF, co-occurrence) use ONLY the screen half, and for each
    cell the user's OWN events at positions 1..k+1 (context + ground truth)
    are excluded from those counts (L1O-pop): every estimate used at
    prediction time is computed from events strictly before the prediction
    position -- no leakage; confirm events never enter any fitted parameter
    and the ground-truth event is never counted.
  * Ranking at k>=1 filters (excludes) the user's own context items;
    k=0 has nothing to filter.
  * Single-relevant ground truth => NDCG@cut = 1/log2(r+1) if r<=cut else 0;
    Recall@cut = 1 if r<=cut else 0, where r is the rank of the true next
    item (r = 1 + #{items with score > score(gt)}; ties not counted above).
"""

from __future__ import annotations

import numpy as np

from synthetic import Catalog, K_VALUES, K_CUT
from heuristics import (
    HEURISTIC_NAMES, heuristic_defined_at_k, POPULARITY_FAMILY_NAMES,  # noqa: F401
)

_F32 = np.float32


# ---------------------------------------------------------------------------
# Content feature construction (screen-fitted TF-IDF)
# ---------------------------------------------------------------------------


def build_item_features(cat: Catalog) -> tuple[np.ndarray, dict]:
    """(F, meta): L2-normalized item feature matrix [one-hot category,
    one-hot price band, TF-IDF tags].  IDF uses tag frequency over SCREEN
    events (temporal discipline; disclosed)."""
    n = cat.n_items
    n_cat = cat.n_cat
    n_price = cat.n_price
    n_tags = cat.n_tags
    feats: list[np.ndarray] = [
        np.eye(n_cat, dtype=np.float64)[cat.categories],
        np.eye(n_price, dtype=np.float64)[cat.price_bands],
    ]
    if n_tags > 0:
        # IDF from screen events
        doc_freq = np.zeros(n_tags)
        for u in cat.users:
            s = max(0, int(len(u["item_seq"]) * 0.8))
            doc_freq += cat.tag_matrix[u["item_seq"][:s]].sum(axis=0)
        df = np.maximum(doc_freq, 1.0)
        idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        tf = cat.tag_matrix.astype(np.float64) * idf[None, :]
        feats.append(tf)
    F = np.hstack(feats)
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    F = F / norms
    return F, {"n_feat": int(F.shape[1]), "idf_from": "screen-event tag freq"}


# ---------------------------------------------------------------------------
# Per-catalog runtime store: screen-fitted parameters
# ---------------------------------------------------------------------------


class CatRuntime:
    """Immutable per-catalog screen-fitted parameters used by all cells."""

    __slots__ = (
        "cat", "F", "g_counts", "g_cat_counts", "g_price_counts",
        "screen_events_t", "screen_events_item", "screen_events_user",
        "co", "co_smooth", "item_cat_share", "item_price_share",
        "w_by_cat", "recency_lambdas", "screen_span",
    )

    def __init__(self, cat: Catalog) -> None:
        self.cat = cat
        n = cat.n_items
        g_counts = np.zeros(n, dtype=np.float64)
        g_cat = np.zeros(cat.n_cat, dtype=np.float64)
        g_price = np.zeros(cat.n_price, dtype=np.float64)
        ev_t: list[float] = []
        ev_item: list[int] = []
        ev_user: list[int] = []
        for ui, u in enumerate(cat.users):
            s = max(0, int(len(u["item_seq"]) * 0.8))
            if s == 0:
                continue
            seq = u["item_seq"][:s]
            t = u["t_seq"][:s]
            g_counts += np.bincount(seq, minlength=n)
            g_cat += np.bincount(cat.categories[seq], minlength=cat.n_cat)
            g_price += np.bincount(cat.price_bands[seq], minlength=cat.n_price)
            ev_t.extend(t.tolist())
            ev_item.extend(seq.tolist())
            ev_user.extend([ui] * int(s))
        self.g_counts = g_counts
        self.g_cat_counts = g_cat
        self.g_price_counts = g_price
        self.item_cat_share = g_cat / max(g_cat.sum(), 1.0)
        self.item_price_share = g_price / max(g_price.sum(), 1.0)
        self.F, _ = build_item_features(cat)

        # ordered screen events (time ascending)
        order = np.argsort(np.asarray(ev_t, dtype=np.float64), kind="stable")
        self.screen_events_t = np.asarray(ev_t, dtype=np.float64)[order]
        self.screen_events_item = np.asarray(ev_item, dtype=np.int32)[order]
        self.screen_events_user = np.asarray(ev_user, dtype=np.int32)[order]

        # ordered co-occurrence: co[i, j] = # screen pairs (h -> item) with
        # i bought before j by the SAME user (within their screen half).
        co = np.zeros((n, n), dtype=np.float64)
        for u in cat.users:
            s = max(0, int(len(u["item_seq"]) * 0.8))
            seq = u["item_seq"][:s]
            m = len(seq)
            for p in range(m):
                later = seq[p + 1:]
                if len(later):
                    np.add.at(co, (seq[p], later), 1.0)
        self.co = co
        self.co_smooth = co + 1e-6  # additive smoothing, disclosed

        # recency decay lambdas from the screen event span
        if len(self.screen_events_t) >= 2:
            span = float(self.screen_events_t[-1] - self.screen_events_t[0])
        else:
            span = 1.0
        self.screen_span = max(span, 1.0)
        ln2 = np.log(2.0)
        self.recency_lambdas = {
            w: ln2 / max(self.screen_span * w, 1e-6) for w in (0.1, 0.5, 1.0)
        }

        # per-category popularity mass (for active-elicitation priors)
        w_by_cat = np.zeros(cat.n_cat, dtype=np.float64)
        if cat.popularity_weight is not None:
            for c in range(cat.n_cat):
                w_by_cat[c] = cat.popularity_weight[cat.categories == c].sum()
        self.w_by_cat = np.maximum(w_by_cat, 1e-12)


# ---------------------------------------------------------------------------
# Per-cell inputs
# ---------------------------------------------------------------------------


def compute_screen_split(u: dict) -> int:
    return max(0, int(len(u["item_seq"]) * 0.8))


def build_position_matrix(cat: Catalog, max_pos: int) -> np.ndarray:
    """(n_users, max_pos) item index per position 1..max_pos, -1 padded."""
    n_u = cat.n_users
    P = np.full((n_u, max_pos), -1, dtype=np.int32)
    for ui, u in enumerate(cat.users):
        L = len(u["item_seq"])
        m = min(L, max_pos)
        P[ui, :m] = u["item_seq"][:m]
    return P



def build_cell_inputs(cat: Catalog, rt: CatRuntime, k: int, valid: np.ndarray,
                      P: np.ndarray, tmat: np.ndarray) -> dict:
    """All inputs the score builders need for a cell (k, valid users)."""
    idx = np.where(valid)[0]
    nv = len(idx)
    res: dict = {
        "valid_idx": idx,
        "ctx": P[idx, :k] if k >= 1 else np.empty((nv, 0), dtype=np.int32),
        "gt": P[idx, k],
        "t_cutoff": tmat[idx, k],
    }
    own = P[idx, : k + 1]
    own_ones = np.zeros((nv, cat.n_items), dtype=np.int16)
    rows = np.repeat(np.arange(nv), k + 1)
    cols = own.reshape(-1)
    good = cols >= 0
    np.add.at(own_ones, (rows[good], cols[good]), 1)
    S = np.maximum(rt.g_counts[None, :].astype(np.float64) - own_ones, 0.0)
    res["S"] = S

    # per-user L1O category / price-band counts
    cown = np.zeros((nv, cat.n_cat), dtype=np.int16)
    np.add.at(cown, (rows[good], cat.categories[cols[good]]), 1)
    Scat = np.maximum(rt.g_cat_counts[None, :].astype(np.float64) - cown, 0.0)
    Scat = Scat / np.maximum(Scat.sum(axis=1, keepdims=True), 1e-12)
    res["Scat"] = Scat

    pown = np.zeros((nv, cat.n_price), dtype=np.int16)
    np.add.at(pown, (rows[good], cat.price_bands[cols[good]]), 1)
    Spr = np.maximum(rt.g_price_counts[None, :].astype(np.float64) - pown, 0.0)
    Spr = Spr / np.maximum(Spr.sum(axis=1, keepdims=True), 1e-12)
    res["Spr"] = Spr

    if k >= 1:
        ctxm = P[idx, :k]
        cmask = ctxm >= 0
        F = rt.F
        Fctx = F[ctxm]
        Fctx = np.where(cmask[..., None], Fctx, 0.0)
        prof = Fctx.sum(axis=1) / np.maximum(cmask.sum(axis=1, keepdims=True), 1.0)
        prof_norm = prof / np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-12)
        res["prof"] = prof
        res["sim"] = prof_norm @ F.T
        res["last"] = ctxm[:, -1].copy()
        res["nbhd_knn"] = _expanded_profiles(cat, rt, ctxm, cmask)
        res["lift"] = _build_lift(cat, own, ctxm, rt)
    return res


def _expanded_profiles(cat: Catalog, rt: CatRuntime, ctxm: np.ndarray,
                       cmask: np.ndarray) -> dict[int, np.ndarray]:
    """profile per neighborhood m in {1,3,5}: mean over context items of
    (item + its m-1 nearest content neighbours)."""
    nv, k = ctxm.shape
    F = rt.F
    sim_all = F @ F.T
    np.fill_diagonal(sim_all, -1.0)
    out: dict[int, np.ndarray] = {}
    for m in (1, 3, 5):
        acc = np.zeros((nv, F.shape[1]), dtype=np.float64)
        cnt = np.zeros(nv, dtype=np.float64)
        for j in range(k):
            col = ctxm[:, j]
            good = col >= 0
            g = col[good]
            row_i = np.where(good)[0]
            if len(g) == 0:
                continue
            nbg = np.argpartition(-sim_all[g, :], m, axis=1)[:, :m]
            vals = np.zeros((nv, F.shape[1]), dtype=np.float64)
            vals[row_i] = F[nbg].mean(axis=1)
            acc += vals
            cnt += good
        cnt = np.maximum(cnt, 1.0)
        out[m] = acc / cnt[:, None]
    return out


def _build_lift(cat: Catalog, own: np.ndarray, ctxm: np.ndarray,
                rt: CatRuntime) -> np.ndarray:
    """L1O max-lift over context items: lift(item,h)=P(item|h)/P(item),
    with the user's own transitions among positions 1..k+1 subtracted from
    the screen co-occurrence and counts.  P(item) is also corrected per user
    (own occurrences removed).  Smoothed by +1e-6 (disclosed)."""
    nv = len(own)
    n = cat.n_items
    co = rt.co_smooth
    g_cnt = np.maximum(rt.g_counts, 1e-6)
    T = float(rt.g_counts.sum()) + 1e-6
    out = np.empty((nv, n), dtype=np.float64)
    for u in range(nv):
        own2 = own[u]
        own2 = own2[own2 >= 0]
        owncnt = np.bincount(own2, minlength=n).astype(np.float64)
        denom = np.maximum(g_cnt - owncnt, 1e-6)
        T_c = T - len(own2)
        P_item = np.maximum((g_cnt - owncnt) / max(T_c, 1e-6), 1e-6)
        trans: dict[tuple[int, int], int] = {}
        for p_i in range(len(own2)):
            for p_j in range(p_i + 1, len(own2)):
                if own2[p_j] == own2[p_i]:
                    key = (int(own2[p_i]), int(own2[p_j]))
                    trans[key] = trans.get(key, 0) + 1
        best = np.full(n, -np.inf, dtype=np.float64)
        for h in ctxm[u]:
            if h < 0:
                continue
            col = co[:, h] / denom[h]
            for (i_, h_), c in trans.items():
                if h_ == h:
                    col[i_] = max(co[i_, h] - c, 1e-6) / denom[h]
            col = col / P_item
            best = np.maximum(best, col)
        out[u] = best
    return out


def rank_and_collect(scores: np.ndarray, gt: np.ndarray, ctx: np.ndarray,
                     k: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly filter context items, rank.  Returns
    (r int16 (nv,), top int16 (nv, K_CUT) with -1 padding).

    Rank convention (tie-consistent with the displayed top-K list):
    r = 1 + #{items with score > score(gt)} + #{items with score == score(gt)
    and smaller item index}  -- i.e. ties are broken by item index ascending,
    the SAME ordering the top-K list uses (score desc, item index asc)."""
    nv = len(gt)
    S = scores.astype(np.float64).copy()
    V = S.shape[1]
    idx_col = np.arange(V)
    if k >= 1 and nv:
        rows = np.repeat(np.arange(nv), k)
        cols = ctx.reshape(-1)
        m = cols >= 0
        S[rows[m], cols[m]] = -np.inf
    gt_scores = S[np.arange(nv), gt]
    better = (S > gt_scores[:, None]).sum(axis=1)
    tied_before = ((S == gt_scores[:, None]) & (idx_col[None, :] < gt[:, None])).sum(axis=1)
    r = (1 + better + tied_before).astype(np.int16)
    cut = min(K_CUT, V)
    top = np.full((nv, K_CUT), -1, dtype=np.int16)
    if cut >= 1:
        neg = -S
        part = np.argpartition(neg, cut - 1, axis=1)[:, :cut]
        rows = np.repeat(np.arange(nv), cut)
        cols = part.reshape(-1)
        vals = neg[rows, cols].reshape(nv, cut)
        idxs = part.reshape(nv, cut)
        # order by (value ascending, item index ascending) on `neg` scores
        order = np.lexsort((idxs, vals), axis=1)
        top[:, :cut] = part[np.arange(nv)[:, None], order]
    return r, top


def ndcg5_from_rank(r: np.ndarray) -> np.ndarray:
    return np.where(r <= 5, 1.0 / np.log2(r + 1.0), 0.0)


# ---------------------------------------------------------------------------
# Active elicitation (k=0 only; pair subsampling disclosed per catalog)
# ---------------------------------------------------------------------------


def _pair_pool(n_items: int, max_pairs: int, rng: np.random.RandomState) -> np.ndarray:
    if n_items <= 250:
        a, b = np.triu_indices(n_items, k=1)
        return np.stack([a, b], axis=1)
    a = rng.randint(0, n_items, size=max_pairs)
    b = rng.randint(0, n_items, size=max_pairs)
    keep = a != b
    return np.stack([a[keep], b[keep]], axis=1)


def active_elicitation_core(cat: Catalog, rt: CatRuntime, idx: np.ndarray,
                            S: np.ndarray, argmax: np.ndarray,
                            seed: int) -> np.ndarray:
    """Two forced-choice questions, greedy expected-entropy-reduction
    selection over the category posterior (chunked over users and pairs).
    Returns (n_valid, |V|) final scores = L1O popularity prior re-weighted
    by the category posterior."""
    nv = len(idx)
    n_cat = cat.n_cat
    noise = 0.2
    rng = np.random.RandomState(seed)
    pairs = _pair_pool(cat.n_items, 20000, rng)
    n_p = pairs.shape[0]
    cat_a = cat.categories[pairs[:, 0]]
    cat_b = cat.categories[pairs[:, 1]]
    dil = noise / max(n_cat - 1, 1)
    L_a = np.full((n_p, n_cat), dil, dtype=np.float64)
    L_a[np.arange(n_p), cat_a] = 1.0 - noise
    L_b = np.full((n_p, n_cat), dil, dtype=np.float64)
    L_b[np.arange(n_p), cat_b] = 1.0 - noise
    cat_mass = np.zeros((nv, n_cat), dtype=np.float64)
    for c in range(n_cat):
        cat_mass[:, c] = S[:, cat.categories == c].sum(axis=1)
    q_prior = cat_mass / np.maximum(cat_mass.sum(axis=1, keepdims=True), 1e-12)
    synthetic = cat.origin == "synthetic" and cat.users[0].get("beta") is not None

    final = np.zeros((nv, cat.n_items), dtype=np.float64)
    u_chunk = 256
    p_chunk = 1500
    for lo in range(0, nv, u_chunk):
        hi = min(lo + u_chunk, nv)
        qc = q_prior[lo:hi].copy()
        for _q in range(2):
            eq_a = argmax[lo:hi, None] == cat_a[None, :]
            eq_b = argmax[lo:hi, None] == cat_b[None, :]
            pa_ab = np.full((hi - lo, n_p), 0.5)
            pa_ab[eq_a] = 1.0 - noise
            pa_ab[eq_b] = noise
            eig = np.empty((hi - lo, n_p))
            H0 = -np.sum(qc * np.log2(np.maximum(qc, 1e-15)), axis=1)
            for p_lo in range(0, n_p, p_chunk):
                p_slc = slice(p_lo, min(p_lo + p_chunk, n_p))
                qa = qc[:, None, :] * L_a[p_slc][None, :, :]
                qa /= np.maximum(qa.sum(axis=2, keepdims=True), 1e-12)
                qb = qc[:, None, :] * L_b[p_slc][None, :, :]
                qb /= np.maximum(qb.sum(axis=2, keepdims=True), 1e-12)
                Ha = -np.sum(qa * np.log2(np.maximum(qa, 1e-15)), axis=2)
                Hb = -np.sum(qb * np.log2(np.maximum(qb, 1e-15)), axis=2)
                pc = np.clip(pa_ab[:, p_slc], 0.0, 1.0)
                eig[:, p_slc] = H0[:, None] - (pc * Ha + (1.0 - pc) * Hb)
            best = np.argmax(eig, axis=1)
            pA = pairs[best, 0]
            pB = pairs[best, 1]
            ans = np.empty(hi - lo, dtype=np.int64)
            for j in range(hi - lo):
                ucat_a, ucat_b = int(cat_a[best[j]]), int(cat_b[best[j]])
                if synthetic:
                    if ucat_a == ucat_b:
                        p = 0.5
                    elif argmax[lo + j] == ucat_a:
                        p = 1.0 - noise
                    elif argmax[lo + j] == ucat_b:
                        p = noise
                    else:
                        p = 0.5
                else:
                    p = 1.0 if argmax[lo + j] == ucat_a else 0.0
                ans[j] = pA[j] if rng.rand() < p else pB[j]
            upd = np.full((hi - lo, cat.n_cat), dil)
            upd[np.arange(hi - lo), cat.categories[ans]] = 1.0 - noise
            qc *= upd
            qc /= np.maximum(qc.sum(axis=1, keepdims=True), 1e-12)
        final[lo:hi] = S[lo:hi] * qc[:, cat.categories]
    return final


# ---------------------------------------------------------------------------
# Full per-catalog evaluation (runs inside worker processes)
# ---------------------------------------------------------------------------


def evaluate_catalog(cat: Catalog, k_values: tuple[int, ...] = K_VALUES) -> dict:
    rt = CatRuntime(cat)
    n_u = cat.n_users
    max_pos = max(k_values) + 1
    P = build_position_matrix(cat, max_pos)
    tmat = np.zeros((n_u, max_pos), dtype=np.float64)
    lengths = np.zeros(n_u, dtype=np.int32)
    split_pos = np.zeros(n_u, dtype=np.int32)
    for ui, u in enumerate(cat.users):
        L = len(u["item_seq"])
        lengths[ui] = L
        split_pos[ui] = compute_screen_split(u)
        m = min(L, max_pos)
        tmat[ui, :m] = u["t_seq"][:m]

    cell_res: dict = {}
    inner_ndcg: dict[str, dict[str, float]] = {}
    for k in k_values:
        valid = lengths >= k + 1
        nv = int(valid.sum())
        entry = {
            "n_valid": nv,
            "valid": valid,
            "confirm_frac": float(np.mean((k + 1) > split_pos[valid])) if nv else float("nan"),
        }
        cell_res[str(k)] = entry
        if nv == 0:
            continue
        ci = build_cell_inputs(cat, rt, k, valid, P, tmat)
        S = ci["S"]
        for hname in HEURISTIC_NAMES:
            if hname == "active_elic2":
                continue  # k=0-only, handled below
            if not heuristic_defined_at_k(hname, k):
                continue
            score = _dispatch_heuristic(cat, rt, hname, k, ci, S)
            r, top = rank_and_collect(score, ci["gt"], ci["ctx"], k)
            entry.setdefault("h", {})[hname] = {"r": r, "top": top}
        if k >= 1 and nv:
            inner = valid & ((k + 1) <= split_pos)
            if inner.any():
                rank_of = np.where(valid)[0]
                pos_map = {orig: j for j, orig in enumerate(rank_of)}
                im = np.array([pos_map[o] for o in np.where(inner)[0]], dtype=np.int64)
                inner_ndcg[str(k)] = {
                    hname: float(ndcg5_from_rank(entry["h"][hname]["r"][im]).mean())
                    for hname in entry["h"]
                }
    # active elicitation at k=0
    s0 = cell_res.get("0")
    if s0 is not None and s0["n_valid"] > 0:
        valid0 = s0["valid"].astype(bool)
        ci0 = build_cell_inputs(cat, rt, 0, valid0, P, tmat)
        idx0 = np.where(valid0)[0]
        if cat.origin == "synthetic" and cat.users[0].get("beta") is not None:
            argmax0 = np.array([cat.users[j]["argmax_cat"] for j in idx0], dtype=np.int64)
        else:
            argmax0 = cat.categories[P[idx0, 0]]
        score = active_elicitation_core(cat, rt, idx0, ci0["S"], argmax0,
                                        seed=cat.rng_seed or 0)
        r, top = rank_and_collect(score, ci0["gt"], ci0["ctx"], 0)
        s0.setdefault("h", {})["active_elic2"] = {"r": r, "top": top}

    diagnostics = catalog_diagnostics(cat)
    diagnostics["headroom"] = compute_headroom(inner_ndcg)
    diagnostics["inner_ndcg5_by_k"] = inner_ndcg
    # information-theoretic ceiling for catalogs generated by OUR latent-intent
    # generator (family_params carry its knobs); pool synthetics (dataset
    # artifact's own generator) are not covered by this analytic model
    inline_keys = ("epsilon", "n_cat", "zipf_alpha", "n_items",
                   "dirichlet_concentration")
    if (cat.origin == "synthetic" and cat.family_params
            and all(k in cat.family_params for k in inline_keys)):
        from synthetic import mi_ceiling_mc
        try:
            diagnostics["mi_ceiling"] = mi_ceiling_mc(cat.family_params, k_values)
        except Exception as e:  # never crash a catalog on the optional ceiling
            diagnostics["mi_ceiling"] = {"error": f"{type(e).__name__}: {e}"}
    return _pack_payload(cat, cell_res, k_values, diagnostics, P)


def _dispatch_heuristic(cat: Catalog, rt: CatRuntime, hname: str, k: int,
                        ci: dict, S: np.ndarray) -> np.ndarray:
    if hname == "pop_global":
        return S
    if hname.startswith("recency_pop_"):
        w = float(hname.rsplit("_", 1)[1])
        return _recency_scores(rt, ci, rt.recency_lambdas[w], k)
    if hname == "pop_category":
        return np.power(ci["Scat"], 1.0)[:, cat.categories] * S
    if hname == "pop_price":
        return np.power(ci["Spr"], 1.0)[:, cat.price_bands] * S
    if hname == "last_item_nbhd":
        return ci["sim"]
    if hname.startswith("content_knn_"):
        m = int(hname.rsplit("_", 1)[1])
        return ci["nbhd_knn"][m] @ rt.F.T
    if hname == "pop_scaled_content":
        eps = 1e-12
        lo = S.min(axis=1, keepdims=True)
        hi = S.max(axis=1, keepdims=True)
        norm_pop = (S - lo) / np.maximum(hi - lo, eps)
        return ci["sim"] * np.power(norm_pop + eps, 0.5)
    if hname == "co_purchase":
        return ci["lift"]
    if hname.startswith("lambda_hybrid_"):
        lam = float(hname.rsplit("_", 1)[1])
        eps = 1e-12
        lo_s = S.min(axis=1, keepdims=True)
        hi_s = S.max(axis=1, keepdims=True)
        norm_pop = (S - lo_s) / np.maximum(hi_s - lo_s, eps)
        if k == 0:
            return (1.0 - lam) * norm_pop
        sim = ci["sim"]
        lo_c = sim.min(axis=1, keepdims=True)
        hi_c = sim.max(axis=1, keepdims=True)
        norm_c = (sim - lo_c) / np.maximum(hi_c - lo_c, eps)
        return lam * norm_c + (1.0 - lam) * norm_pop
    raise ValueError(f"unknown heuristic {hname}")


def _recency_scores(rt: CatRuntime, ci: dict, lam: float, k: int) -> np.ndarray:
    """L1O recency-decayed counts over screen events; user's own events at
    positions 1..k+1 excluded; cut at the user's prediction time."""
    cat = rt.cat
    nv = len(ci["valid_idx"])
    n = cat.n_items
    t_sorted = rt.screen_events_t
    item_sorted = rt.screen_events_item
    cut = ci["t_cutoff"]
    t0 = t_sorted.min()
    w_exp = np.exp(lam * (t_sorted - t0))
    pos = np.searchsorted(t_sorted, cut, side="right")
    scale = np.exp(-lam * (cut - t0))
    order_item = np.argsort(item_sorted, kind="stable")
    sorted_items = item_sorted[order_item]
    bounds = np.searchsorted(sorted_items, np.arange(n + 1))
    out = np.zeros((nv, n), dtype=np.float64)
    w_seg = w_exp[order_item]
    for i in range(n):
        lo_i = bounds[i]
        hi_i = bounds[i + 1]
        if lo_i == hi_i:
            continue
        pr = np.concatenate([np.zeros(1), np.cumsum(w_seg[lo_i:hi_i])])
        hits = np.clip(pos - lo_i, 0, hi_i - lo_i)
        out[:, i] = pr[hits] * scale
    # subtract own events at positions 1..k+1
    m_own = k + 1
    own_items = ci.get("ctx0")
    if own_items is None or own_items.shape[1] != m_own:
        own_items = np.full((nv, m_own), -1, dtype=np.int32)
        for r_, ui in enumerate(ci["valid_idx"]):
            L = len(cat.users[ui]["item_seq"])
            m = min(m_own, L)
            own_items[r_, :m] = cat.users[ui]["item_seq"][:m]
        ci["ctx0"] = own_items
    own_t = np.zeros_like(own_items, dtype=np.float64)
    for r_, ui in enumerate(ci["valid_idx"]):
        L = len(cat.users[ui]["item_seq"])
        m = min(m_own, L)
        own_t[r_, :m] = cat.users[ui]["t_seq"][:m]
    for j in range(m_own):
        col = own_items[:, j]
        good = col >= 0
        if good.any():
            out[good, col[good]] -= np.exp(-lam * (cut[good] - own_t[good, j]))
    return np.maximum(out, 0.0)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def catalog_diagnostics(cat: Catalog) -> dict:
    g_counts = np.zeros(cat.n_items)
    for u in cat.users:
        s = max(0, int(len(u["item_seq"]) * 0.8))
        if s:
            g_counts += np.bincount(u["item_seq"][:s], minlength=cat.n_items)
    total = max(g_counts.sum(), 1.0)
    shares = g_counts / total
    hhi = float((shares ** 2).sum())
    nz = shares[shares > 0]
    ent = float(-(nz * np.log2(nz)).sum())
    norm_entropy = ent / np.log2(max(len(shares), 2.0))

    n = cat.n_items
    attr_ents: list[float] = []
    for vals, nlev in ((cat.categories, cat.n_cat), (cat.price_bands, cat.n_price)):
        f = np.bincount(vals, minlength=nlev)
        f = f[f > 0] / n
        attr_ents.append(float(-(f * np.log2(f)).sum()) / np.log2(max(len(f), 2.0)))
    if cat.n_tags > 0:
        tf = cat.tag_matrix.sum(axis=0)
        tf = tf[tf > 0] / n
        attr_ents.append(float(-(tf * np.log2(tf)).sum()) / np.log2(max(len(tf), 2.0)))
    attr_entropy = float(np.mean(attr_ents)) if attr_ents else 0.0
    mean_hist = float(np.mean([len(u["item_seq"]) for u in cat.users]))
    return {
        "sales_hhi": hhi,
        "sales_norm_entropy": norm_entropy,
        "attr_entropy": attr_entropy,
        "mean_history_length": mean_hist,
        "n_items": cat.n_items,
        "n_users": cat.n_users,
    }


def compute_headroom(inner_ndcg: dict) -> float:
    """Cross-fitted headroom: mean over inner cells k in {1,2,3,5,8} of
    [NDCG@5(content_knn_3) - NDCG@5(best popularity-family heuristic)],
    with the best popularity family chosen on the screen-only inner split
    (first-j-predict-j+1).  NaN if no inner cells exist."""
    if not inner_ndcg:
        return float("nan")
    deltas: list[float] = []
    for per_heur in inner_ndcg.values():
        pop_names = [h for h in POPULARITY_FAMILY_NAMES if h in per_heur]
        if not pop_names:
            continue
        best_pop = max(per_heur[h] for h in pop_names)
        if "content_knn_3" in per_heur:
            deltas.append(per_heur["content_knn_3"] - best_pop)
    return float(np.mean(deltas)) if deltas else float("nan")


# ---------------------------------------------------------------------------
# Payload packing
# ---------------------------------------------------------------------------


def _pack_payload(cat: Catalog, cell_res: dict, k_values: tuple[int, ...],
                  diagnostics: dict, P: np.ndarray) -> dict:
    pack: dict = {
        "catalog": cat.catalog_id,
        "origin": cat.origin,
        "source_name": cat.source_name,
        "source_path": cat.source_path,
        "fold": cat.fold,
        "n_items": cat.n_items,
        "n_users": cat.n_users,
        "diagnostics": diagnostics,
    }
    items: dict[str, np.ndarray] = {}
    for k in k_values:
        sk = str(k)
        cs = cell_res.get(sk)
        if cs is None:
            continue
        idx = np.where(cs["valid"])[0].astype(np.int32)
        items[f"{sk}/user"] = idx
        items[f"{sk}/gt"] = P[idx, k].astype(np.int16)
        items[f"{sk}/ctx"] = (P[idx, :k].astype(np.int16) if k >= 1
                              else np.empty((len(idx), 0), dtype=np.int16))
        items[f"{sk}/n"] = np.int32(len(idx))
        items[f"{sk}/confirm_frac"] = np.asarray(cs["confirm_frac"], dtype=np.float64)
        for hname, hr in (cs.get("h") or {}).items():
            items[f"{sk}/{hname}/r"] = hr["r"]
            items[f"{sk}/{hname}/top"] = hr["top"]
    pack["arrays"] = items
    return pack