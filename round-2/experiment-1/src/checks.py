#!/usr/bin/env python3
"""Unit / logic checks C1-C8 (testing plan of the artifact plan).

Each check returns (name, passed, detail).  run_checks() exits non-zero on
any failure.  Checks C2/C4/C6 use instrumented micro-catalogs; C6 asserts
the hypothesis-crossing DIRECTIONS are detectable --- if they do not hold
the sweep reports them loudly instead of "fixing" the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

import synthetic
from synthetic import Catalog, generate_catalog, make_config
from eval_metrics import evaluate_catalog, build_item_features
from heuristics import HEURISTIC_NAMES


def _summary_r(payload: dict, k: int, h: str) -> np.ndarray:
    return payload["arrays"][f"{k}/{h}/r"]


def _ndcg5(r: np.ndarray) -> np.ndarray:
    return np.where(r <= 5, 1.0 / np.log2(r + 1.0), 0.0)


def _ndcg10(r: np.ndarray) -> np.ndarray:
    return np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)


# ---------------------------------------------------------------------------


def c1_metric_recomputation() -> tuple[bool, str]:
    """C1: NDCG@5/Recall@5 from stored predict_ rankings equal closed form."""
    cat = generate_catalog(synthetic.smoke_specs(300)[0])
    payload = evaluate_catalog(cat)
    arrs = payload["arrays"]
    item_ids = list(cat.item_ids)
    rng = np.random.default_rng(0)
    checked = 0
    for k in (1, 3, 5, 8):
        if f"{k}/n" not in arrs:
            continue
        nv = int(arrs[f"{k}/n"])
        idx = rng.choice(nv, size=min(40, nv), replace=False)
        for h in HEURISTIC_NAMES:
            if f"{k}/{h}/r" not in arrs:
                continue
            for j in idx:
                r = int(arrs[f"{k}/{h}/r"][j])
                top = [int(x) for x in arrs[f"{k}/{h}/top"][j] if int(x) >= 0]
                name_to_pos = {item_ids[i]: pos for pos, i in enumerate(top)}
                gt_name = str(item_ids[int(arrs[f"{k}/gt"][j])])
                pos = name_to_pos.get(gt_name)
                ndcg_rank = (1.0 / np.log2(r + 1)) if r <= 5 else 0.0
                rec_rank = 1.0 if r <= 5 else 0.0
                if pos is not None and pos < 5:
                    ndcg_top, rec_top = 1.0 / np.log2(pos + 2), 1.0
                else:
                    ndcg_top, rec_top = 0.0, 0.0
                if abs(ndcg_rank - ndcg_top) > 1e-9 or abs(rec_rank - rec_top) > 1e-9:
                    return False, f"mismatch k={k} h={h} user={j}: r={r} ndcg={ndcg_rank:.4f} vs {ndcg_top:.4f}"
                checked += 1
    return True, f"recomputed {checked} (user,k,h) cells from rankings == closed forms"


def c2_popularity_bayes_optimal_k0() -> tuple[bool, str]:
    """C2: with pure-popularity users (eps=0), pop_global k=0 ranking
    correlates near-maximally with the generator's popularity_weight."""
    cfg = make_config(100, "LOW", 1.5, 0.2, 10, 0, 300)
    cfg["epsilon"] = 0.0  # all draws from the crowd arm
    cfg["dirichlet_concentration"] = 4.0
    cat = generate_catalog(cfg)
    payload = evaluate_catalog(cat)
    r = _summary_r(payload, 0, "pop_global")  # per-user rank of true next
    # mean score implied: items with lower mean rank should track w
    arrs = payload["arrays"]
    nv = int(arrs["0/n"])
    # correlate mean L1O popularity count per item with w_i
    from eval_metrics import CatRuntime, build_position_matrix
    rt = CatRuntime(cat)
    P = build_position_matrix(cat, 9)
    own = P[:, :1]
    own_ones = np.zeros((cat.n_users, cat.n_items), dtype=np.int16)
    np.add.at(own_ones, (np.arange(cat.n_users), own[:, 0]), 1)
    S = np.maximum(rt.g_counts[None, :] - own_ones, 0.0)
    mean_count = S.mean(axis=0)
    w = cat.popularity_weight
    corr = float(np.corrcoef(mean_count, w)[0, 1])
    ok = corr > 0.95
    return ok, f"Pearson(mean L1O pop count, w_i) = {corr:.4f} (>0.95? {ok})"


def c3_content_knn_separation() -> tuple[bool, str]:
    """C3: a user whose history is entirely in one category ranks that
    category's items above a disjoint category's items on average."""
    cfg = make_config(100, "HIGH", 1.5, 0.6, 10, 0, 200)
    cat = generate_catalog(cfg)
    mut = generate_catalog(cfg)
    # force every user's first 3 purchases into the most populated category
    cat_counts = np.bincount(cat.categories, minlength=cat.n_cat)
    least = int(np.argmax(cat_counts))
    picks = np.where(mut.categories == least)[0]
    if len(picks) < 3:
        return False, "fewer than 3 items in a single category (n too small)"
    for u in mut.users:
        m = min(3, len(u["item_seq"]))
        u["item_seq"][:m] = picks[:m]
    from eval_metrics import CatRuntime
    rt = CatRuntime(mut)
    F = rt.F
    ok = True
    worst = float("inf")
    for u in mut.users[:50]:
        hist = u["item_seq"][: min(3, len(u["item_seq"]))]
        prof = F[hist].mean(axis=0)
        prof = prof / max(np.linalg.norm(prof), 1e-12)
        sim = prof @ F.T
        same = sim[mut.categories == least]
        other = sim[mut.categories != least]
        sep = float(same.mean() - other.mean())
        worst = min(worst, sep)
        if sep <= 0:
            ok = False
            break
    return ok, f"min cosine separation (same-cat mean - other-cat mean) = {worst:.6f} (>0)"


def c4_recency_trend() -> tuple[bool, str]:
    """C4: contrast all-time popularity with a SHORT recency window on equal
    overall counts but different timing.  Item G is bought by EVERY user at
    EARLY screen positions 0..3; item T is bought by TREND_FRAC of users at
    LATE screen positions 4..7 (both 4 purchases/user, so G's total screen
    count exceeds T's and all-time popularity favors G).  The short recency
    window (recency_pop_0.1) must rank T ABOVE its popularity rank, measured
    over the NORMAL (non-trend) users at k=8 for whom T is not in their own
    context (so T is rankable); a no-trend control must NOT show that gap."""
    TREND_FRAC = 0.35
    rng = np.random.default_rng(3)
    n_items, n_users = 20, 600
    w = (np.arange(1, n_items + 1) ** -1.5)
    w = w / w.sum()
    item_ids = np.array([f"i{i}" for i in range(n_items)], dtype=object)
    categories = np.zeros(n_items, dtype=np.int32)
    categories[10:] = 1
    price_bands = (np.arange(n_items) % 2).astype(np.int32)
    tags = np.zeros((n_items, 2), dtype=bool)
    for i in range(n_items):
        tags[i, i % 2] = True
    G, T = 3, 19  # perennial item (rank 3 in w), trend item (rank 19 in w)

    def build(n_trend: int) -> Catalog:
        users: list[dict] = []
        for u in range(n_users):
            seq = np.empty(10, dtype=np.int32)  # L=10 -> screen=0..7 (8 events)
            seq[0:4] = G                      # everyone buys G EARLY (screen 0..3)
            if u < n_trend:                   # trend users buy T LATEST (4..7)
                seq[4:8] = T
            else:
                seq[4:8] = rng.choice(n_items, size=4, p=w)
            seq[8:10] = rng.choice(n_items, size=2, p=w)  # confirm 8..9
            users.append({"user_id": f"u{u}", "item_seq": seq,
                          "t_seq": np.arange(10, dtype=np.float64) + rng.uniform(size=10),
                          "beta": None, "epsilon": None, "argmax_cat": None})
        return Catalog("c4", "synthetic", "check", "screen", {"C4": True}, 3,
                       item_ids, categories, price_bands, tags, w, users)

    from eval_metrics import CatRuntime, build_cell_inputs, _recency_scores, \
        build_position_matrix
    n_tr = int(TREND_FRAC * n_users)

    def score_rank(cat: Catalog, h: str, item_i: int) -> np.ndarray:
        """Rank of item_i in the UNFILTERED k=8 score vector (so context
        filtering cannot remove the popularity champion), over NORMAL users
        (slot >= n_tr) at the k=8 cell (prediction position = purchase 9)."""
        rt = CatRuntime(cat)
        max_pos = 9
        Pmat = build_position_matrix(cat, max_pos)
        tmat = np.zeros((cat.n_users, max_pos), dtype=np.float64)
        lengths = np.zeros(cat.n_users, dtype=np.int32)
        for ui, u in enumerate(cat.users):
            L = len(u["item_seq"])
            lengths[ui] = L
            m = min(L, max_pos)
            tmat[ui, :m] = u["t_seq"][:m]
        valid = lengths >= 9
        ci = build_cell_inputs(cat, rt, 8, valid, Pmat, tmat)
        scores = (_recency_scores(rt, ci, rt.recency_lambdas[0.1], 8)
                  if h == "recency_pop_0.1" else ci["S"])
        ranks: list[int] = []
        for j, u_slot in enumerate(ci["valid_idx"]):
            if u_slot < n_tr:  # trend users only: T is not their ranking signal
                continue
            order = np.argsort(-scores[j], kind="stable")
            pos = np.where(order == item_i)[0]
            ranks.append(int(pos[0]) + 1 if len(pos) else 999)
        return np.array(ranks, dtype=np.float64)

    p_tr = build(int(TREND_FRAC * n_users))
    p_ct = build(0)
    r_T_rec_t = score_rank(p_tr, "recency_pop_0.1", T)
    r_T_pop_t = score_rank(p_tr, "pop_global", T)
    r_T_rec_c = score_rank(p_ct, "recency_pop_0.1", T)
    r_T_pop_c = score_rank(p_ct, "pop_global", T)
    gap_trend = float(r_T_rec_t.mean() - r_T_pop_t.mean())
    gap_ctrl = float(r_T_rec_c.mean() - r_T_pop_c.mean())
    if not (gap_trend < -0.5):
        return False, (f"trend: T rank recency={r_T_rec_t.mean():.2f} vs "
                       f"pop={r_T_pop_t.mean():.2f} (gap {gap_trend:+.2f}) -- "
                       f"recency did not beat popularity on the trend item")
    if gap_ctrl > 0.5:
        return False, (f"control: gap {gap_ctrl:+.2f} unexpectedly large")
    return True, (f"trend gap (recency-pop on T) = {gap_trend:+.2f} (recency wins), "
                  f"control gap = {gap_ctrl:+.2f} (no trend preserved)")


def c5_monotonicity() -> tuple[bool, str]:
    """C5: NDCG@10 >= NDCG@5 for every user cell."""
    cat = generate_catalog(synthetic.smoke_specs(300)[0])
    payload = evaluate_catalog(cat)
    arrs = payload["arrays"]
    cells = 0
    for k in (0, 1, 2, 3, 5, 8):
        for h in HEURISTIC_NAMES:
            if f"{k}/{h}/r" not in arrs:
                continue
            r = arrs[f"{k}/{h}/r"]
            d = _ndcg10(r) - _ndcg5(r)
            if (d < -1e-9).any():
                return False, f"k={k} h={h}: NDCG@10 < NDCG@5 for {int((d < 0).sum())} cells"
            cells += len(r)
    return True, f"monotonic over {cells} user cells"


def c6_crossover_direction() -> tuple[bool, str]:
    """C6: (i) tiny concentrated LOW-entropy: pop_global >= pop_category at small k;
    (ii) large HIGH-entropy sticky long-history: content_knn_3 overtakes pop early
    (k* small).  Logs loudly rather than 'fixing' the data."""
    tiny = generate_catalog(synthetic.make_config(20, "LOW", 1.5, 0.6, 4, 0, 600))
    p_tiny = evaluate_catalog(tiny)
    msgs: list[str] = []
    ok = True
    for k in (0, 1):
        m_pop = float(_ndcg5(_summary_r(p_tiny, k, "pop_global")).mean())
        m_cat = float(_ndcg5(_summary_r(p_tiny, k, "pop_category")).mean())
        msgs.append(f"tiny k={k}: pop_global {m_pop:.4f} vs pop_category {m_cat:.4f}")
        if m_pop + 1e-6 < m_cat:
            ok = False
    big = generate_catalog(synthetic.make_config(1000, "HIGH", 1.5, 0.6, 10, 1, 600))
    p_big = evaluate_catalog(big)
    k_star = None
    for k in (1, 2, 3, 5, 8):
        m_cont = float(_ndcg5(_summary_r(p_big, k, "content_knn_3")).mean())
        m_pop = float(_ndcg5(_summary_r(p_big, k, "pop_global")).mean())
        msgs.append(f"big k={k}: content_knn_3 {m_cont:.4f} vs pop_global {m_pop:.4f}")
        if k_star is None and m_cont > m_pop + 1e-6:
            k_star = k
    msgs.append(f"k* (content beats pop first): {k_star}")
    if k_star is None or k_star > 3:
        ok = False
    detail = " | ".join(msgs)
    if not ok:
        detail += "  <<< did NOT match the predicted directions; reported as-is"
    return ok, detail


def c7_determinism() -> tuple[bool, str]:
    """C7: same seed twice => byte-identical payload JSON + parts."""
    cat1 = generate_catalog(synthetic.smoke_specs(200)[0])
    cat2 = generate_catalog(synthetic.smoke_specs(200)[0])
    p1 = evaluate_catalog(cat1)
    p2 = evaluate_catalog(cat2)
    for k in (0, 1, 2, 3, 5, 8):
        for h in HEURISTIC_NAMES:
            if f"{k}/{h}/r" in p1["arrays"]:
                a = p1["arrays"][f"{k}/{h}/r"]
                b = p2["arrays"][f"{k}/{h}/r"]
                if not np.array_equal(a, b):
                    return False, f"rank arrays differ at k={k} h={h}"
    # serialize examples twice through output.build_examples
    import output as output_mod
    meta = {
        "item_ids": {"c": [str(x) for x in cat1.item_ids]},
        "user_ids": {"c": [u["user_id"] for u in cat1.users]},
        "screen_pos": {"c": [max(0, int(len(u["item_seq"]) * 0.8)) for u in cat1.users]},
    }
    p1["catalog"] = "c"
    p2["catalog"] = "c"
    e1 = json.dumps(output_mod.build_examples(p1, meta), sort_keys=True)
    e2 = json.dumps(output_mod.build_examples(p2, meta), sort_keys=True)
    if e1 != e2:
        return False, "example JSON differs between identical runs"
    return True, "byte-identical payloads and example JSON across identical runs"


def c8_no_leak() -> tuple[bool, str]:
    """C8: k=0 pop_global ranks (and the screen-fitted counts behind them)
    are unchanged when confirm-half purchases are replaced with a different
    item -- i.e. confirm events never enter any fitted parameter.  (Truncating
    histories is NOT a valid probe: it shifts the 0.8 screen boundary.)"""
    cat = generate_catalog(synthetic.smoke_specs(400)[0])
    p_orig = evaluate_catalog(cat)
    import copy
    cat2 = copy.deepcopy(cat)
    for u in cat2.users:
        s = max(0, int(len(u["item_seq"]) * 0.8))
        if len(u["item_seq"]) > s:
            u["item_seq"][s:] = 0  # scramble confirm-half items (s unchanged)
    p_red = evaluate_catalog(cat2)

    from eval_metrics import CatRuntime
    g0 = CatRuntime(cat).g_counts
    g1 = CatRuntime(cat2).g_counts
    if not np.array_equal(g0, g1):
        return False, "screen-fitted popularity counts depend on confirm events (leak!)"

    def per_user(payload: dict) -> dict:
        arrs = payload["arrays"]
        users = arrs["0/user"]
        r = arrs["0/pop_global/r"]
        gt = arrs["0/gt"]
        return {int(users[j]): (int(r[j]), int(gt[j])) for j in range(len(users))}

    m0 = per_user(p_orig)
    m1 = per_user(p_red)
    for u_slot, (r0, g0v) in m0.items():
        r1, g1v = m1[u_slot]
        if g0v != g1v:
            return False, f"k=0 ground truth changed for user {u_slot} (bug)"
        if r0 != r1:
            return False, f"k=0 pop_global rank changed for user {u_slot} (leak: r0={r0} r1={r1})"
    return True, ("screen counts + k=0 pop_global ranks unchanged for all "
                  f"{len(m0)} users after confirm items scrambled")


def _check_functions() -> dict:
    return {
        "C1": c1_metric_recomputation,
        "C2": c2_popularity_bayes_optimal_k0,
        "C3": c3_content_knn_separation,
        "C4": c4_recency_trend,
        "C5": c5_monotonicity,
        "C6": c6_crossover_direction,
        "C7": c7_determinism,
        "C8": c8_no_leak,
    }


def run_checks(names: list[str] | None = None) -> bool:
    fns = _check_functions()
    sel = names or list(fns)
    ok_all = True
    for name in sel:
        if name not in fns:
            logger.error(f"unknown check {name}")
            ok_all = False
            continue
        try:
            passed, detail = fns[name]()
        except Exception as e:
            passed, detail = False, f"EXCEPTION {type(e).__name__}: {e}"
        if name == "C6":
            # hypothesis-direction checkpoint: report loudly, do NOT gate
            # (the plan: if the directions do not hold, log and report what
            # the sweep actually shows; do not "fix" the data)
            logger.warning(f"{name} CHECKPOINT (non-gating): {'held' if passed else 'did NOT hold'} | {detail}")
            continue
        logger.info(f"{name}: {'PASS' if passed else 'FAIL'} | {detail}")
        ok_all = ok_all and passed
    logger.info("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if run_checks() else 1)