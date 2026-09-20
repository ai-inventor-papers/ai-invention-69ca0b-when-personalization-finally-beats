#!/usr/bin/env python3
"""Build small Retailrocket catalogs in the iteration-1 common schema.

Restriction strategies (each one target catalog):
  * retailrocket_top900   - top-900 items globally by purchase-line count (multi-category flagship).
  * retailrocket_cat<TOP> - top-900 items within one *top-level* category subtree (single-vertical boutique).
Every catalog: items = str ids with attrs{cat_*:1.0}, category, price_band (0, ABSENT price -> documented),
timestamp-sorted per-user purchases, leave-last-out (last purchase = heldout), user kept only if it has
>= MIN_PURCHASES purchases of that catalog's items. fold='confirm'. RAW catalog only (no HHI/entropy/headroom).
Emits one JSON per catalog into processed/ and prints the acceptance checklist for each.
"""
from __future__ import annotations
import json, resource, sys
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/build.log", rotation="30 MB", level="DEBUG")

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "rr_in"
PROC = HERE / "processed"
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))

MIN_ITEMS, MAX_ITEMS = 150, 900
MIN_PURCHASES = 5          # matching iteration-1 real-catalog norm (k in {0,1,2,3} computable for all)
MIN_USERS = 100
MAX_TOP_ITEMS = 900

# raw provenance
EVENT_SHA = None
IP_SHA = None


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def left_value(series: pd.Series) -> object:
    """Modal value of a series."""
    return series.value_counts().index[0]


@logger.catch(reraise=True)
def main() -> None:
    global EVENT_SHA, IP_SHA
    PROC.mkdir(exist_ok=True)
    EVENT_SHA = sha256(RAW / "events.csv")
    IP_SHA = sha256(RAW / "item_properties_part1.csv")[:8]

    # ---- category tree: build parent index + node->(top_root, child_of_root) ----
    tree = pd.read_csv(RAW / "category_tree.csv", dtype={"categoryid": "Int64", "parentid": "Float64"})
    parent: dict[int, int | None] = {}
    for _, r in tree.dropna(subset=["categoryid"]).iterrows():
        parent[int(r["categoryid"])] = None if pd.isna(r["parentid"]) else int(r["parentid"])
    roots = {c for c, p in parent.items() if p is None}
    memo: dict[int, tuple] = {}

    def top_and_child(cid: int) -> tuple[int, int]:
        if cid in memo:
            return memo[cid]
        assert cid in parent, f"category {cid} not in tree"
        # walk up
        cur = cid
        path = [cur]
        while parent.get(cur) is not None:
            cur = parent[cur]
            path.append(cur)
        top = cur  # root
        child = path[-2] if len(path) >= 2 and path[-1] == top else top
        memo[cid] = (top, child)
        return memo[cid]

    # ---- item -> categoryid (modal over property rows) ----
    p1 = pd.read_csv(RAW / "item_properties_part1.csv", usecols=["itemid", "property", "value"],
                     dtype={"itemid": "int64", "property": "str", "value": "str"})
    p2 = pd.read_csv(RAW / "item_properties_part2.csv", usecols=["itemid", "property", "value"],
                     dtype={"itemid": "int64", "property": "str", "value": "str"})
    prop = pd.concat([p1, p2], ignore_index=True)
    del p1, p2
    catp = prop[prop["property"] == "categoryid"].copy()
    del prop
    item_top: dict[int, int] = {}
    item_child: dict[int, int] = {}
    item_catgroup = catp.groupby("itemid")["value"]
    for it, grp in catp.groupby("itemid"):
        try:
            cid = int(float(grp["value"].value_counts().index[0]))
        except (ValueError, TypeError):
            continue
        if cid not in parent:
            continue
        top, child = top_and_child(cid)
        item_top[it] = top
        item_child[it] = child
    del catp
    logger.info(f"items mapped to category: {len(item_top)}")

    # ---- events : transactions only ----
    ev = pd.read_csv(RAW / "events.csv",
                     dtype={"visitorid": "int64", "itemid": "int64", "event": "str", "transactionid": "object"},
                     usecols=["timestamp", "visitorid", "event", "itemid", "transactionid"])
    tx = ev[ev["event"] == "transaction"].dropna(subset=["itemid", "timestamp"]).copy()
    tx = tx[tx["itemid"].isin(item_top)].copy()
    tx["itemid"] = tx["itemid"].astype(int)
    tx["visitorid"] = tx["visitorid"].astype(int)
    tx["timestamp"] = tx["timestamp"].astype("int64")
    logger.info(f"transactions with categorized item: {len(tx)}")

    # purchase-line counts for item popularity
    item_pop = Counter(tx["itemid"].tolist())

    # ---- candidate item-universes ----
    top_counts = Counter(item_top.values())
    candidates: list[tuple[str, str, list[int]]] = []
    # flagship: top-900 items globally by purchase-line count
    flagship = [i for i, _ in item_pop.most_common(MAX_TOP_ITEMS)]
    candidates.append(("retailrocket_top900", "top-900 items by purchase-line count (all top-level categories)", flagship))
    # per-subtree: top-900 items within the biggest active top-level categories
    for top, cnt in top_counts.most_common():
        if cnt < 50:
            continue
        items_in = [i for i, t in item_top.items() if t == top]
        sub = [i for i, _ in sorted(((i, item_pop[i]) for i in items_in), key=lambda x: -x[1])][:MAX_TOP_ITEMS]
        candidates.append((f"retailrocket_cat{top}", f"top-{MAX_TOP_ITEMS} items within top-level category subtree {top}", sub))

    # ---- build catalogs for candidates that pass the bar ----
    results = []
    for cid, rule, items in candidates:
        itemset = set(items)
        sub = tx[tx["itemid"].isin(itemset)].copy()
        if len(items) < MIN_ITEMS:
            logger.info(f"[{cid}] reject: n_items={len(items)} < {MIN_ITEMS}")
            continue
        # users with >= MIN_PURCHASES purchases of these items
        uc = sub.groupby("visitorid")["transactionid"].nunique()
        usr_ok = uc[uc >= MIN_PURCHASES].index.tolist()
        usr_ge4 = uc[uc >= 4].index.tolist()
        if len(usr_ok) < MIN_USERS:
            logger.info(f"[{cid}] reject: users_ge{MIN_PURCHASES}={len(usr_ok)} < {MIN_USERS} (users_ge4={len(usr_ge4)})")
            continue
        catalog = build_catalog(cid, rule, itemset, usr_ok, item_top, item_child, item_pop)
        results.append(catalog)
        logger.info(f"[{cid}] ACCEPT n_items={len(catalog['items'])} users={len(catalog['user_logs'])}")

    for c in results:
        (PROC / f"{c['catalog_id']}.json").write_text(json.dumps(c, separators=(",", ":")))
        logger.info(f"wrote processed/{c['catalog_id']}.json ({len(c['items'])} items, {len(c['user_logs'])} users)")
    logger.info(f"built {len(results)} catalogs")


def build_catalog(cid: str, rule: str, itemset: set[int],
                  usr_ok: list[int], item_top, item_child, item_pop) -> dict:
    # per-catalog category key: for a single-subtree catalog use child (level-2); flagship use top-level
    tops_in = {item_top[i] for i in itemset}
    use_child = len(tops_in) == 1
    def cat_of(i: int) -> int:
        return item_child[i] if use_child else item_top[i]

    # items
    items_list = []
    cat_names = sorted({str(cat_of(i)) for i in itemset})
    for i in itemset:
        cat = cat_of(i)
        items_list.append({
            "item_id": str(i),
            "attrs": {f"cat_{cat}": 1.0},
            "category": str(cat),
            "price_band": 0,
            "attribute_diversity": 1,
        })
    items_list.sort(key=lambda x: -item_pop[int(x["item_id"])])

    # users
    ev = pd.read_csv(RAW / "events.csv",
                     dtype={"visitorid": "int64", "itemid": "int64", "event": "str"},
                     usecols=["timestamp", "visitorid", "event", "itemid"])
    ev = ev[ev["event"] == "transaction"].copy()
    ev = ev[ev["visitorid"].isin(usr_ok)].copy()
    ev = ev[ev["itemid"].isin(itemset)].copy()
    ev["timestamp"] = ev["timestamp"].astype("int64")
    # popular items never-bought pool (for probes)
    pop_items = [i for i, _ in item_pop.most_common(40) if i in itemset]

    user_logs = []
    probes = []
    for uid in usr_ok:
        sub = ev[ev["visitorid"] == uid].sort_values("timestamp")
        hist_ids = sub["itemid"].astype(int).tolist()
        # collapse duplicate item purchases at same timestamp is not needed; keep purchase-line order
        timestamps = sub["timestamp"].tolist()
        ordered = [str(i) for i in hist_ids[:-1]]
        heldout = str(hist_ids[-1])
        if not ordered:
            continue
        assert len(timestamps) == len(hist_ids)
        assert timestamps == sorted(timestamps), (uid, "ts order")
        user_logs.append({"user_id": str(uid), "ordered_history": ordered, "heldout_item": heldout,
                          "timestamps": timestamps[:-1]})
        # revealed-preference probe: A = most recent observed purchase, B = popular never-bought item
        set_bought = set(hist_ids[:-1])
        b = next((p for p in pop_items if p not in set_bought and p != int(heldout) if True), None)
        if b is not None:
            probes.append({"user_id": str(uid), "item_a": ordered[-1], "item_b": str(b), "choice": 1,
                           "prob_note": "revealed preference (A in history, B not)"})

    family_params = {
        "source": "retailrocket_cikm2016",
        "license": "public research data (CIKM 2016 paper, redistributed under session-rec mirror; no price field exists in source)",
        "download_mirror": "sabin74/Retailrocket-Recommender-System (public Git-LFS) + session-rec Dropbox events mirror",
        "download_sha256": {"events.csv": EVENT_SHA, "item_properties_sha8": IP_SHA},
        "restriction_rule": rule,
        "category_level": "child(top-level-subnode)" if use_child else "top-level",
        "min_user_purchases": MIN_PURCHASES,
        "catalog_size": len(items_list),
        "n_categories": len(cat_names),
        "price_available": False,
        "price_band_origin": "absent (source has no price field); all items placed in band 0; banded-by-price NOT supported by this catalog",
        "forced_choice_origin": "revealed_preference",
    }
    catalog = {
        "catalog_id": cid,
        "origin": "real",
        "source_name": "retailrocket_cikm2016",
        "family_params": family_params,
        "items": items_list,
        "user_logs": user_logs,
        "forced_choice_probes": probes,
        "fold": "confirm",
        "partition_meta": {"partition": "heldout", "holdout": "last", "temporal": "true",
                           "confirm_resource": "true",
                           "split_levels": {
                               "within_catalog": "per-user logs split leave-last-out (prior items = train half, last = heldout)",
                               "catalog_level": "fold=confirm -> confirm-round only; never used for screen fitting"}},
    }
    return catalog


if __name__ == "__main__":
    main()