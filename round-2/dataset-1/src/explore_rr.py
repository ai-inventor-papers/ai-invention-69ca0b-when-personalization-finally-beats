#!/usr/bin/env python3
"""Exploratory analysis of the Retailrocket dataset (sabin74 LFS mirror):
event-type distribution, item->category coverage, category-tree structure,
and per-top-level-subtree transaction/user/item counts. No catalogs are
emitted here; it prints the numbers used to pick category-subtree catalogs."""
from __future__ import annotations
import resource, sys
from pathlib import Path
import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "rr_in"

# cap RAM conservatively (Retailrocket item_properties ~ 7.4M rows)
_avail = resource.getrlimit(resource.RLIMIT_AS)[1]
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))

@logger.catch(reraise=True)
def main() -> None:
    # --- events ---
    ev = pd.read_csv(RAW / "events.csv", dtype={"visitorid": "int64", "itemid": "int64"},
                     usecols=["timestamp", "visitorid", "event", "itemid", "transactionid"])
    logger.info(f"events rows={len(ev)}")
    logger.info(f"event value counts:\n{ev['event'].value_counts().sort_index()}")
    tx = ev[ev["event"] == "transaction"].copy()
    logger.info(f"transactions(event='transaction') rows={len(tx)} unique_visitors={tx['visitorid'].nunique()} unique_items={tx['itemid'].nunique()}")

    # track which items ever appear in a transaction
    tx_items = set(tx["itemid"].unique())
    logger.info(f"items ever purchased: {len(tx_items)}")

    # --- item properties: keep only categoryid ---
    p1 = pd.read_csv(RAW / "item_properties_part1.csv", usecols=["itemid", "property", "value"],
                     dtype={"itemid": "int64", "property": "object", "value": "object"})
    p2 = pd.read_csv(RAW / "item_properties_part2.csv", usecols=["itemid", "property", "value"],
                     dtype={"itemid": "int64", "property": "object", "value": "object"})
    prop = pd.concat([p1, p2], ignore_index=True)
    del p1, p2
    logger.info(f"item_properties total rows={len(prop)}; property value counts top:\n{prop['property'].value_counts().head(12)}")
    catp = prop[prop["property"] == "categoryid"].copy()
    logger.info(f"categoryid rows={len(catp)}; unique items w/ categoryid={catp['itemid'].nunique()}")
    # per item, take the modal category value
    item_cat = catp.groupby("itemid")["value"].agg(lambda s: s.value_counts().index[0]).to_dict()
    cated_items = set(item_cat.keys())
    logger.info(f"items with a category: {len(cated_items)}; of which are purchased: {len(cated_items & tx_items)}")

    # --- category tree ---
    tree = pd.read_csv(RAW / "category_tree.csv", dtype={"categoryid": "Int64", "parentid": "Float64"})
    tree.columns = ["categoryid", "parentid"]
    parent = {}
    for _, r in tree.dropna(subset=["categoryid"]).iterrows():
        cid = int(r["categoryid"])
        pid = None if pd.isna(r["parentid"]) else int(r["parentid"])
        parent[cid] = pid
    roots = [c for c, p in parent.items() if p is None]
    logger.info(f"category tree nodes={len(parent)} roots={roots}")

    # top-level (child of a root) mapping
    def top_level(cid):
        seen = set()
        cur = cid
        while parent.get(cur) is not None and cur not in seen:
            seen.add(cur)
            cur = parent[cur]
        return str(cur)
    # items -> top-level category label
    top_counts = {}
    for it, cat in item_cat.items():
        try:
            cid = int(float(cat))
        except ValueError:
            continue
        t = top_level(cid)
        top_counts.setdefault(t, []).append(it)

    # --- collect per-subtree stats using transactions ---
    # map item -> top-level
    item_top = {}
    for it, cat in item_cat.items():
        try:
            cid = int(float(cat))
            item_top[it] = top_level(cid)
        except (ValueError, KeyError):
            continue
    tx["top"] = tx["itemid"].map(item_top)
    tx_withcat = tx.dropna(subset=["top"]).copy()
    logger.info(f"transactions with categorized item: {len(tx_withcat)} / {len(tx)}")

    print("\n=== per-top-level-category subtree stats (purchased items only) ===")
    rows = []
    for t in sorted(top_counts, key=lambda x: int(x)):
        items_in = [i for i in top_counts[t] if i in tx_items]
        n_items = len(items_in)
        sub = tx_withcat[tx_withcat["top"] == t]
        n_users = sub["visitorid"].nunique()
        users_ge5 = sub.groupby("visitorid")["transactionid"].nunique()
        n_users_ge5 = int((users_ge5 >= 5).sum())
        n_tx = len(sub)
        rows.append((t, n_items, n_tx, n_users, n_users_ge5))
        print(f"  top={t:>5}  items_purchased={n_items:>5}  tx={n_tx:>7}  users={n_users:>6}  users_ge5={n_users_ge5:>5}")

if __name__ == "__main__":
    main()