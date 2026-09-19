#!/usr/bin/env python3
"""Convert the real e-commerce sources into the common catalog schema.

Sources (raw/, all real, fold=confirm):
  * uci_online_retail_1  — UCI Online Retail (I).xlsx  (2010-12..2011-12)
  * uci_online_retail_2  — UCI Online Retail II.xlsx   (2009-12..2011-12)
  * olist                — Brazilian E-Commerce (HF mirror of the Olist 100k)
    (orders, order_items, products, product_category_name_translation)

Each source is restricted to a <=1000-item subset (small-catalog regime). The
restriction rule is recorded verbatim in family_params. Timestamps -> epoch ms.
Item 'attrs' are one-hot nominal (description-keyword category for UCI; English
product category for Olist). price_band is a 0..4 within-catalog band derived
from per-item median unit price. forced_choice_probes for real data are
revealed-preference pairs (A in the user's history, B a popular never-bought
item), documented as such.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

logger.remove()
import sys
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "processed"
OUT.mkdir(exist_ok=True)

MAX_ITEMS = 1000
MIN_PURCHASES = 5  # per-user floor for the cold-start k-grid


def _ts(x) -> int:
    return int(pd.Timestamp(x).timestamp() * 1000)


# --------------------------------------------------------------------------- #
# UCI category derivation (description keywords)
# --------------------------------------------------------------------------- #
def uci_category(desc: str) -> str:
    d = (desc or "").upper()
    rules = [
        ("lights", ["LIGHT"]),
        ("candles", ["CANDLE"]),
        ("holders", ["HOLDER"]),
        ("christmas", ["CHRISTMAS", "XMAS"]),
        ("bags", [" BAG"]),
        ("boxes", ["BOX"]),
        ("vases", ["VASE"]),
        ("mugs", ["MUG"]),
        ("clocks", ["CLOCK"]),
        ("bears", ["BEAR"]),
    ]
    for name, kws in rules:
        if any(k in d for k in kws):
            return name
    return "other"


_XLSX_CACHE: dict = {}


def _norm_uci(df: pd.DataFrame) -> pd.DataFrame:
    """UCI I uses CustomerID/UnitPrice/InvoiceNo; UCI II uses 'Customer ID'/'Price'/'Invoice'."""
    rename = {}
    if "Customer ID" in df.columns and "CustomerID" not in df.columns:
        rename["Customer ID"] = "CustomerID"
    if "Price" in df.columns and "UnitPrice" not in df.columns:
        rename["Price"] = "UnitPrice"
    if "Invoice" in df.columns and "InvoiceNo" not in df.columns:
        rename["Invoice"] = "InvoiceNo"
    return df.rename(columns=rename)


def _read_uci(xlsx: Path) -> pd.DataFrame:
    if xlsx not in _XLSX_CACHE:
        logger.info(f"Loading UCI {xlsx.name}")
        _XLSX_CACHE[xlsx] = _norm_uci(pd.read_excel(xlsx))
    return _XLSX_CACHE[xlsx]


def build_uci_catalog(
    xlsx: Path, catalog_id: str, source_name: str, restrict: str, seed: int,
    min_purchases: int = MIN_PURCHASES,
) -> dict:
    """restrict: 'all' or a keyword category label (e.g. 'lights')."""
    df = _read_uci(xlsx)
    df = df.dropna(subset=["CustomerID", "StockCode", "Description"])
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    # positive-quantity purchase lines only (drop returns / bad qty)
    df = df[df["Quantity"] > 0]
    df["ts"] = df["InvoiceDate"].map(_ts)
    df["cat"] = df["Description"].map(uci_category)

    # item-level features
    item_med_price = (
        df[df["UnitPrice"] > 0].groupby("StockCode")["UnitPrice"].median()
    )
    item_cat = df.groupby("StockCode")["cat"].first()
    item_count = df.groupby("StockCode").size()

    # choose the restricted item subset
    if restrict == "all":
        subset = item_count.sort_values(ascending=False).index[:MAX_ITEMS].tolist()
        rule = f"top-{len(subset)} items by purchase-line count (<= {MAX_ITEMS})"
    else:
        cand = [sc for sc in item_cat.index if item_cat[sc] == restrict]
        cand = sorted(cand, key=lambda sc: -item_count[sc])
        subset = cand[:MAX_ITEMS]
        rule = f"description-keyword subtree '{restrict}' ({len(subset)} items)"

    keepsrc = df[df["StockCode"].isin(subset)].copy()
    logger.info(f"{catalog_id}: {len(subset)} items, {keepsrc.shape[0]} lines")

    # prices -> bands (0..4) within the restricted catalog
    sel_price = item_med_price.reindex(subset).fillna(item_med_price.median())
    edges = np.quantile(sel_price.values, [0.2, 0.4, 0.6, 0.8])
    band = {sc: int(np.digitize(v, edges)) for sc, v in sel_price.items()}

    items = []
    for sc in subset:
        items.append(
            {
                "item_id": sc,
                "attrs": {f"cat_{item_cat[sc]}": 1.0},
                "category": item_cat[sc],
                "price_band": band[sc],
                "attribute_diversity": 1,
            }
        )
    item_set = set(subset)

    # per-user purchase sequences, timestamp-sorted, hold out last
    user_logs = []
    for cid, g in keepsrc.groupby("CustomerID"):
        g = g.sort_values("ts")
        seq = g["StockCode"].tolist()
        ts = g["ts"].tolist()
        if len(seq) < min_purchases:
            continue
        hist, h_ts = seq[:-1], ts[:-1]
        heldout = seq[-1]
        # dedupe consecutive timestamp ties? keep raw line order
        user_logs.append(
            {
                "user_id": cid,
                "ordered_history": hist,
                "heldout_item": heldout,
                "timestamps": h_ts,
            }
        )

    probes = _revealed_probes(user_logs, keepsrc, subset)
    return {
        "catalog_id": catalog_id,
        "origin": "real",
        "source_name": source_name,
        "family_params": {
            "restriction_rule": rule,
            "min_user_purchases": min_purchases,
            "n_items": len(subset),
            "forced_choice_origin": "revealed_preference",
            "seed_used": seed,
        },
        "items": items,
        "user_logs": user_logs,
        "forced_choice_probes": probes,
        "fold": "confirm",
        "partition_meta": {
            "partition": "heldout", "holdout": "last",
            "temporal": "true", "confirm_resource": "true",
        },
    }


def _revealed_probes(user_logs, keepsrc, subset) -> list:
    """Revealed-preference probes: A = most recent bought, B = popular never-bought."""
    pop = keepsrc["StockCode"].value_counts().index[:50].tolist()
    probes = []
    for u in user_logs:
        if not u["ordered_history"]:
            continue
        a = u["ordered_history"][-1]
        for b in pop[:2]:
            if b == a or b in set(u["ordered_history"]):
                continue
            probes.append(
                {
                    "user_id": u["user_id"],
                    "item_a": a,
                    "item_b": b,
                    "choice": 1,
                    "prob_note": "revealed preference (A in history, B not)",
                }
            )
            break
    return probes


# --------------------------------------------------------------------------- #
# Olist
# --------------------------------------------------------------------------- #
def build_olist_catalog(
    restrict: str, catalog_id: str, seed: int,
    min_purchases: int = MIN_PURCHASES,
) -> dict:
    odir = RAW / "olist"
    orders = pd.read_csv(odir / "olist_orders_dataset.csv")
    oi = pd.read_csv(odir / "olist_order_items_dataset.csv")
    prod = pd.read_csv(odir / "olist_products_dataset.csv")
    trans = pd.read_csv(odir / "product_category_name_translation.csv")

    prod = prod.merge(trans, on="product_category_name", how="left")
    prod["cat_en"] = prod["product_category_name_english"].fillna("other")

    m = oi.merge(
        orders[["order_id", "customer_id", "order_purchase_timestamp"]],
        on="order_id",
    ).merge(prod[["product_id", "cat_en"]], on="product_id", how="left")
    m["ts"] = pd.to_datetime(m["order_purchase_timestamp"]).astype("int64") // 10**6
    m = m.dropna(subset=["cat_en"])

    item_med = m.groupby("product_id")["price"].median()
    item_cat = m.groupby("product_id")["cat_en"].first()
    item_count = m.groupby("product_id").size()

    if restrict == "all":
        subset = item_count.sort_values(ascending=False).index[:MAX_ITEMS].tolist()
        rule = f"top-{len(subset)} products by purchase count (<= {MAX_ITEMS})"
    else:
        cand = [pid for pid in item_cat.index if item_cat[pid] == restrict]
        cand = sorted(cand, key=lambda pid: -item_count[pid])
        subset = cand[:MAX_ITEMS]
        rule = f"category subtree '{restrict}' ({len(subset)} products)"

    m = m[m["product_id"].isin(subset)]
    logger.info(f"{catalog_id}: {len(subset)} items, {m.shape[0]} rows")

    sel_price = item_med.reindex(subset).fillna(item_med.median())
    edges = np.quantile(sel_price.values, [0.2, 0.4, 0.6, 0.8])
    band = {pid: int(np.digitize(v, edges)) for pid, v in sel_price.items()}

    items = [
        {
            "item_id": pid,
            "attrs": {f"cat_{item_cat[pid]}": 1.0},
            "category": item_cat[pid],
            "price_band": band[pid],
            "attribute_diversity": 1,
        }
        for pid in subset
    ]

    user_logs = []
    for cid, g in m.groupby("customer_id"):
        g = g.sort_values("ts")
        seq = g["product_id"].tolist()
        ts = g["ts"].tolist()
        if len(seq) < min_purchases:
            continue
        user_logs.append(
            {
                "user_id": cid,
                "ordered_history": seq[:-1],
                "heldout_item": seq[-1],
                "timestamps": ts[:-1],
            }
        )

    probes = _revealed_probes_olist(user_logs, m, subset)
    return {
        "catalog_id": catalog_id,
        "origin": "real",
        "source_name": "olist_brazilian_ecommerce",
        "family_params": {
            "restriction_rule": rule,
            "min_user_purchases": min_purchases,
            "n_items": len(subset),
            "forced_choice_origin": "revealed_preference",
            "seed_used": seed,
        },
        "items": items,
        "user_logs": user_logs,
        "forced_choice_probes": probes,
        "fold": "confirm",
        "partition_meta": {
            "partition": "heldout", "holdout": "last",
            "temporal": "true", "confirm_resource": "true",
        },
    }


def _revealed_probes_olist(user_logs, m, subset) -> list:
    pop = m["product_id"].value_counts().index[:50].tolist()
    probes = []
    for u in user_logs:
        if not u["ordered_history"]:
            continue
        a = u["ordered_history"][-1]
        for b in pop[:2]:
            if b == a or b in set(u["ordered_history"]):
                continue
            probes.append(
                {
                    "user_id": u["user_id"],
                    "item_a": a,
                    "item_b": b,
                    "choice": 1,
                    "prob_note": "revealed preference (A in history, B not)",
                }
            )
            break
    return probes


def main() -> None:
    # UCI Online Retail I -> 2 catalogs (all-subset + 'lights' subtree)
    c1 = build_uci_catalog(RAW / "OnlineRetail.xlsx", "uci_retail_1", "uci_online_retail_1", "all", 0)
    c2 = build_uci_catalog(RAW / "OnlineRetail.xlsx", "uci_retail_1_lights", "uci_online_retail_1", "lights", 1)
    # UCI Online Retail II -> 2 catalogs (all-subset + 'lights' subtree)
    c3 = build_uci_catalog(RAW / "online_retail_ii" / "online_retail_II.xlsx", "uci_retail_2", "uci_online_retail_2", "all", 2)
    c3b = build_uci_catalog(RAW / "online_retail_ii" / "online_retail_II.xlsx", "uci_retail_2_lights", "uci_online_retail_2", "lights", 3)
    # Olist -> 2 catalogs (repeat-buyer-rich slices; min 4 purchases)
    c4 = build_olist_catalog("all", "olist_all", 4, min_purchases=4)
    c5 = build_olist_catalog("furniture_decor", "olist_furniture_decor", 5, min_purchases=4)

    for c in [c1, c2, c3, c3b, c4, c5]:
        (OUT / f"{c['catalog_id']}.json").write_text(json.dumps(c))
        n_ge5 = sum(1 for u in c["user_logs"] if len(u["ordered_history"]) >= 4)
        logger.info(
            f"wrote {c['catalog_id']}: items={len(c['items'])} "
            f"users={len(c['user_logs'])} "
            f"(users>=5 purchases={n_ge5}) probes={len(c['forced_choice_probes'])}"
        )


if __name__ == "__main__":
    main()