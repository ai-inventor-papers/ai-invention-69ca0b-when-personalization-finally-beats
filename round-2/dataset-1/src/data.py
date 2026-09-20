#!/usr/bin/env python3
"""data.py - emit the concluded corpus in the exp_sel_data_out shape.

The artifact plan for THIS step (gen_art_dataset_1) set out to acquire additional
genuinely small e-commerce purchase logs OR honestly document the thin public
small-catalog landscape. After an exhaustive acquisition sweep (search_log.md,
rejections.json) NO additional real catalog met the pre-registered acceptance
bar (Retailrocket free mirror: 96 users with >=4 purchases best case + no price;
Dunnhumby: gated; boutique sweep: empty). The bar was not lowered and nothing was
fabricated.

Per the plan, the documented empty result is itself the finding. data.py therefore
emits out/full_data_out.json (and a copy out/data_out.json) in the exp_sel_data_out
schema: one dataset group "acquisition_rejection_log" whose EXAMPLES ARE THE
REJECTION RECORDS (one example per rejected candidate), each carrying
metadata_source / metadata_criterion_failed / metadata_counts. top-level metadata
records n_accepted=0. This is structurally identical to the schema demanded for
accepted-catalog corpora, so downstream discovery machinery sees a truthful empty
result instead of a padded catalog set. Run with: uv run data.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

HERE = Path(__file__).resolve().parent
OUT = HERE  # deliverable files live at the workspace root (full_data_out.json etc.)

# Retailrocket per-top-level-category subtree statistics, parsed verbatim from
# logs/explore_rr.log (filter: events.csv `event=='transaction'`; items restricted to those
# with a `categoryid` in category_tree.csv; counts of (purchased items, tx, users, users_ge5)).
# Each subtree is a distinct candidate restricted catalog that failed the acceptance bar.
TOP = {
    80: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    140: {"items_purchased": 3547, "tx": 7524, "users": 4119, "users_ge5": 85},
    168: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    171: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    181: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    231: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    250: {"items_purchased": 246, "tx": 422, "users": 348, "users_ge5": 1},
    280: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    300: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    306: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    307: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    345: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    347: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    378: {"items_purchased": 85, "tx": 146, "users": 120, "users_ge5": 1},
    395: {"items_purchased": 746, "tx": 1663, "users": 1050, "users_ge5": 33},
    431: {"items_purchased": 1, "tx": 1, "users": 1, "users_ge5": 0},
    462: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    554: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    566: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    653: {"items_purchased": 823, "tx": 1333, "users": 781, "users_ge5": 22},
    659: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    679: {"items_purchased": 271, "tx": 545, "users": 419, "users_ge5": 3},
    721: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    755: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    791: {"items_purchased": 269, "tx": 305, "users": 202, "users_ge5": 3},
    803: {"items_purchased": 4, "tx": 6, "users": 5, "users_ge5": 0},
    859: {"items_purchased": 67, "tx": 98, "users": 79, "users_ge5": 0},
    899: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    919: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    930: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    974: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1046: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1057: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1062: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1123: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1158: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1182: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1224: {"items_purchased": 1056, "tx": 1537, "users": 857, "users_ge5": 23},
    1319: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1394: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1428: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1446: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1482: {"items_purchased": 1165, "tx": 2174, "users": 1307, "users_ge5": 16},
    1490: {"items_purchased": 321, "tx": 504, "users": 384, "users_ge5": 3},
    1532: {"items_purchased": 1504, "tx": 2744, "users": 1569, "users_ge5": 31},
    1538: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1571: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1579: {"items_purchased": 12, "tx": 12, "users": 10, "users_ge5": 0},
    1594: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1597: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1600: {"items_purchased": 1437, "tx": 2742, "users": 1552, "users_ge5": 39},
    1602: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1692: {"items_purchased": 0, "tx": 0, "users": 0, "users_ge5": 0},
    1698: {"items_purchased": 91, "tx": 226, "users": 175, "users_ge5": 1},
}

RR_SRC = "Retailrocket (CIKM 2016) - sabin74/Retailrocket-Recommender-System public Git-LFS mirror (events.csv + item_properties_part1/2.csv + category_tree.csv, sha256-verified)"


def _rr_examples() -> list[dict]:
    ex = []
    # global flagship restriction
    ex.append({
        "input": json.dumps({"source": RR_SRC, "attempt": "global top-900 items by purchase-line count (all categories), MIN_PURCHASES in {4,5}"}, separators=(",", ":")),
        "output": "REJECTED",
        "metadata_source": RR_SRC,
        "metadata_attempt": "global top-900 items by purchase-line count",
        "metadata_criterion_failed": "(4) users_with_purchases<100; (2) price absent (no price field in source)",
        "metadata_verdict": "Best freely-obtainable retailrocket configuration: 900 items but only 96 users with >=4 purchases (83 with >=5), below the ~100 floor; and no price field.",
        "metadata_counts": json.dumps({"restriction": "top-900 items global", "n_items": 900, "users_ge4": 96, "users_ge5": 83}, separators=(",", ":")),
    })
    # per top-level category subtree
    for t, c in sorted(TOP.items()):
        cr = "(1) n_items<150 (0 purchased+category-labelled items in subtree)" if c["items_purchased"] == 0 else "(4) users_with_->=5_purchases<100 within restricted subtree"
        ex.append({
            "input": json.dumps({"source": RR_SRC, "attempt": f"category-subtree restriction: top-level category {t}"}, separators=(",", ":")),
            "output": "REJECTED",
            "metadata_source": RR_SRC,
            "metadata_attempt": f"top-level category {t} subtree",
            "metadata_criterion_failed": f"{cr}; (2) price absent",
            "metadata_verdict": "Subtree does not meet the acceptance bar; sparsity makes any confirm k-grid degenerate.",
            "metadata_counts": json.dumps(c, separators=(",", ":")),
        })
    return ex


def _named_examples() -> list[dict]:
    spec = [
        ("Dunnhumby Complete Journey / Customer First 2019", "Kaggle frtgnn download", "(acquisition) gated behind Kaggle login (no api token present)",
         "Not obtainable in this environment; the price-bearing real-catalog evidence source is unavailable.", {"access": "kaggle_login_required"}),
        ("Dunnhumby source-files (dunnhumby.com)", "full 9-part RDS download", "(acquisition) gated Contentful form-fill buttons; only tiny ungated samples",
         "Only 'Data-Sample' (22,437 B) and 'Sample-2K-baskets' ungated; far below the 100-user bar.", {"data_sample_zip_bytes": 22437}),
        ("HuggingFace mirror 54-acme/dunnhumby_2019", "HF Hub API load", "(acquisition) gated / 404 on the Hub API; no loadable mirror",
         "No accessible HF mirror of the Dunnhumby Customer First data found.", {"hf_api": "invalid_username_or_password"}),
        ("Maven Roasters coffee shop (HF jason1966 mirror)", "small-shop boutique sweep", "(3)/(4) no customer/user identifier (only transaction_id, product, unit_price, timestamp)",
         "Genuinely small shop with category+price+timestamp but NO user dimension; per-user histories impossible.", {"has_product_category": True, "has_unit_price": True, "has_user_id": False}),
        ("Northwind purchase orders (HF AyoubChLin/boussad)", "classic small retailer attempt", "(1)/(3) PDF purchase-order documents for text classification, not structured per-user transaction logs",
         "Not a transaction log; classic Northwind RDB is only ~77 products / ~91 customers.", {"classic_products": 77, "classic_customers": 91}),
        ("supermarket_germany.csv (workspace raw/)", "small supermarket attempt", "(1) only 6 distinct items (Product_line categories; no SKU column)",
         "6-item universe far below the 150-item floor.", {"rows": 1000, "distinct_customers": 198, "distinct_items": 6}),
        ("ecommerce_orders.csv (workspace raw/)", "small per-user e-commerce order log attempt", "(5) provenance/origin: unverifiable source, synthetic signature (365 distinct timestamps for 10k rows; sequential customer_id 1..2999, product_id 1..1000)",
         "Cannot be trusted as REAL catalog evidence; accepting it would violate the no-fabricated-provenance rule.", {"rows": 10000, "distinct_customers": 2713, "distinct_items": 999, "distinct_timestamps": 365}),
        ("Ta Feng grocery tafeng_D11-02.zip (workspace raw/)", "small grocery basket attempt", "(acquisition) corrupt/truncated zip transfer",
         "Unusable download.", {"file_bytes": 91366}),
        ("Systematic HuggingFace + web boutique/marketplace sweep", "~50 broad-term sweep, 30+ candidate families previewed", "no candidate combined per-user timestamped purchases + item category + price in a freely-downloadable, properly-licensed, <=1000-item log",
         "Landscape yield effectively zero (OCR/text/video/product-catalog or already-in-corpus).", {"candidate_families_previewed": "30+", "qualifying": 0}),
    ]
    return [{
        "input": json.dumps({"source": s, "attempt": a}, separators=(",", ":")),
        "output": "REJECTED",
        "metadata_source": s,
        "metadata_attempt": a,
        "metadata_criterion_failed": c,
        "metadata_verdict": v,
        "metadata_counts": json.dumps(cnt, separators=(",", ":")),
    } for s, a, c, v, cnt in spec]


@logger.catch(reraise=True)
def main() -> None:
    examples = _rr_examples() + _named_examples()
    data_out = {
        "metadata": {
            "corpus": "small_ecommerce_catalog_screen_iter2",
            "schema": "catalog_schema.json (iter-1 common schema)",
            "iteration": 2,
            "artifact": "gen_art_dataset_1 (real-catalog confirm-evidence widening)",
            "outcome": "no_additional_real_catalog_accepted",
            "n_accepted": 0,
            "n_rejection_records": len(examples),
            "k_grid": [0, 1, 2, 3, 5, 8],
            "finding": ("The public freely-downloadable landscape of genuinely small "
                        "(<=1000-item) per-user purchase logs with item category+price is "
                        "effectively empty in this environment: Retailrocket's most complete "
                        "free mirror yields at most 96 users with >=4 purchases (best "
                        "configuration) and has no price; Dunnhumby is gated behind login/form; "
                        "the boutique/marketplace sweep produced no qualifying candidate. "
                        "The acceptance bar was not lowered. This empty result is the "
                        "corpus-contribution finding; see search_log.md and rejections.json. "
                        "Every example is one rejected candidate: metadata_criterion_failed + "
                        "metadata_counts carry the measured counts (Retailrocket subtree counts "
                        "parsed from logs/explore_rr.log and logs/build.log)."),
        },
        "datasets": [
            {"dataset": "acquisition_rejection_log", "examples": examples},
        ],
    }
    (OUT / "full_data_out.json").write_text(json.dumps(data_out, indent=2))
    logger.info(f"wrote full_data_out.json ({len(examples)} rejection examples, n_accepted=0)")


if __name__ == "__main__":
    main()