"""Step 4 - per-user chronological 80/10/10 split.

The single most important correctness step in the pipeline. A random split
would let a model learn from a user's later ratings while predicting their
earlier ones, which inflates every metric in a way that cannot hold in
deployment.

Two details that matter and are easy to get wrong:

  1. MovieLens timestamps tie constantly, because users rate in bulk sessions.
     Sorting on timestamp alone gives a different order on different machines
     and different pandas versions. We break ties on movie_idx so the split is
     reproducible.

  2. Items that appear in test but never in train have no learned vector, so a
     collaborative model literally cannot score them. We do not drop them -
     that would quietly inflate results - we count them and report the number.

Produces:
    processed/train, processed/val, processed/test
    reports/step4_split.json
"""


from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from . import config as C
from .io_utils import banner, load_table, log, save_table, write_json


def assign_splits(ratings: pd.DataFrame) -> pd.DataFrame:
    """Label every row train / val / test by its position in that user's history."""
    df = ratings.sort_values(
        ["user_idx", "timestamp", "movie_idx"], kind="mergesort"
    ).reset_index(drop=True)

    grp = df.groupby("user_idx", sort=False)
    df["pos"] = grp.cumcount()                       # 0-based position in history
    df["n"] = grp["movie_idx"].transform("size")     # that user's total ratings

    # Guarantee at least one row in val and test whenever the user has enough
    # ratings at all; floor() alone would give a 20-rating user 2/2 but a
    # 12-rating user 1/1, and a 9-rating user 0/0.
    n = df["n"]
    n_test = np.maximum(1, np.floor(n * C.TEST_FRAC)).astype(int)
    n_val = np.maximum(1, np.floor(n * C.VAL_FRAC)).astype(int)
    n_train = n - n_val - n_test

    df["split"] = np.where(
        df["pos"] < n_train, "train",
        np.where(df["pos"] < n_train + n_val, "val", "test"),
    )

    # Users too short to split at all go entirely to train; they still inform
    # the model but are not evaluated on.
    too_short = n < C.MIN_RATINGS_FOR_SPLIT
    df.loc[too_short, "split"] = "train"

    return df.drop(columns=["pos", "n"])


def run():
    banner("STEP 4 - CHRONOLOGICAL SPLIT")

    ratings = load_table(C.PROCESSED_DIR / "ratings")
    log(f"  {len(ratings):,} ratings, {ratings['user_idx'].nunique():,} users")

    df = assign_splits(ratings)

    train = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val = df[df["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    test = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

    # ------------------------------------------------------- leakage check
    # Every validation and test interaction must come strictly after that
    # user's last training interaction. If this fails, the sort is wrong.
    last_train = train.groupby("user_idx")["timestamp"].max()
    viol = 0
    for part in (val, test):
        first = part.groupby("user_idx")["timestamp"].min()
        common = last_train.index.intersection(first.index)
        viol += int((first.loc[common] < last_train.loc[common]).sum())

    # ------------------------------------------------------- diagnostics
    train_items = set(train["movie_idx"].unique())
    train_users = set(train["user_idx"].unique())
    cold_items = int((~test["movie_idx"].isin(train_items)).sum())
    cold_users = int((~test["user_idx"].isin(train_users)).sum())

    rel = test["rating"] >= C.RELEVANCE_THRESHOLD
    users_with_relevant = int(test.loc[rel, "user_idx"].nunique())

    counts = train["user_idx"].value_counts()
    short = set(counts[counts < 10].index)
    long = set(counts[counts > 50].index)

    report = {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "pct_train": round(100 * len(train) / len(df), 2),
        "pct_val": round(100 * len(val) / len(df), 2),
        "pct_test": round(100 * len(test) / len(df), 2),
        "users_total": int(df["user_idx"].nunique()),
        "users_in_test": int(test["user_idx"].nunique()),
        "users_with_relevant_test_item": users_with_relevant,
        "relevance_threshold": C.RELEVANCE_THRESHOLD,
        "test_rows_with_cold_item": cold_items,
        "test_rows_with_cold_user": cold_users,
        "timestamp_violations": viol,
        "cold_start_users_lt10_train": len(short),
        "heavy_users_gt50_train": len(long),
    }

    log(f"  train {len(train):>10,}  ({report['pct_train']}%)")
    log(f"  val   {len(val):>10,}  ({report['pct_val']}%)")
    log(f"  test  {len(test):>10,}  ({report['pct_test']}%)")
    log("")
    log(f"  users with >=1 relevant test item : {users_with_relevant:,}")
    log(f"  test rows whose movie is unseen in train : {cold_items:,}")
    log(f"  cold-start users (<10 train ratings)     : {len(short):,}")
    log(f"  heavy users (>50 train ratings)          : {len(long):,}")
    log("")

    if viol:
        log(f"  FAIL: {viol:,} users have val/test rows older than their last "
            f"train row - the split is leaking. Do not proceed.")
    else:
        log("  OK: no user has a val/test interaction before their last train one")

    save_table(train, C.PROCESSED_DIR / "train")
    save_table(val, C.PROCESSED_DIR / "val")
    save_table(test, C.PROCESSED_DIR / "test")
    write_json(report, C.REPORT_DIR / "step4_split.json")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
