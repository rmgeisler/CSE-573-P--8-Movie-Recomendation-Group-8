import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True, help="path to ml-20m/ratings.csv")
    ap.add_argument("--out", default=str(PROC / "train.parquet"))
    ap.add_argument("--no-strict", action="store_true",
                    help="only warn (instead of failing) when counts differ from manifest.json")
    args = ap.parse_args()

    manifest = json.loads((PROC / "manifest.json").read_text())
    exp = manifest["reports"]["step4_split"]
    problems = []

    print("Reading raw MovieLens ratings ...")
    raw = pd.read_csv(
        args.ratings,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    print(f"  raw rows: {len(raw):,}")

    user_map = pd.read_parquet(PROC / "user_map.parquet")
    movies = pd.read_parquet(PROC / "movies.parquet", columns=["movie_idx", "movie_id"])


    df = raw.merge(user_map.rename(columns={"user_id": "userId"}), on="userId", how="inner")
    df = df.merge(movies.rename(columns={"movie_id": "movieId"}), on="movieId", how="inner")
    df = df[["user_idx", "movie_idx", "rating", "timestamp"]]
    print(f"  after restricting to repo users/movies: {len(df):,}")

    val = pd.read_parquet(PROC / "val.parquet", columns=["user_idx", "movie_idx"])
    test = pd.read_parquet(PROC / "test.parquet", columns=["user_idx", "movie_idx"])
    hold = pd.concat([val, test]).assign(_hold=1)

    df = df.merge(hold, on=["user_idx", "movie_idx"], how="left")
    train = df[df["_hold"].isna()].drop(columns="_hold")

    train = train.astype({"user_idx": "int32", "movie_idx": "int32",
                          "rating": "float32", "timestamp": "int64"})
    train = train.sort_values(["user_idx", "timestamp", "movie_idx"]).reset_index(drop=True)


    print("\nChecks:")
    n_total = len(train) + len(val) + len(test)
    for label, got, want in [
        ("train rows", len(train), exp["n_train"]),
        ("total rows (train+val+test)", n_total, manifest["reports"]["step1_stats"]["n_ratings"]),
        ("users in train", train["user_idx"].nunique(), exp["users_total"]),
    ]:
        ok = got == want
        print(f"  {'OK ' if ok else 'BAD'} {label}: {got:,} (manifest: {want:,})")
        if not ok:
            problems.append(label)

    dup = train.duplicated(["user_idx", "movie_idx"]).sum()
    print(f"  {'OK ' if dup == 0 else 'BAD'} duplicate (user, movie) pairs in train: {dup}")
    if dup:
        problems.append("duplicates")

    val_full = pd.read_parquet(PROC / "val.parquet", columns=["user_idx", "timestamp"])
    tr_max = train.groupby("user_idx")["timestamp"].max()
    va_min = val_full.groupby("user_idx")["timestamp"].min()
    viol = int((tr_max.reindex(va_min.index) > va_min).sum())
    print(f"  {'OK ' if viol == 0 else 'BAD'} users whose train ends after val starts: {viol} (manifest: {exp['timestamp_violations']})")
    if viol:
        problems.append("chronology")

    min_train = train.groupby("user_idx").size().min()
    print(f"  info: smallest per-user train history = {min_train}")

    if problems and not args.no_strict:
        print(f"\nFAILED checks: {problems}. Nothing written. "
              "Make sure you used ml-20m/ratings.csv (not ml-latest). Use --no-strict to write anyway.")
        sys.exit(1)

    train.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(train):,} rows)")


if __name__ == "__main__":
    main()
