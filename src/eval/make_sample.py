import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
RELEVANCE = 4.0
HEAVY_MIN = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--out", default="data/eval/eval_users.json")
    ap.add_argument("--n-main", type=int, default=5000)
    ap.add_argument("--n-llm", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists. The sample is frozen; use --force only if you "
                         "are sure nothing has been evaluated with it yet.")

    proc = Path(args.proc)
    train = pd.read_parquet(proc / "train.parquet", columns=["user_idx"])
    test = pd.read_parquet(proc / "test.parquet", columns=["user_idx", "rating"])

    n_train = train.groupby("user_idx").size()
    eligible = np.sort(test.loc[test["rating"] >= RELEVANCE, "user_idx"].unique())
    print(f"eligible users (>=1 relevant test item): {len(eligible):,}  (manifest says 127,687)")

    rng = np.random.RandomState(SEED)
    shuffled = rng.permutation(eligible)
    main_users = shuffled[: args.n_main]
    llm_users = main_users[: args.n_llm]
    heavy_users = [int(u) for u in main_users if n_train.get(u, 0) > HEAVY_MIN]

    payload = {
        "seed": SEED,
        "relevance_threshold": RELEVANCE,
        "heavy_min_train_ratings": HEAVY_MIN,
        "n_eligible": int(len(eligible)),
        "main": [int(u) for u in main_users],
        "llm": [int(u) for u in llm_users],
        "heavy": heavy_users,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))

    print(f"main:  {len(payload['main']):,} users")
    print(f"llm:   {len(payload['llm']):,} users (subset of main)")
    print(f"heavy: {len(payload['heavy']):,} users in main with >{HEAVY_MIN} train ratings")
    print(f"min train ratings in main: {int(n_train.reindex(main_users).min())}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
