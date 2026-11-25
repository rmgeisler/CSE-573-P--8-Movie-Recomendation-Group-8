import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import build_genres, build_novelty, score_lists, summarize
from src.eval.run_systems import ALL_SYSTEMS, K, LABELS, RUNNERS
from src.kg.build_graph import KnowledgeGraph


def truncate(train, users, n):
    in_group = train["user_idx"].isin(users)
    kept = (train[in_group]
            .sort_values(["user_idx", "timestamp", "movie_idx"], kind="stable")
            .groupby("user_idx", sort=False)
            .head(n))
    return pd.concat([train[~in_group], kept], ignore_index=True)


def write_summary(out_dir, systems, ns, heavy):
    rows = []
    for s in systems:
        full_f = out_dir / f"per_user_{s}.csv"
        if not full_f.exists():
            continue
        full = pd.read_csv(full_f)
        row = {"system": LABELS[s],
               "NDCG@10 full": full["ndcg@10"].mean(),
               "NDCG@10 heavy(>50)": full.loc[full["user_idx"].isin(heavy), "ndcg@10"].mean()}
        for n in ns:
            f = out_dir / f"cold_first{n}" / f"per_user_{s}.csv"
            if f.exists():
                row[f"NDCG@10 first {n}"] = pd.read_csv(f)["ndcg@10"].mean()
        row["Recall@100 full"] = full["recall@100"].mean()
        for n in ns:
            f = out_dir / f"cold_first{n}" / f"per_user_{s}.csv"
            if f.exists():
                row[f"Recall@100 first {n}"] = pd.read_csv(f)["recall@100"].mean()
        rows.append(row)
    df = pd.DataFrame(rows).set_index("system")
    text = df.round(4).to_markdown()
    (out_dir / "coldstart_summary.md").write_text(text + "\n")
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--users", choices=["main", "llm"], default="main")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    ap.add_argument("--ns", default="5,10", help="history lengths to simulate")
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--users-file", default="data/eval/eval_users.json")
    ap.add_argument("--out", default=None, help="base results folder (default results/<split>)")
    ap.add_argument("--kg", default="data/kg/kg.pkl")
    ap.add_argument("--kg-params", default="tests/results/kg_best_params.json")
    ap.add_argument("--als-factors", type=int, default=64)
    ap.add_argument("--als-reg", type=float, default=0.01)
    ap.add_argument("--knn-neighbors", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out or f"results/{args.split}")
    proc = Path(args.proc)
    ns = [int(x) for x in args.ns.split(",")]
    systems = [s.strip() for s in args.systems.split(",")]

    train = pd.read_parquet(proc / "train.parquet", columns=["user_idx", "movie_idx", "rating", "timestamp"])
    split = pd.read_parquet(proc / f"{args.split}.parquet", columns=["user_idx", "movie_idx", "rating"])
    movies = pd.read_parquet(proc / "movies.parquet", columns=["movie_idx", "genres"])

    cfg = json.loads(Path(args.users_file).read_text())
    users = np.array(cfg[args.users][: args.limit] if args.limit else cfg[args.users], dtype=np.int64)
    rel_all = split[split["rating"] >= cfg["relevance_threshold"]].groupby("user_idx")["movie_idx"].apply(set)
    relevant = {int(u): rel_all[u] for u in users if u in rel_all.index}
    users = np.array(sorted(relevant), dtype=np.int64)
    print(f"split={args.split}  scoring {len(users):,} users at history lengths {ns}")

    seen = train[train["user_idx"].isin(users)].groupby("user_idx")["movie_idx"].apply(set).to_dict()
    genres = build_genres(movies)
    novelty = build_novelty(train, len(genres))

    for n in ns:
        train_n = truncate(train, users, n)
        first = train_n[train_n["user_idx"].isin(users)]
        liked = first[first["rating"] >= 4.0].groupby("user_idx").size()
        no_liked = 1 - len(liked.reindex(users).dropna()) / len(users)
        print(f"\n##### first {n} ratings: {no_liked:.1%} of users have NO rating >= 4 in that history")
        sub = out_dir / f"cold_first{n}"
        sub.mkdir(parents=True, exist_ok=True)
        for name in systems:
            t0 = time.time()
            print(f"=== {LABELS[name]} (first {n}) ===")
            lists = RUNNERS[name](train_n, users, seen, args)
            per_user = score_lists(lists, relevant, genres, novelty)
            per_user.to_csv(sub / f"per_user_{name}.csv", index=False)
            pd.DataFrame(
                [(u, r + 1, m) for u, l in lists.items() for r, m in enumerate(l[:K])],
                columns=["user_idx", "rank", "movie_idx"],
            ).to_parquet(sub / f"top100_{name}.parquet", index=False)
            s = summarize(per_user)
            print(f"  NDCG@10 {s['ndcg@10']:.4f}  Recall@100 {s['recall@100']:.4f}  ({time.time() - t0:.0f}s)")

    write_summary(out_dir, systems, ns, set(cfg["heavy"]))
    print(f"\nSaved to {out_dir}/cold_first*/ and {out_dir}/coldstart_summary.md")


if __name__ == "__main__":
    main()
