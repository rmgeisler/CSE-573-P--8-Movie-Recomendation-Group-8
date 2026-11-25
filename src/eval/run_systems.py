import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import METRIC_COLS, build_genres, build_novelty, score_lists, summarize
from src.models.als import ALSModel
from src.models.item_knn import ItemKNN
from src.models.popularity import PopularityModel

from src.kg.build_graph import KnowledgeGraph

K = 100
ALL_SYSTEMS = ["popularity", "als", "knn", "kg"]
LABELS = {"popularity": "Popularity", "als": "ALS (64f, reg 0.01)",
          "knn": "Item-KNN (k=50)", "kg": "KG retriever"}


def run_popularity(train, users, seen, args):
    m = PopularityModel()
    m.fit(train)
    return {u: m.recommend(seen[u], k=K) for u in users}


def run_als(train, users, seen, args):
    m = ALSModel(factors=args.als_factors, regularization=args.als_reg, iterations=20)
    m.fit(train)
    out = {}
    for u in users:

        ids, _ = m.model.recommend(int(u), m.user_movie_matrix[int(u)], N=K,
                                   filter_already_liked_items=True,
                                   filter_items=np.array(sorted(seen[u]), dtype=np.int32))
        out[u] = ids.tolist()
    return out


def run_knn(train, users, seen, args):
    m = ItemKNN(neighbors=args.knn_neighbors)
    m.fit(train)
    return {u: [int(x) for x in m.recommend(int(u), seen[u], k=K)] for u in users}


def run_kg(train, users, seen, args):
    from src.kg.retrieval import KGRetriever, load_params

    kg = KnowledgeGraph.load(args.kg)
    params = load_params(args.kg_params)
    train_s = train[train["user_idx"].isin(users)]
    m = KGRetriever(kg, **params).fit(train_s, pop_source=train)
    return {u: [int(x) for x in m.recommend(int(u), seen[u], k=K)] for u in users}


RUNNERS = {"popularity": run_popularity, "als": run_als, "knn": run_knn, "kg": run_kg}


def write_summary(out_dir):
    rows = {}
    for f in sorted(out_dir.glob("per_user_*.csv")):
        name = f.stem.replace("per_user_", "")
        rows[LABELS.get(name, name)] = summarize(pd.read_csv(f))
    if not rows:
        return
    df = pd.DataFrame(rows).T
    df["n_users"] = df["n_users"].astype(int)
    df["n_recs"] = df["n_recs"].round(1)
    text = df.round(4).to_markdown()
    (out_dir / "summary.md").write_text(text + "\n")
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--users", choices=["main", "llm", "heavy"], default="main")
    ap.add_argument("--limit", type=int, default=0, help="only use the first N users (dry runs)")
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--users-file", default="data/eval/eval_users.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--kg", default="data/kg/kg.pkl")
    ap.add_argument("--kg-params", default="tests/results/kg_best_params.json")
    ap.add_argument("--als-factors", type=int, default=64)
    ap.add_argument("--als-reg", type=float, default=0.01)
    ap.add_argument("--knn-neighbors", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out or f"results/{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = Path(args.proc)

    train = pd.read_parquet(proc / "train.parquet", columns=["user_idx", "movie_idx", "rating", "timestamp"])
    split = pd.read_parquet(proc / f"{args.split}.parquet", columns=["user_idx", "movie_idx", "rating"])
    movies = pd.read_parquet(proc / "movies.parquet", columns=["movie_idx", "genres"])

    cfg = json.loads(Path(args.users_file).read_text())
    users = cfg[args.users]
    if args.limit:
        users = users[: args.limit]
    users = np.array(users, dtype=np.int64)

    rel_all = split[split["rating"] >= cfg["relevance_threshold"]].groupby("user_idx")["movie_idx"].apply(set)
    relevant = {int(u): rel_all[u] for u in users if u in rel_all.index}
    users = np.array(sorted(relevant), dtype=np.int64)
    print(f"split={args.split}  user group={args.users}  scoring {len(users):,} users "
          f"(users without a relevant item in this split are skipped)")

    seen = (train[train["user_idx"].isin(users)].groupby("user_idx")["movie_idx"].apply(set).to_dict())
    genres = build_genres(movies)
    novelty = build_novelty(train, len(genres))

    for name in args.systems.split(","):
        name = name.strip()
        t0 = time.time()
        print(f"\n=== {LABELS[name]} ===")
        lists = RUNNERS[name](train, users, seen, args)
        per_user = score_lists(lists, relevant, genres, novelty)
        per_user.to_csv(out_dir / f"per_user_{name}.csv", index=False)
        pd.DataFrame(
            [(u, r + 1, m) for u, l in lists.items() for r, m in enumerate(l[:K])],
            columns=["user_idx", "rank", "movie_idx"],
        ).to_parquet(out_dir / f"top100_{name}.parquet", index=False)
        s = summarize(per_user)
        print(f"  NDCG@10 {s['ndcg@10']:.4f}  P@10 {s['precision@10']:.4f}  "
              f"Recall@100 {s['recall@100']:.4f}  ({time.time() - t0:.0f}s)")

    write_summary(out_dir)
    print(f"\nSaved results to {out_dir}/")


if __name__ == "__main__":
    main()
