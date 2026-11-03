import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  
from src.kg.build_graph import KnowledgeGraph  
from src.kg.retrieval import DEFAULT_REL_WEIGHTS, KGRetriever  

try:  
    from src.evaluation import ndcg_at_k, precision_at_k, recall_at_k  
except ImportError:
    def recall_at_k(recs, relevant, k):
        return sum(m in relevant for m in recs[:k]) / len(relevant)

    def precision_at_k(recs, relevant, k):
        return sum(m in relevant for m in recs[:k]) / k

    def ndcg_at_k(recs, relevant, k):
        dcg = sum(1 / np.log2(r + 2) for r, m in enumerate(recs[:k]) if m in relevant)
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(relevant), k)))
        return dcg / idcg if idcg > 0 else 0.0

GRID = {
    "pop_alpha": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
    "smoothing": [0.0, 0.02, 0.05, 0.1, 0.2, 0.5],
    "normalize_candidates": [True, False],
    "DIRECTED_BY": [0.5, 1.0, 2.0],
    "STARS": [0.3, 0.6, 1.2],
    "HAS_KEYWORD": [0.25, 0.5, 1.0],
    "HAS_GENRE": [0.05, 0.15, 0.3, 0.6],
    "backfill": [False, True],
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kg", default="data/kg/kg.pkl")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--n_users", type=int, default=5000, help="0 = all validation users")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default="tests/results")
    p.add_argument("--pop_alpha", type=float, default=None,
                   help="fix pop_alpha to this value instead of searching it")
    p.add_argument("--include_new_users", action="store_true",
                   help="also score each trial with only the first 5 ratings and average the two")
    args = p.parse_args()
    d = Path(args.data)

    kg = KnowledgeGraph.load(args.kg)
    train = pd.read_parquet(d / "train.parquet")
    val = pd.read_parquet(d / "val.parquet", columns=["user_idx", "movie_idx", "rating"])

    relevant_val = val[val["rating"] >= 4.0].groupby("user_idx")["movie_idx"].apply(set).to_dict()
    users = np.array(sorted(relevant_val))
    if args.n_users:
        rng = np.random.default_rng(args.seed)
        users = np.sort(rng.choice(users, size=min(args.n_users, len(users)), replace=False))
    train_s = train[train["user_idx"].isin(users)]
    print(f"Tuning on {len(users):,} validation users")

    def evaluate(settings, max_history=None):
        kwargs = {
            "pop_alpha": settings["pop_alpha"],
            "smoothing": settings["smoothing"],
            "normalize_candidates": settings["normalize_candidates"],
            "backfill": settings["backfill"],
            "rel_weights": {r: settings[r] for r in DEFAULT_REL_WEIGHTS},
        }
        model = KGRetriever(kg, **kwargs).fit(train_s, max_history=max_history, pop_source=train)
        cands = model.recommend_batch(users, k=100, verbose=False)
        lists = cands.groupby("user_idx")["movie_idx"].apply(list).to_dict()
        rec, prec, ndcg = [], [], []
        for u in users:
            recs, rel = lists.get(u, []), relevant_val[u]
            rec.append(recall_at_k(recs, rel, k=100))
            prec.append(precision_at_k(recs, rel, k=100))
            ndcg.append(ndcg_at_k(recs, rel, k=10))
        return kwargs, {"recall@100": np.mean(rec), "precision@100": np.mean(prec),
                        "ndcg@10": np.mean(ndcg)}

    grid = dict(GRID)
    start = {"pop_alpha": 0.0, "smoothing": 0.0, "normalize_candidates": True, "backfill": False,
             **DEFAULT_REL_WEIGHTS}
    if args.pop_alpha is not None:
        grid.pop("pop_alpha")
        start["pop_alpha"] = args.pop_alpha
        print(f"pop_alpha fixed at {args.pop_alpha}")
    best = dict(start)
    results, cache = [], {}

    def run_trial(trial):
        key = json.dumps(trial, sort_keys=True)
        if key not in cache:
            full = evaluate(trial)[1]
            if args.include_new_users:
                cold5 = evaluate(trial, max_history=5)[1]
                cache[key] = {**full, "recall@100_first5": cold5["recall@100"],
                              "ndcg@10_first5": cold5["ndcg@10"],
                              "objective": (full["recall@100"] + cold5["recall@100"]) / 2}
            else:
                cache[key] = {**full, "objective": full["recall@100"]}
            return cache[key], True
        return cache[key], False

    for name, values in grid.items():
        print(f"\nTesting {name}...")
        stage = []
        for v in values:
            trial = {**best, name: v}
            t0 = time.time()
            m, new = run_trial(trial)
            if new:
                line = (f"  {name}={v!s:<6} Recall@100: {m['recall@100']:.4f}  "
                        f"Precision@100: {m['precision@100']:.4f}  NDCG@10: {m['ndcg@10']:.4f}")
                if args.include_new_users:
                    line += (f"  | first 5 ratings: Recall@100 {m['recall@100_first5']:.4f}"
                             f"  | objective {m['objective']:.4f}")
                print(line + f"  ({time.time() - t0:.0f}s)")
                results.append({"stage": name, **trial, **m})
            stage.append((m["objective"], v))
        best[name] = max(stage, key=lambda x: x[0])[1]
        print(f"  -> best {name} = {best[name]}")

    best_kwargs, best_full = evaluate(best)
    cold = {h: evaluate(best, max_history=h)[1] for h in (10, 5)}
    baseline, _ = run_trial(start)

    results_df = pd.DataFrame(results).sort_values("objective", ascending=False)
    objective_desc = ("mean of full and first-5 Recall@100" if args.include_new_users
                      else "Recall@100, full history")
    print(f"\nKG Tuning Results (sorted by {objective_desc}):")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest configuration: {best_kwargs}")
    print(f"  Full history   Recall@100 {best_full['recall@100']:.4f}  NDCG@10 {best_full['ndcg@10']:.4f}"
          f"   (starting settings: {baseline['recall@100']:.4f} / {baseline['ndcg@10']:.4f})")
    for h, m in cold.items():
        print(f"  First {h:<2} ratings Recall@100 {m['recall@100']:.4f}  NDCG@10 {m['ndcg@10']:.4f}")

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "kg_best_params.json", "w") as f:
        json.dump(best_kwargs, f, indent=2)
    with open(out / "kg_tuning_results.md", "w") as f:
        f.write("# KG Retriever Tuning Results\n\n")
        f.write(f"Validation split, {len(users):,} users (seed {args.seed}). "
                f"Coordinate search; selection metric = {objective_desc}.\n\n")
        f.write("## Best configuration\n\n```json\n" + json.dumps(best_kwargs, indent=2) + "\n```\n\n")
        f.write("| History | Recall@100 | Precision@100 | NDCG@10 |\n|---|---|---|---|\n")
        f.write(f"| Starting settings (full) | {baseline['recall@100']:.4f} | "
                f"{baseline['precision@100']:.4f} | {baseline['ndcg@10']:.4f} |\n")
        for label, m in [("Full", best_full), ("First 10 ratings", cold[10]), ("First 5 ratings", cold[5])]:
            f.write(f"| {label} | {m['recall@100']:.4f} | {m['precision@100']:.4f} | {m['ndcg@10']:.4f} |\n")
        f.write("\n## All trials\n\n")
        f.write(results_df.to_markdown(index=False, floatfmt=".4f") if _has_tabulate()
                else "```\n" + results_df.to_string(index=False) + "\n```\n")
    print(f"\nSaved {out / 'kg_tuning_results.md'} and {out / 'kg_best_params.json'}")


def _has_tabulate():
    try:
        import tabulate  
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()