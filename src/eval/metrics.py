from itertools import combinations

import numpy as np
import pandas as pd

K = 10
POOL_K = 100
METRIC_COLS = ["ndcg@10", "hr@10", "mrr@10", "precision@10", "recall@10",
               "recall@100", "ild@10", "novelty@10", "n_recs"]


def build_genres(movies):
    n = int(movies["movie_idx"].max()) + 1
    out = [frozenset()] * n
    for idx, g in zip(movies["movie_idx"], movies["genres"]):
        if isinstance(g, str) and g and g != "(no genres listed)":
            out[int(idx)] = frozenset(g.split("|"))
    return out


def build_novelty(train, n_movies):
    n_users = train["user_idx"].nunique()
    counts = np.zeros(n_movies, dtype=np.float64)
    c = train["movie_idx"].value_counts()
    counts[c.index.to_numpy()] = c.to_numpy()
    return -np.log2((counts + 1.0) / (n_users + 1.0))


def intra_list_diversity(movies, genres):
    dists = []
    for a, b in combinations(movies, 2):
        ga, gb = genres[a], genres[b]
        union = ga | gb
        if union:
            dists.append(1.0 - len(ga & gb) / len(union))
    return float(np.mean(dists)) if dists else np.nan


def user_metrics(recs, relevant, genres=None, novelty=None):
    recs = [int(m) for m in recs][:POOL_K]
    top = recs[:K]
    n_rel = len(relevant)
    hit = [m in relevant for m in top]
    hits = sum(hit)
    dcg = sum(1.0 / np.log2(i + 2) for i, h in enumerate(hit) if h)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_rel, K)))
    out = {
        "ndcg@10": dcg / idcg if idcg > 0 else 0.0,
        "hr@10": float(hits > 0),
        "mrr@10": next((1.0 / (i + 1) for i, h in enumerate(hit) if h), 0.0),
        "precision@10": hits / K,
        "recall@10": hits / n_rel,
        "recall@100": sum(m in relevant for m in recs) / n_rel,
        "ild@10": intra_list_diversity(top, genres) if genres is not None else np.nan,
        "novelty@10": float(np.mean([novelty[m] for m in top])) if (novelty is not None and top) else np.nan,
        "n_recs": len(recs),
    }
    return out


def score_lists(lists, relevant, genres=None, novelty=None):
    rows = []
    for u, rel in relevant.items():
        row = user_metrics(lists.get(u, []), rel, genres, novelty)
        row["user_idx"] = u
        rows.append(row)
    return pd.DataFrame(rows)[["user_idx"] + METRIC_COLS]


def summarize(per_user):
    s = per_user[METRIC_COLS].mean(numeric_only=True)
    s["n_users"] = len(per_user)
    return s
