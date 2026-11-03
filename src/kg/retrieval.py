"""
Knowledge-graph path retrieval.

Usage to write candidates for every user with the tuned settings:
    python -m src.kg.retrieval --params tests/results/kg_best_params.json --evidence_users 500
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

from src.kg.build_graph import KnowledgeGraph

# starting points, tests/kg/tune_kg.py searches these on the validation split
DEFAULT_REL_WEIGHTS = {
    "DIRECTED_BY": 1.0,
    "STARS": 0.6,
    "HAS_KEYWORD": 0.5,
    "HAS_GENRE": 0.15,
}


def load_params(path):
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


class KGRetriever:
    def __init__(self, kg, rel_weights=None, min_rating=4.0, normalize_candidates=True,
                 pop_alpha=0.0, smoothing=0.0, backfill=False):
        self.kg = kg
        self.rel_weights = {**DEFAULT_REL_WEIGHTS, **(rel_weights or {})}
        self.min_rating = min_rating
        self.normalize_candidates = normalize_candidates
        self.pop_alpha = pop_alpha
        self.smoothing = smoothing
        self.backfill = backfill

        f = kg.features
        idf = np.log(kg.n_movies / f["degree"].to_numpy())
        rel_w = f["relation"].map(self.rel_weights).fillna(0.0).to_numpy()
        w = (rel_w * idf).astype(np.float32)
        w[f["degree"].to_numpy() < 2] = 0.0  # an entity on only one movie can't link two movies
        self.feature_weight = w

        self.A = kg.A.tocsr()
        self.A_csc = kg.A.tocsc()
        n_feats = np.diff(self.A.indptr).astype(np.float32)
        self.cand_norm = (1.0 / np.sqrt(np.maximum(n_feats, 1.0))).astype(np.float32) \
            if normalize_candidates else np.ones(kg.n_movies, dtype=np.float32)

    # fit
    def fit(self, train, max_history=None, pop_source=None):
        """
        train:       DataFrame with user_idx, movie_idx, rating, timestamp (the TRAIN split).
        max_history: keep only each user's first N ratings by time (simulated cold start).
        pop_source:  ratings used to count popularity (defaults to `train` before truncation).
                     Pass the FULL train split when `train` is only a sample of users.
        """
        n_movies = self.kg.n_movies
        pop_df = train if pop_source is None else pop_source
        counts = pop_df.loc[pop_df["rating"] >= self.min_rating, "movie_idx"].value_counts()
        self.popularity = np.zeros(n_movies, dtype=np.float32)
        self.popularity[counts.index.to_numpy()] = counts.to_numpy()
        self.pop_order = np.argsort(-self.popularity, kind="stable")
        self.pop_order = self.pop_order[self.popularity[self.pop_order] > 0]
        self.pop_prior = ((1.0 + self.popularity) ** self.pop_alpha).astype(np.float32)

        df = train[["user_idx", "movie_idx", "rating", "timestamp"]]
        if max_history is not None:
            df = (
                df.sort_values(["user_idx", "timestamp", "movie_idx"])
                .groupby("user_idx", sort=False)
                .head(max_history)
            )
        n_users = int(df["user_idx"].max()) + 1

        # Seeds: liked movies, weight 1 (rating 4) to 2 (rating 5)
        # user with no rating >= 4 falls back to their highest rated movies
        user_max = df.groupby("user_idx")["rating"].transform("max")
        liked = df["rating"] >= self.min_rating
        fallback = (user_max < self.min_rating) & (df["rating"] == user_max)
        seeds = df[liked | fallback]
        seed_w = np.where(
            seeds["rating"] >= self.min_rating, seeds["rating"] - (self.min_rating - 1), 0.5
        ).astype(np.float32)

        self.S = sparse.csr_matrix(
            (seed_w, (seeds["user_idx"], seeds["movie_idx"])), shape=(n_users, n_movies)
        )
        self.H = sparse.csr_matrix(
            (np.ones(len(df), dtype=np.int8), (df["user_idx"], df["movie_idx"])),
            shape=(n_users, n_movies),
        )
        return self

    # scoring
    def _path_scores(self, user_ids):
        E = self.S[np.asarray(user_ids)] @ self.A           
        E = E.multiply(self.feature_weight).tocsr()         # weight by relation and rarity
        return (E @ self.A.T).toarray().astype(np.float32)  # back to movies

    def _score_users(self, user_ids):
        path = self._path_scores(user_ids)
        graph = path * self.cand_norm
        top = graph.max(axis=1, keepdims=True)
        graph = np.divide(graph, top, out=np.zeros_like(graph), where=top > 0)
        return (self.smoothing + graph) * self.pop_prior, path

    def _rank(self, row, path_row, excluded, k):
        row[excluded] = -np.inf
        n_pos = int((row > 0).sum())
        kk = min(k, n_pos)
        if kk:
            idx = np.argpartition(-row, kk - 1)[:kk]
            top = idx[np.argsort(-row[idx], kind="stable")]
        else:
            top = np.array([], dtype=np.int64)
        sources = ["kg" if path_row[m] > 0 else "popular_prior" for m in top]
        if self.backfill and len(top) < k:
            taken = set(top.tolist()) | set(np.asarray(excluded).tolist())
            extra = []
            for m in self.pop_order:
                if m not in taken:
                    extra.append(m)
                    if len(top) + len(extra) == k:
                        break
            top = np.concatenate([top, np.array(extra, dtype=np.int64)])
            sources += ["popular_backfill"] * len(extra)
        return top, sources

    def recommend(self, user_idx, seen=None, k=100):
        if user_idx >= self.S.shape[0]:
            return []
        scores, path = self._score_users([user_idx])
        excluded = set(self.H[user_idx].indices.tolist()) | set(seen or [])
        top, _ = self._rank(scores[0], path[0], np.fromiter(excluded, dtype=np.int64), k)
        return top.tolist()

    def recommend_batch(self, user_ids, k=100, batch_size=1024, verbose=True):
        out = []
        user_ids = np.asarray(user_ids)
        for start in range(0, len(user_ids), batch_size):
            batch = user_ids[start:start + batch_size]
            scores, path = self._score_users(batch)
            H = self.H[batch]
            for i, u in enumerate(batch):
                excluded = H.indices[H.indptr[i]:H.indptr[i + 1]]
                top, sources = self._rank(scores[i], path[i], excluded, k)
                out.append(pd.DataFrame({
                    "user_idx": u,
                    "movie_idx": top,
                    "rank": np.arange(1, len(top) + 1),
                    "kg_score": np.where(np.array(sources) == "popular_backfill", 0.0,
                                         scores[i, top]) if len(top) else np.array([]),
                    "source": sources,
                }))
            if verbose:
                print(f"  scored {min(start + batch_size, len(user_ids)):,}/{len(user_ids):,} users",
                      end="\r")
        if verbose:
            print()
        return pd.concat(out, ignore_index=True)

    # evidence
    def explain(self, user_idx, movie_idx, max_paths=5, max_seeds_per_path=3):
        """
        Paths connecting the user's liked movies to a candidate
        Returns [] for movies with source "popular_prior" or "popular_backfill"
        """
        seeds = self.S[user_idx]
        seed_w = dict(zip(seeds.indices, seeds.data))
        cand_feats = self.A[movie_idx].indices
        titles = self.kg.movie_titles
        feats = self.kg.features

        paths = []
        for f in cand_feats:
            w = self.feature_weight[f]
            if w == 0:
                continue
            col = self.A_csc[:, f].indices
            shared = [(s, seed_w[s]) for s in col if s in seed_w and s != movie_idx]
            if not shared:
                continue
            shared.sort(key=lambda x: -x[1])
            meta = feats.iloc[f]
            paths.append({
                "relation": meta["relation"],
                "entity_id": meta["entity_id"],
                "entity_name": meta["entity_name"],
                "contribution": round(float(w * sum(sw for _, sw in shared)), 4),
                "seed_movies": [
                    {"movie_idx": int(s), "title": titles.get(int(s), "")}
                    for s, _ in shared[:max_seeds_per_path]
                ],
                "n_seed_movies": len(shared),
            })
        paths.sort(key=lambda p: -p["contribution"])
        return paths[:max_paths]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kg", default="data/kg/kg.pkl")
    p.add_argument("--train", default="data/processed/train.parquet")
    p.add_argument("--params", default=None, help="tuned settings JSON from tests/kg/tune_kg.py")
    p.add_argument("--out_dir", default="data/kg")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--evidence_users", type=int, default=0,
                   help="also write path evidence for this many random users (for the LLM stage)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    kg = KnowledgeGraph.load(args.kg)
    train = pd.read_parquet(args.train)
    params = load_params(args.params)
    print(f"Settings: {params or 'defaults'}")
    model = KGRetriever(kg, **params).fit(train)

    users = np.sort(train["user_idx"].unique())
    print(f"Generating top-{args.k} KG candidates for {len(users):,} users")
    cands = model.recommend_batch(users, k=args.k)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cands.to_parquet(out / "kg_candidates.parquet", index=False)
    print(f"Saved {out / 'kg_candidates.parquet'} ({len(cands):,} rows, "
          f"{(cands['source'] == 'kg').mean():.1%} with graph paths)")

    if args.evidence_users:
        rng = np.random.default_rng(args.seed)
        sample = rng.choice(users, size=min(args.evidence_users, len(users)), replace=False)
        sub = cands[cands["user_idx"].isin(sample)]
        with open(out / "kg_evidence_sample.jsonl", "w") as f:
            for r in sub.itertuples(index=False):
                f.write(json.dumps({
                    "user_idx": int(r.user_idx),
                    "movie_idx": int(r.movie_idx),
                    "rank": int(r.rank),
                    "kg_score": float(r.kg_score),
                    "source": r.source,
                    "paths": model.explain(int(r.user_idx), int(r.movie_idx)) if r.source == "kg" else [],
                }) + "\n")
        print(f"Saved {out / 'kg_evidence_sample.jsonl'} ({len(sub):,} candidate records)")


if __name__ == "__main__":
    main()