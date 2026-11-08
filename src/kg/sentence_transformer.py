"""
Plot-summary
Encodes every movie's plot overview with a sentence transformer
For a (user, candidate) pair, returns the movies in the user's train history whose plots are most similar to the
candidate's plot

Usage: python -m src.kg.sentence_transformer

Output: data/kg/plot_embeddings.npz   
one normalized vector per movie, plus titles
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_NAME = "all-MiniLM-L6-v2"


def build_embeddings(metadata_path, model_name=MODEL_NAME, batch_size=64):
    from sentence_transformers import SentenceTransformer

    md = pd.read_parquet(metadata_path, columns=["movie_idx", "title", "overview"])
    n_movies = int(md["movie_idx"].max()) + 1
    overview = md["overview"].fillna("").astype(str).str.strip()
    has_text = (overview.str.len() > 0).to_numpy()

    model = SentenceTransformer(model_name)
    vecs = model.encode(
        overview[has_text].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit length, so a dot product is cosine similarity
        convert_to_numpy=True,
    ).astype(np.float32)

    embeddings = np.zeros((n_movies, vecs.shape[1]), dtype=np.float32)
    embeddings[md.loc[has_text, "movie_idx"].to_numpy()] = vecs
    has_overview = np.zeros(n_movies, dtype=bool)
    has_overview[md.loc[has_text, "movie_idx"].to_numpy()] = True
    titles = np.full(n_movies, "", dtype=object)
    titles[md["movie_idx"].to_numpy()] = md["title"].to_numpy()
    return embeddings, has_overview, titles.astype(str)


class SemanticNeighbors:
    def __init__(self, embeddings, has_overview, titles):
        self.embeddings = embeddings
        self.has_overview = has_overview
        self.titles = titles

    @classmethod
    def load(cls, path="data/kg/plot_embeddings.npz"):
        d = np.load(path)
        return cls(d["embeddings"], d["has_overview"], d["titles"])

    def fit(self, train):
        """train: the TRAIN split (user_idx, movie_idx, rating). Only these ratings are searched."""
        df = train[["user_idx", "movie_idx", "rating"]]
        df = df[self.has_overview[df["movie_idx"].to_numpy()]].sort_values("user_idx")
        users, starts = np.unique(df["user_idx"].to_numpy(), return_index=True)
        movies = np.split(df["movie_idx"].to_numpy(), starts[1:])
        ratings = np.split(df["rating"].to_numpy(), starts[1:])
        self.history = dict(zip(users.tolist(), zip(movies, ratings)))
        return self

    def neighbors(self, user_idx, movie_idx, k=3, min_rating=None):
        """
        The k movies the user rated (in train) whose plots are most similar to movie_idx.
        min_rating: only search movies the user rated at least this high (default: all rated).
        Returns [] if the candidate has no overview or the user has no history.
        """
        if not self.has_overview[movie_idx] or user_idx not in self.history:
            return []
        movies, ratings = self.history[user_idx]
        keep = movies != movie_idx
        if min_rating is not None:
            keep &= ratings >= min_rating
        movies, ratings = movies[keep], ratings[keep]
        if len(movies) == 0:
            return []
        sims = self.embeddings[movies] @ self.embeddings[movie_idx]
        top = np.argsort(-sims, kind="stable")[:k]
        return [
            {
                "movie_idx": int(movies[i]),
                "title": str(self.titles[movies[i]]),
                "rating": float(ratings[i]),
                "similarity": round(float(sims[i]), 4),
            }
            for i in top
        ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="data/processed/metadata.parquet")
    p.add_argument("--out", default="data/kg/plot_embeddings.npz")
    p.add_argument("--model", default=MODEL_NAME)
    args = p.parse_args()

    embeddings, has_overview, titles = build_embeddings(args.metadata, args.model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, embeddings=embeddings, has_overview=has_overview, titles=titles)
    print(f"Encoded {has_overview.sum():,} of {len(has_overview):,} movies "
          f"({embeddings.shape[1]}-dim, model {args.model})")
    print(f"Saved {args.out}")

    # sanity check: movies with the most similar plots to the first movie
    q = int(np.argmax(has_overview))
    sims = embeddings @ embeddings[q]
    sims[q] = -1
    print(f"\nMost similar plots to {titles[q]}:")
    for m in np.argsort(-sims)[:5]:
        print(f"  {sims[m]:.3f}  {titles[m]}")


if __name__ == "__main__":
    main()