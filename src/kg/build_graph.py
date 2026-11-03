"""
Building the movie knowledge graph from metadata.parquet.

Usage:
    python -m src.kg.build_graph

Outputs:
    data/kg/kg.pkl            graph object used by src/kg/retrieval.py
    data/kg/triples.parquet   every edge as (head, relation, tail)
"""
import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

RELATIONS = ["DIRECTED_BY", "STARS", "HAS_GENRE", "HAS_KEYWORD"]
ENTITY_TYPE = {
    "DIRECTED_BY": "person",
    "STARS": "person",
    "HAS_GENRE": "genre",
    "HAS_KEYWORD": "keyword",
}

@dataclass
class KnowledgeGraph:
    n_movies: int
    movie_titles: dict          # movie_idx to title
    triples: pd.DataFrame       # movie_idx, relation, entity_id, entity_name, order
    features: pd.DataFrame      # one row per (relation, entity): feature_id, relation, entity_id, entity_name, degree
    A: sparse.csr_matrix        # movies x features, 1 if the movie has that edge

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

def _as_list(x):
    """Parquet list columns as numpy arrays, missing values as None/NaN"""
    if x is None:
        return []
    if isinstance(x, float):
        return []
    if isinstance(x, str):
        return [x] if x.strip() else []
    return [s for s in list(x) if isinstance(s, str) and s.strip()]

def build_kg(metadata_path):
    md = pd.read_parquet(
        metadata_path,
        columns=["movie_idx", "title", "genres", "director", "directors", "cast", "keywords"],
    )

    rows = []  # (movie_idx, relation, raw_name, order)
    for r in md.itertuples(index=False):
        m = int(r.movie_idx)

        directors = _as_list(r.directors) or _as_list(r.director)
        for i, name in enumerate(directors):
            rows.append((m, "DIRECTED_BY", name, i))

        for i, name in enumerate(_as_list(r.cast)):
            rows.append((m, "STARS", name, i))

        genres = r.genres.split("|") if isinstance(r.genres, str) else []
        for i, g in enumerate(genres):
            if g and g != "(no genres listed)":
                rows.append((m, "HAS_GENRE", g, i))

        for i, kw in enumerate(_as_list(r.keywords)):
            rows.append((m, "HAS_KEYWORD", kw, i))

    triples = pd.DataFrame(rows, columns=["movie_idx", "relation", "entity_name", "order"])
    triples["entity_name"] = triples["entity_name"].str.strip()
    # same person is one node whether they direct or act.
    triples["entity_id"] = (
        triples["relation"].map(ENTITY_TYPE) + ":" + triples["entity_name"].str.lower()
    )
    triples = triples.drop_duplicates(["movie_idx", "relation", "entity_id"]).reset_index(drop=True)

    # a "feature" is a (relation, entity) pair, directing a film and acting in it are different evidence, so they get separate columns and separate weights in retrieval
    features = (
        triples.groupby(["relation", "entity_id"], sort=True)
        .agg(entity_name=("entity_name", "first"), degree=("movie_idx", "nunique"))
        .reset_index()
    )
    features.insert(0, "feature_id", np.arange(len(features)))
    triples = triples.merge(features[["relation", "entity_id", "feature_id"]], on=["relation", "entity_id"])

    n_movies = int(md["movie_idx"].max()) + 1
    A = sparse.csr_matrix(
        (np.ones(len(triples), dtype=np.float32), (triples["movie_idx"], triples["feature_id"])),
        shape=(n_movies, len(features)),
    )

    return KnowledgeGraph(
        n_movies=n_movies,
        movie_titles=dict(zip(md["movie_idx"].astype(int), md["title"])),
        triples=triples,
        features=features,
        A=A,
    )

def print_stats(kg):
    t, f = kg.triples, kg.features
    print(f"Movies:        {kg.n_movies:,}")
    for etype in ["person", "genre", "keyword"]:
        n = t.loc[t["entity_id"].str.startswith(etype + ":"), "entity_id"].nunique()
        print(f"{etype.capitalize() + ' nodes:':<15}{n:,}")
    print("Edges by relation:")
    for rel in RELATIONS:
        print(f"  {rel:<12} {int((t['relation'] == rel).sum()):,}")
    no_edges = int((np.diff(kg.A.indptr) == 0).sum())
    print(f"Movies with no edges: {no_edges:,}")
    useful = f[f["degree"] >= 2]
    print(f"Features linking 2+ movies (usable for retrieval): {len(useful):,} / {len(f):,}")
    print("Highest-degree entities (these get down-weighted):")
    for r in f.sort_values("degree", ascending=False).head(5).itertuples():
        print(f"  {r.relation:<12} {r.entity_name:<25} {r.degree:,} movies")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="data/processed/metadata.parquet")
    p.add_argument("--out_dir", default="data/kg")
    args = p.parse_args()

    kg = build_kg(args.metadata)
    out = Path(args.out_dir)
    kg.save(out / "kg.pkl")

    tr = kg.triples.copy()
    tr["head"] = "movie:" + tr["movie_idx"].astype(str)
    tr[["head", "relation", "entity_id", "entity_name", "order"]].rename(
        columns={"entity_id": "tail"}
    ).to_parquet(out / "triples.parquet", index=False)

    print_stats(kg)
    print(f"\nSaved {out / 'kg.pkl'} and {out / 'triples.parquet'}")

if __name__ == "__main__":
    main()