"""Step 1 - load MovieLens, clean it, and remap IDs to contiguous integers.

Produces:
    processed/ratings          user_idx, movie_idx, rating, timestamp
    processed/movies           movie_idx, movie_id, title, year, genres
    processed/links            movie_idx, movie_id, imdb_id, tmdb_id
    processed/user_map         user_idx, user_id
    reports/step1_stats.json
"""


from __future__ import annotations
import argparse
import re

import numpy as np
import pandas as pd

from . import config as C
from .discover import locate
from .io_utils import banner, load_table, log, save_table, write_json

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _extract_year(title: str):
    m = YEAR_RE.search(str(title).strip())
    return int(m.group(1)) if m else None


def _strip_year(title: str) -> str:
    return YEAR_RE.sub("", str(title).strip()).strip()


def load_raw():
    banner("STEP 1 - LOAD AND CLEAN RATINGS")

    ratings_path = locate("ratings.csv", prefer="movielens")
    movies_path = locate("movies.csv", prefer="movielens")
    links_path = locate("links.csv", prefer="movielens")
    for label, pth in (("ratings", ratings_path), ("movies", movies_path),
                       ("links", links_path)):
        log(f"  {label:<8} <- {pth}")

    ratings = pd.read_csv(
        ratings_path,
        dtype={"userId": np.int32, "movieId": np.int32,
               "rating": np.float32, "timestamp": np.int64},
    )
    movies = pd.read_csv(movies_path)
    links = pd.read_csv(links_path)

    # Sanity-check we got MovieLens 20M and not another ratings file. The
    # Kaggle archive ships a different ratings.csv with the same columns.
    if not 19_000_000 < len(ratings) < 21_000_000:
        log(f"  NOTE: expected ~20.0M ratings for ml-20m, got {len(ratings):,}.")
        log(f"        Check that {ratings_path} is the MovieLens 20M file.")

    log(f"  ratings : {len(ratings):>10,} rows")
    log(f"  movies  : {len(movies):>10,} rows")
    log(f"  links   : {len(links):>10,} rows")
    return ratings, movies, links


def deduplicate(ratings: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent rating when a user rated the same movie twice.

    Sorting by timestamp then taking the last occurrence is deliberate: a user
    who revised a rating meant the newer one.
    """
    before = len(ratings)
    ratings = (
        ratings.sort_values(["userId", "movieId", "timestamp"], kind="mergesort")
        .drop_duplicates(subset=["userId", "movieId"], keep="last")
    )
    removed = before - len(ratings)
    log(f"  deduplicated: removed {removed:,} duplicate (user, movie) pairs")
    return ratings


def iterative_filter(ratings: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Drop sparse movies, then sparse users, repeating until stable.

    One pass is not enough: removing rarely-rated movies can push a user below
    the user threshold, and removing those users can push a movie below the
    movie threshold. We loop until nothing changes and report every round.
    """
    history = []
    for round_no in range(1, C.MAX_FILTER_ROUNDS + 1):
        n_before = len(ratings)

        movie_counts = ratings["movieId"].value_counts()
        keep_movies = movie_counts[movie_counts >= C.MIN_MOVIE_RATINGS].index
        ratings = ratings[ratings["movieId"].isin(keep_movies)]

        user_counts = ratings["userId"].value_counts()
        keep_users = user_counts[user_counts >= C.MIN_USER_RATINGS].index
        ratings = ratings[ratings["userId"].isin(keep_users)]

        dropped = n_before - len(ratings)
        history.append({
            "round": round_no,
            "ratings_dropped": int(dropped),
            "ratings_remaining": int(len(ratings)),
            "users_remaining": int(ratings["userId"].nunique()),
            "movies_remaining": int(ratings["movieId"].nunique()),
        })
        log(f"  filter round {round_no}: dropped {dropped:,} -> "
            f"{len(ratings):,} ratings, {ratings['userId'].nunique():,} users, "
            f"{ratings['movieId'].nunique():,} movies")

        if dropped == 0:
            break
    else:
        log(f"  WARNING: filtering did not converge in {C.MAX_FILTER_ROUNDS} rounds")

    return ratings, history


def remap_ids(ratings: pd.DataFrame):
    """Map userId/movieId to contiguous 0..n-1 indices.

    Models index arrays and sparse matrices by position, so the raw IDs - which
    have gaps - cannot be used directly. We keep both mappings because results
    have to be translated back into titles for the demo and for explanations.
    """
    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())

    user_map = pd.DataFrame({
        "user_idx": np.arange(len(user_ids), dtype=np.int32),
        "user_id": user_ids.astype(np.int32),
    })
    movie_map = pd.DataFrame({
        "movie_idx": np.arange(len(movie_ids), dtype=np.int32),
        "movie_id": movie_ids.astype(np.int32),
    })

    u_lookup = dict(zip(user_map["user_id"], user_map["user_idx"]))
    m_lookup = dict(zip(movie_map["movie_id"], movie_map["movie_idx"]))

    out = pd.DataFrame({
        "user_idx": ratings["userId"].map(u_lookup).astype(np.int32),
        "movie_idx": ratings["movieId"].map(m_lookup).astype(np.int32),
        "rating": ratings["rating"].astype(np.float32),
        "timestamp": ratings["timestamp"].astype(np.int64),
    })

    log(f"  remapped to {len(user_map):,} users x {len(movie_map):,} movies")
    return out, user_map, movie_map


def run():
    ratings, movies, links = load_raw()
    ratings = deduplicate(ratings)
    ratings, filter_history = iterative_filter(ratings)
    ratings, user_map, movie_map = remap_ids(ratings)

    # Movie metadata, restricted to movies that survived filtering.
    movies = movies.rename(columns={"movieId": "movie_id"})
    movies["year"] = movies["title"].map(_extract_year)
    movies["title_clean"] = movies["title"].map(_strip_year)
    movies_out = movie_map.merge(movies, on="movie_id", how="left")
    movies_out = movies_out[
        ["movie_idx", "movie_id", "title", "title_clean", "year", "genres"]
    ]

    links = links.rename(
        columns={"movieId": "movie_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"}
    )
    links_out = movie_map.merge(links, on="movie_id", how="left")
    n_missing_tmdb = int(links_out["tmdb_id"].isna().sum())
    log(f"  movies with no tmdbId in links.csv: {n_missing_tmdb:,} "
        f"({100 * n_missing_tmdb / len(links_out):.2f}%)")

    save_table(ratings, C.PROCESSED_DIR / "ratings")
    save_table(movies_out, C.PROCESSED_DIR / "movies")
    save_table(links_out, C.PROCESSED_DIR / "links")
    save_table(user_map, C.PROCESSED_DIR / "user_map")

    n_users, n_movies = len(user_map), len(movie_map)
    density = len(ratings) / (n_users * n_movies)
    stats = {
        "n_ratings": int(len(ratings)),
        "n_users": n_users,
        "n_movies": n_movies,
        "density": density,
        "sparsity_pct": 100 * (1 - density),
        "movies_missing_tmdb_id": n_missing_tmdb,
        "filter_history": filter_history,
        "config": C.as_dict(),
    }
    write_json(stats, C.REPORT_DIR / "step1_stats.json")

    log("")
    log(f"  final matrix: {n_users:,} x {n_movies:,}, "
        f"{len(ratings):,} ratings, {100 * (1 - density):.2f}% empty")
    log(f"  wrote {C.PROCESSED_DIR}")
    return stats


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
