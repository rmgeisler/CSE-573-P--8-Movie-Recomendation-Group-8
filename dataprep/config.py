"""Central configuration. Every tunable lives here so the manifest can record them."""


from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"                          # everything is found under here
RAW_DIR = ROOT / "data" / "raw" / "ml-20m"        # legacy hint; discovery is used instead
TMDB_CACHE_DIR = ROOT / "data" / "raw" / "tmdb"   # one JSON per movie, sharded
KAGGLE_DIR = ROOT / "data" / "raw" / "the-movies-dataset"  # movies_metadata.csv etc.
PROCESSED_DIR = ROOT / "data" / "processed"       # everything this pipeline writes
REPORT_DIR = ROOT / "reports"                     # coverage, split diagnostics

# ---------------------------------------------------------------- filtering
# MovieLens 20M already guarantees >= 20 ratings per user, but we re-apply the
# threshold because dropping sparse MOVIES can push some users below it.
MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 5
MAX_FILTER_ROUNDS = 10   # filtering is iterative; it converges in 2-3 in practice

# ---------------------------------------------------------------- metadata
TOP_CAST_N = 5           # keep only top-billed actors; the long tail is noise
MAX_KEYWORDS = 10

# ---------------------------------------------------------------- split
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10
MIN_RATINGS_FOR_SPLIT = 3          # need at least one row per split
RELEVANCE_THRESHOLD = 4.0          # held-out rating >= this counts as relevant

# ---------------------------------------------------------------- TMDB API
TMDB_BASE = "https://api.themoviedb.org/3/movie/{tmdb_id}"
TMDB_APPEND = "credits,keywords"
TMDB_SLEEP = 0.05        # seconds between requests; raise if you see 429s
TMDB_MAX_RETRIES = 4
TMDB_SHARDS = 100        # subdirectories, so no folder holds 27k files


# The metadata table's canonical column order. Both metadata routes reindex to
# this before saving, so "which source produced this file" never changes the
# schema that downstream code reads.
METADATA_COLUMNS = [
    "movie_idx", "movie_id", "title", "title_clean", "year", "genres",
    "imdb_id", "tmdb_id", "tmdb_title", "overview", "release_date",
    "tmdb_year", "runtime", "popularity", "vote_average", "vote_count",
    "director", "directors", "cast", "tmdb_genres", "keywords",
    "metadata_complete", "has_metadata",
]


def set_data_dir(path):
    """Point the pipeline at a different data directory (used by --data-dir)."""
    global DATA_DIR, TMDB_CACHE_DIR
    DATA_DIR = Path(path).expanduser().resolve()
    TMDB_CACHE_DIR = DATA_DIR / "tmdb_cache"


def as_dict():
    """Everything the manifest should record about how the data was built."""
    return {
        "min_user_ratings": MIN_USER_RATINGS,
        "min_movie_ratings": MIN_MOVIE_RATINGS,
        "top_cast_n": TOP_CAST_N,
        "max_keywords": MAX_KEYWORDS,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "min_ratings_for_split": MIN_RATINGS_FOR_SPLIT,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "split_rule": "per-user chronological, ties broken by movie_id",
    }
