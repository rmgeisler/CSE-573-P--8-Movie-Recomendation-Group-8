"""Table I/O and hashing.

Writes Parquet when pyarrow/fastparquet is installed, otherwise falls back to
CSV so the pipeline still runs. Parquet is strongly preferred: it stores column
types, so movie_id stays an int32 instead of being re-guessed on every load.
"""


from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

try:
    import pyarrow  # noqa: F401
    PARQUET = True
except ImportError:
    try:
        import fastparquet  # noqa: F401
        PARQUET = True
    except ImportError:
        PARQUET = False

EXT = ".parquet" if PARQUET else ".csv"


def save_table(df: pd.DataFrame, base: Path) -> Path:
    """Save to <base><EXT>. `base` has no extension."""
    base.parent.mkdir(parents=True, exist_ok=True)
    path = base.with_suffix(EXT)
    if PARQUET:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def load_table(base: Path) -> pd.DataFrame:
    path = base.with_suffix(EXT)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run the earlier pipeline steps first."
        )
    return pd.read_parquet(path) if PARQUET else pd.read_csv(path)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def log(msg: str) -> None:
    print(msg, flush=True, file=sys.stdout)


def banner(title: str) -> None:
    log("")
    log("=" * 68)
    log(title)
    log("=" * 68)


def tmdb_cache_path(tmdb_id) -> Path:
    """Where a movie's cached TMDB JSON lives.

    Sharded into subdirectories so no folder holds 27k files. Defined here
    rather than in the fetcher so the Kaggle route can resolve cache paths
    without importing `requests`.
    """
    from . import config as C
    shard = int(tmdb_id) % C.TMDB_SHARDS
    return C.TMDB_CACHE_DIR / f"{shard:02d}" / f"{int(tmdb_id)}.json"
