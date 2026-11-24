"""Step 2b - build metadata from the Kaggle dumps instead of the TMDB API.

Replaces steps 2 and 3 in one pass. Output schema is identical, so steps 4 and
5 run unchanged.

Expects in data/raw/the-movies-dataset/:
    movies_metadata.csv    45k films, pre-July-2017
    credits.csv            cast + crew, as stringified JSON in a CSV cell
    keywords.csv           plot keywords, likewise

Three things about these files will break a naive read, and all three are
handled below:

  1. The list columns are Python-literal strings, not JSON. "[{'id': 28,
     'name': 'Action'}]" uses single quotes, so json.loads fails. We use
     ast.literal_eval, which parses Python literals WITHOUT executing code -
     this matters because the file is third-party data.

  2. A handful of rows in movies_metadata.csv have shifted columns, leaving a
     date string where the numeric id should be. int() raises on those. We
     filter to rows whose id is all digits and report how many were dropped.

  3. The id column repeats. Duplicate films must be collapsed or the join
     multiplies rows.

Produces:
    processed/metadata
    reports/step3_coverage.json   (same path as the API route, so downstream
                                   reporting does not care which source ran)
"""


from __future__ import annotations
import argparse
import ast

import pandas as pd

from . import config as C
from .discover import locate
from .io_utils import banner, load_table, log, save_table, write_json


def parse_literal_list(cell) -> list:
    """Parse a stringified Python list of dicts; return [] on anything odd.

    ast.literal_eval only evaluates literals (lists, dicts, strings, numbers),
    so a malicious string cannot execute code the way eval() would.
    """
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        val = ast.literal_eval(cell)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return []
    return val if isinstance(val, list) else []


def names(items, key="name", limit=None) -> list:
    out = [i.get(key) for i in items
           if isinstance(i, dict) and i.get(key)]
    return out[:limit] if limit else out


def load_kaggle():
    md_path = locate("movies_metadata.csv")
    cr_path = locate("credits.csv")
    kw_path = locate("keywords.csv")
    for label, pth in (("metadata", md_path), ("credits", cr_path),
                       ("keywords", kw_path)):
        log(f"  {label:<9} <- {pth}")

    md = pd.read_csv(md_path, low_memory=False)  # mixed dtypes expected here
    credits = pd.read_csv(cr_path)
    keywords = pd.read_csv(kw_path)

    log(f"    movies_metadata : {len(md):>8,} rows")
    log(f"    credits         : {len(credits):>8,} rows")
    log(f"    keywords        : {len(keywords):>8,} rows")

    # --- problem 2: rows whose id is not a number (shifted columns) ---------
    before = len(md)
    md["id"] = md["id"].astype(str).str.strip()
    md = md[md["id"].str.fullmatch(r"\d+")]
    malformed = before - len(md)
    md["id"] = md["id"].astype("int64")
    log(f"    dropped {malformed} malformed rows (non-numeric id)")

    # --- problem 3: duplicate ids ------------------------------------------
    dupes = int(md["id"].duplicated().sum())
    md = md.drop_duplicates(subset="id", keep="first")
    credits = credits.drop_duplicates(subset="id", keep="first")
    keywords = keywords.drop_duplicates(subset="id", keep="first")
    log(f"    dropped {dupes} duplicate ids")

    return md, credits, keywords, malformed, dupes


def build(md, credits, keywords) -> pd.DataFrame:
    """Flatten the three Kaggle tables into one row per TMDB id."""
    md = md.merge(credits, on="id", how="left").merge(keywords, on="id", how="left")

    # --- problem 1: stringified Python literals ----------------------------
    log("  parsing stringified JSON columns (this takes a minute)")
    crew = md["crew"].map(parse_literal_list)
    cast = md["cast"].map(parse_literal_list)
    genres = md["genres"].map(parse_literal_list)
    kws = md["keywords"].map(parse_literal_list)

    directors = crew.map(
        lambda items: [c.get("name") for c in items
                       if isinstance(c, dict)
                       and c.get("job") == "Director" and c.get("name")]
    )
    # TMDB's "order" is billing order, so sorting by it gives the leads first.
    top_cast = cast.map(
        lambda items: names(
            sorted([c for c in items if isinstance(c, dict)],
                   key=lambda c: c.get("order", 10_000)),
            limit=C.TOP_CAST_N,
        )
    )

    release = md["release_date"].fillna("").astype(str)

    out = pd.DataFrame({
        "tmdb_id": md["id"].astype("int64"),
        "tmdb_title": md["title"],
        "overview": md["overview"].fillna("").astype(str).str.strip(),
        "release_date": release,
        "tmdb_year": pd.to_numeric(release.str[:4], errors="coerce"),
        "runtime": pd.to_numeric(md["runtime"], errors="coerce"),
        "popularity": pd.to_numeric(md["popularity"], errors="coerce"),
        "vote_average": pd.to_numeric(md["vote_average"], errors="coerce"),
        "vote_count": pd.to_numeric(md["vote_count"], errors="coerce"),
        "director": directors.map(lambda d: d[0] if d else None),
        "directors": directors,
        "cast": top_cast,
        "tmdb_genres": genres.map(lambda g: names(g)),
        "keywords": kws.map(lambda k: names(k, limit=C.MAX_KEYWORDS)),
    })
    return out


def run():
    banner("STEP 2b - BUILD METADATA FROM KAGGLE DUMPS")

    links = load_table(C.PROCESSED_DIR / "links")
    movies = load_table(C.PROCESSED_DIR / "movies")

    md, credits, keywords, malformed, dupes = load_kaggle()
    flat = build(md, credits, keywords)

    # Join to our catalog on tmdb_id. Films in the dump that MovieLens does not
    # carry are simply dropped by the inner-style merge below.
    targets = links.dropna(subset=["tmdb_id"]).copy()
    targets["tmdb_id"] = targets["tmdb_id"].astype("int64")
    n_no_id = int(links["tmdb_id"].isna().sum())

    meta = targets.merge(flat, on="tmdb_id", how="inner")
    n_unmatched = len(targets) - len(meta)

    log("")
    log(f"  catalog movies            : {len(links):,}")
    log(f"    no tmdb_id in links.csv : {n_no_id:,}")
    log(f"    tmdb_id not in the dump : {n_unmatched:,}")
    log(f"    matched                 : {len(meta):,}")

    if meta.empty:
        raise SystemExit("Nothing matched - check that KAGGLE_DIR points at the dump.")

    # ------------------------------------------------------------ coverage
    total = len(links)
    has_overview = meta["overview"].str.len() > 0
    has_director = meta["director"].notna()
    has_cast3 = meta["cast"].map(len) >= 3
    has_genre = meta["tmdb_genres"].map(len) >= 1
    complete = has_overview & has_director & has_cast3 & has_genre

    coverage = {
        "source": "kaggle:the-movies-dataset",
        "movies_in_catalog": int(total),
        "movies_with_metadata": int(len(meta)),
        "match_rate_pct": round(100 * len(meta) / total, 2),
        "field_coverage_pct": {
            "overview": round(100 * has_overview.sum() / total, 2),
            "director": round(100 * has_director.sum() / total, 2),
            "cast_3_plus": round(100 * has_cast3.sum() / total, 2),
            "genre": round(100 * has_genre.sum() / total, 2),
            "keywords": round(100 * (meta["keywords"].map(len) >= 1).sum() / total, 2),
            "all_four_complete": round(100 * complete.sum() / total, 2),
        },
        "excluded": {
            "no_tmdb_id": n_no_id,
            "tmdb_id_not_in_dump": int(n_unmatched),
            "malformed_rows_dropped": int(malformed),
            "duplicate_ids_dropped": int(dupes),
        },
    }

    log("")
    log(f"  MATCH RATE: {coverage['match_rate_pct']}%")
    for k, v in coverage["field_coverage_pct"].items():
        log(f"    {k:<20} {v:>6.2f}%")

    if coverage["field_coverage_pct"]["all_four_complete"] < 85:
        log("")
        log("  WARNING: complete-metadata coverage is below 85%.")
        log("  Either fall back to MovieLens genres + tags.csv for the gap, or")
        log("  fetch only the unmatched ids from the TMDB API. Report it either way.")

    meta["metadata_complete"] = complete
    merged = movies.merge(
        meta.drop(columns=["movie_id"]), on="movie_idx", how="left"
    )
    merged["has_metadata"] = merged["tmdb_id"].notna()
    # Movies with no metadata merge in as NaN; make the flag a real bool
    # so downstream code can filter on it without tripping over NaN.
    merged["metadata_complete"] = (
        merged["metadata_complete"].fillna(False).astype(bool)
    )

    # Guarantee both metadata routes emit the identical schema.
    missing = [c for c in C.METADATA_COLUMNS if c not in merged.columns]
    if missing:
        raise SystemExit(f"metadata is missing expected columns: {missing}")
    merged = merged[C.METADATA_COLUMNS]

    save_table(merged, C.PROCESSED_DIR / "metadata")
    write_json(coverage, C.REPORT_DIR / "step3_coverage.json")
    log(f"  wrote {C.PROCESSED_DIR / 'metadata'}")
    return coverage


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
