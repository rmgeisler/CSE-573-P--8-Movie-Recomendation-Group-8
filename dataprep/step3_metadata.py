"""Step 3 - parse cached TMDB JSON into one flat metadata table, and report coverage.

The coverage number is a result, not a formality. It tells you whether the
knowledge graph will be dense enough to be worth building, and it belongs in
the report. Measure it before you build three weeks of pipeline on top.

Produces:
    processed/metadata         one row per movie_idx
    reports/step3_coverage.json
"""


from __future__ import annotations
import argparse
import json

import pandas as pd

from . import config as C
from .io_utils import (banner, load_table, log, save_table, write_json,
                       tmdb_cache_path as cache_path)



def parse_one(payload: dict) -> dict:
    """Pull the handful of fields the knowledge graph and prompts actually need."""
    credits = payload.get("credits") or {}

    # Director: the crew list holds dozens of roles; we want job == "Director".
    # Co-directed films have several, so keep them all and use the first as primary.
    directors = [
        c.get("name") for c in credits.get("crew", [])
        if c.get("job") == "Director" and c.get("name")
    ]

    # Cast: TMDB's "order" field is billing order, so the first N are the leads.
    cast = sorted(
        [c for c in credits.get("cast", []) if c.get("name")],
        key=lambda c: c.get("order", 10_000),
    )[: C.TOP_CAST_N]

    kw = (payload.get("keywords") or {}).get("keywords", [])

    release = payload.get("release_date") or ""

    return {
        "tmdb_id": payload.get("id"),
        "tmdb_title": payload.get("title"),
        "overview": (payload.get("overview") or "").strip(),
        "release_date": release,
        "tmdb_year": int(release[:4]) if release[:4].isdigit() else None,
        "runtime": payload.get("runtime"),
        "popularity": payload.get("popularity"),
        "vote_average": payload.get("vote_average"),
        "vote_count": payload.get("vote_count"),
        "director": directors[0] if directors else None,
        "directors": directors,
        "cast": [c["name"] for c in cast],
        # Named tmdb_genres, not genres: MovieLens already has a `genres` column
        # and an unqualified merge would silently produce genres_x / genres_y.
        "tmdb_genres": [g["name"] for g in (payload.get("genres") or [])
                        if g.get("name")],
        "keywords": [k["name"] for k in kw if k.get("name")][: C.MAX_KEYWORDS],
    }


def run():
    banner("STEP 3 - PARSE METADATA AND REPORT COVERAGE")

    links = load_table(C.PROCESSED_DIR / "links")
    movies = load_table(C.PROCESSED_DIR / "movies")

    rows, n_no_id, n_no_file, n_bad_json = [], 0, 0, 0

    for rec in links.to_dict("records"):
        tmdb_id = rec.get("tmdb_id")
        if pd.isna(tmdb_id):
            n_no_id += 1
            continue

        p = cache_path(int(tmdb_id))
        if not p.exists():
            n_no_file += 1
            continue

        try:
            payload = json.loads(p.read_text())
        except json.JSONDecodeError:
            n_bad_json += 1
            continue

        parsed = parse_one(payload)
        parsed["movie_idx"] = rec["movie_idx"]
        parsed["movie_id"] = rec["movie_id"]
        parsed["imdb_id"] = rec.get("imdb_id")
        rows.append(parsed)

    meta = pd.DataFrame(rows)
    log(f"  parsed {len(meta):,} of {len(links):,} movies")
    log(f"    no tmdb_id in links : {n_no_id:,}")
    log(f"    not fetched / 404   : {n_no_file:,}")
    log(f"    unparseable json    : {n_bad_json:,}")

    if meta.empty:
        raise SystemExit("No metadata parsed - did step 2 run?")

    # ------------------------------------------------------------ coverage
    total = len(links)
    has_overview = meta["overview"].str.len() > 0
    has_director = meta["director"].notna()
    has_cast3 = meta["cast"].map(len) >= 3
    has_genre = meta["tmdb_genres"].map(len) >= 1
    complete = has_overview & has_director & has_cast3 & has_genre

    coverage = {
        # Recorded so the manifest says which metadata route produced this
        # dataset version - the two sources are not byte-identical.
        "source": "tmdb-api",
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
            "not_fetched_or_404": n_no_file,
            "bad_json": n_bad_json,
        },
    }

    log("")
    log(f"  MATCH RATE: {coverage['match_rate_pct']}%")
    for k, v in coverage["field_coverage_pct"].items():
        log(f"    {k:<20} {v:>6.2f}%")

    if coverage["field_coverage_pct"]["all_four_complete"] < 85:
        log("")
        log("  WARNING: complete-metadata coverage is below 85%.")
        log("  Consider the MovieLens genres + tags.csv fallback for the gap,")
        log("  and report the number in the paper either way.")

    # Movies with no metadata are excluded from retrieval and ranking, so flag
    # them explicitly rather than letting them fail silently downstream.
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
    log(f"  wrote {(C.PROCESSED_DIR / 'metadata')}")
    return coverage


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
