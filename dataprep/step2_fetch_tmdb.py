"""Step 2 - fetch TMDB metadata for every movie, once, resumably.

This is the long one. Roughly 27,000 requests, several hours wall-clock. Start
it on day one and let it run while you build the rest of the pipeline.

Resumable by design: every movie is written to its own cache file, and a movie
whose file already exists is skipped. If the process dies at hour two, re-run
the same command and it picks up where it stopped.

Usage:
    export TMDB_API_KEY=...
    python -m prep.step2_fetch_tmdb               # fetch everything missing
    python -m prep.step2_fetch_tmdb --limit 100   # smoke test first
"""


from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

from . import config as C
from .io_utils import banner, load_table, log, write_json, tmdb_cache_path


# Re-exported for callers that still import it from here.
cache_path = tmdb_cache_path


def fetch_one(session, tmdb_id: int, api_key: str):
    """Return (status, payload). status is 'ok' | 'not_found' | 'error'.

    One request pulls details, credits and keywords together via
    append_to_response, which is three times fewer calls than fetching them
    separately and counts as one request against the rate limit.
    """
    url = C.TMDB_BASE.format(tmdb_id=int(tmdb_id))
    params = {"api_key": api_key, "append_to_response": C.TMDB_APPEND}

    import requests  # imported here so the Kaggle route never needs it

    for attempt in range(C.TMDB_MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=20)
        except requests.RequestException as e:
            wait = 2 ** attempt
            log(f"    network error on {tmdb_id} ({e.__class__.__name__}), "
                f"retry in {wait}s")
            time.sleep(wait)
            continue

        if r.status_code == 200:
            return "ok", r.json()
        if r.status_code == 404:
            return "not_found", {"status_code": 404}
        if r.status_code == 429:
            # TMDB tells us exactly how long to wait; obey it rather than guessing.
            wait = int(r.headers.get("Retry-After", 2)) + 1
            log(f"    rate limited, sleeping {wait}s")
            time.sleep(wait)
            continue
        if 500 <= r.status_code < 600:
            wait = 2 ** attempt
            log(f"    server error {r.status_code} on {tmdb_id}, retry in {wait}s")
            time.sleep(wait)
            continue
        return "error", {"status_code": r.status_code, "body": r.text[:200]}

    return "error", {"status_code": "max_retries_exceeded"}


def run(limit: int | None = None, api_key: str | None = None):
    banner("STEP 2 - FETCH TMDB METADATA")

    api_key = api_key or os.environ.get("TMDB_API_KEY")
    if not api_key:
        raise SystemExit(
            "TMDB_API_KEY is not set.\n"
            "  Register a free key at themoviedb.org/settings/api, then:\n"
            "    export TMDB_API_KEY=your_key_here"
        )

    links = load_table(C.PROCESSED_DIR / "links")
    targets = links.dropna(subset=["tmdb_id"]).copy()
    targets["tmdb_id"] = targets["tmdb_id"].astype(int)
    ids = targets["tmdb_id"].drop_duplicates().tolist()

    if limit:
        ids = ids[:limit]
        log(f"  LIMIT set: fetching only the first {limit} ids")

    todo = [i for i in ids if not cache_path(i).exists()]
    log(f"  {len(ids):,} ids total, {len(ids) - len(todo):,} already cached, "
        f"{len(todo):,} to fetch")

    if not todo:
        log("  nothing to do")
        return {"fetched": 0, "not_found": 0, "errors": 0}

    import requests
    session = requests.Session()
    counts = {"ok": 0, "not_found": 0, "error": 0}
    failures = []
    started = time.time()

    for n, tmdb_id in enumerate(todo, 1):
        status, payload = fetch_one(session, tmdb_id, api_key)
        counts[status] += 1

        if status == "ok":
            p = cache_path(tmdb_id)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload))
        else:
            # Record it, but do not write a cache file - we want a re-run to retry.
            failures.append({"tmdb_id": int(tmdb_id), "status": status, **payload})

        if n % 250 == 0 or n == len(todo):
            rate = n / max(time.time() - started, 1e-9)
            eta_min = (len(todo) - n) / max(rate, 1e-9) / 60
            log(f"  {n:,}/{len(todo):,}  ok={counts['ok']:,} "
                f"missing={counts['not_found']:,} err={counts['error']:,} "
                f"({rate:.1f}/s, ~{eta_min:.0f} min left)")

        time.sleep(C.TMDB_SLEEP)

    if failures:
        write_json(failures, C.REPORT_DIR / "step2_failures.json")
        log(f"  {len(failures):,} failures written to reports/step2_failures.json")

    log(f"  done in {(time.time() - started) / 60:.1f} min")
    return {"fetched": counts["ok"], "not_found": counts["not_found"],
            "errors": counts["error"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="fetch only the first N ids (smoke test)")
    args = ap.parse_args()
    run(limit=args.limit)
