"""Locate input CSVs wherever they happen to live under data/.

Hardcoding `data/raw/ml-20m/ratings.csv` assumed everyone unzips the same way.
They do not. This searches recursively instead, prints what it found, and
refuses to guess when a filename is genuinely ambiguous.

The ambiguity matters more than it sounds. The Kaggle "The Movies Dataset"
archive ships its OWN ratings.csv and links.csv, which are not the MovieLens
20M versions. Unzipped into one flat folder, one silently overwrites the other
and the pipeline quietly runs on the wrong data.

Run `python -m prep.discover` to see what it resolves before running anything.
"""

from __future__ import annotations

from pathlib import Path

from . import config as C
from .io_utils import banner, log

# Directories never worth searching.
SKIP = {".venv", "__pycache__", ".git", "processed", "node_modules", ".idea"}

# Filenames that appear in BOTH archives. For these, which copy we take
# actually changes the results, so we disambiguate rather than guess.
CONTESTED = {"ratings.csv", "links.csv", "movies.csv"}

# A file that identifies each archive, used to tell the two apart.
ML_MARKER = "genome-scores.csv"          # ml-20m only
KAGGLE_MARKER = "movies_metadata.csv"    # The Movies Dataset only


def _candidates(filename: str, base: Path) -> list[Path]:
    if not base.exists():
        return []
    out = []
    for p in base.rglob(filename):
        if any(part in SKIP for part in p.parts):
            continue
        out.append(p)
    return sorted(out, key=lambda p: (len(p.parts), str(p)))


def _classify(d: Path) -> str:
    """Guess which archive a directory came from, by looking for marker files."""
    if (d / KAGGLE_MARKER).exists():
        return "kaggle"
    if (d / ML_MARKER).exists():
        return "movielens"
    # tags.csv exists in both; genome files are the reliable MovieLens tell.
    if any(d.glob("genome-*.csv")):
        return "movielens"
    if (d / "credits.csv").exists() or (d / "keywords.csv").exists():
        return "kaggle"
    return "unknown"


def locate(filename: str, prefer: str | None = None,
           base: Path | None = None, required: bool = True):
    """Find `filename` under the data directory.

    prefer: "movielens" or "kaggle" - which archive's copy to take when the
    same filename exists in both.
    """
    base = base or C.DATA_DIR
    found = _candidates(filename, base)

    if not found:
        if not required:
            return None
        raise SystemExit(
            f"\nCould not find '{filename}' anywhere under {base}\n"
            f"  Put your CSVs somewhere under that folder (any layout), or\n"
            f"  point elsewhere with --data-dir /path/to/your/data\n"
        )

    if len(found) == 1:
        return found[0]

    # Several copies. If we know which archive we want, use the markers.
    if prefer:
        matching = [p for p in found if _classify(p.parent) == prefer]
        if len(matching) == 1:
            return matching[0]
        if len(matching) > 1:
            return matching[0]

    if filename in CONTESTED:
        listing = "\n".join(
            f"    {p}   [looks like: {_classify(p.parent)}]" for p in found
        )
        raise SystemExit(
            f"\nFound {len(found)} copies of '{filename}' and cannot tell which to use:\n"
            f"{listing}\n\n"
            f"  This file ships in BOTH MovieLens 20M and the Kaggle dump, and the\n"
            f"  two are not interchangeable. Put each archive in its own subfolder,\n"
            f"  e.g. data/ml-20m/ and data/the-movies-dataset/, then re-run.\n"
        )

    return found[0]


def resolve_all(base: Path | None = None, source: str = "kaggle") -> dict:
    """Resolve every input the pipeline needs. Returns {logical name: Path}."""
    base = base or C.DATA_DIR

    paths = {
        "ratings": locate("ratings.csv", prefer="movielens", base=base),
        "movies": locate("movies.csv", prefer="movielens", base=base),
        "links": locate("links.csv", prefer="movielens", base=base),
    }

    if source == "kaggle":
        paths["movies_metadata"] = locate("movies_metadata.csv", base=base)
        paths["credits"] = locate("credits.csv", base=base)
        paths["keywords"] = locate("keywords.csv", base=base)

    return paths


def report(base: Path | None = None) -> None:
    """Print what is discoverable, so layout problems surface before a long run."""
    base = base or C.DATA_DIR
    banner("DATA DISCOVERY")
    log(f"  searching under: {base}")
    log("")

    if not base.exists():
        log(f"  That directory does not exist.")
        log(f"  Create it and put your CSVs inside, or pass --data-dir.")
        return

    wanted = ["ratings.csv", "movies.csv", "links.csv",
              "movies_metadata.csv", "credits.csv", "keywords.csv"]
    for name in wanted:
        hits = _candidates(name, base)
        if not hits:
            log(f"  {name:<22} NOT FOUND")
        elif len(hits) == 1:
            log(f"  {name:<22} {hits[0].relative_to(base)}")
        else:
            log(f"  {name:<22} {len(hits)} copies:")
            for h in hits:
                log(f"  {'':<22}   {h.relative_to(base)}  [{_classify(h.parent)}]")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    report(Path(a.data_dir) if a.data_dir else None)
