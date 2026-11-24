"""Step 5 - freeze the dataset: hash every file and write a manifest.

This is the step that lets you answer "which data produced this number?" in
week seven. Hash the processed files, record the config that built them, and
commit the manifest (not the data - it is far too big for git).

When preprocessing changes, write a NEW version directory. Never overwrite
v1 in place, or old results become untraceable.

Produces:
    processed/manifest.json
"""


from __future__ import annotations
import argparse
import datetime as dt
import platform
import sys

from . import config as C
from .io_utils import banner, log, sha256, write_json


def run(version: str | None = None):
    banner("STEP 5 - FREEZE AND MANIFEST")

    version = version or f"v1_{dt.date.today().isoformat()}"

    files = sorted(
        p for p in C.PROCESSED_DIR.iterdir()
        if p.is_file() and p.name != "manifest.json"
    )
    if not files:
        raise SystemExit(f"No processed files in {C.PROCESSED_DIR}")

    hashes, sizes = {}, {}
    for p in files:
        hashes[p.name] = sha256(p)
        sizes[p.name] = p.stat().st_size
        log(f"  {p.name:<28} {sizes[p.name]:>12,} B  {hashes[p.name][:16]}...")

    # Fold in whatever the earlier steps reported, so one file answers everything.
    reports = {}
    for name in ("step1_stats", "step3_coverage", "step4_split"):
        rp = C.REPORT_DIR / f"{name}.json"
        if rp.exists():
            import json
            reports[name] = json.loads(rp.read_text())

    manifest = {
        "version": version,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "MovieLens 20M + TMDB API",
        "config": C.as_dict(),
        "files": {n: {"sha256": hashes[n], "bytes": sizes[n]} for n in hashes},
        "reports": reports,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }

    path = write_json(manifest, C.PROCESSED_DIR / "manifest.json")
    log("")
    log(f"  version: {version}")
    log(f"  wrote {path}")
    log("  commit this manifest to git; leave the data files out.")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=None,
                    help="version tag, e.g. v2_2026-10-20")
    args = ap.parse_args()
    run(version=args.version)
