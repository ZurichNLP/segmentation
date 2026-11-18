"""Download the NCSLGR Corpus annotation archives.

ASLLRP is the *project*; it distributes two distinct corpora, and this directory
covers only the first:

  * **NCSLGR Corpus** (SignStream 2, released 2007) — annotation archives are
    served directly from bu.edu, **no account required**. That is what this
    script fetches, and what `explore.py` here analyses.
  * **ASLLRP SignStream 3 Corpus** — a separate corpus with richer annotations
    (both hands, sign type, handshapes). Downloads go through the DAI 2
    **Download Cart** and need a free account. Not handled here; it should get
    its own `datasets/asllrp_signstream3/` when we take it on.

Videos for either corpus are *not* covered by this script — NCSLGR video comes
through the original DAI, SignStream 3 video through the cart. We will need
video eventually; this phase is annotations only, so we can confirm the timing
semantics and gloss statistics for ~2 MB before committing to anything larger.

Licence (https://www.bu.edu/asllrp/dai-terms.html): research and education use
only; **no redistribution without permission**; no commercial use; citation
required. The download directory is group-only for that reason. It is the key
`data.ncslgr.root` of the paths file (`../../paths.py`), unless `--dest` names
another; it is created when missing.

Usage:
    python download_ncslgr.py              # download + extract
    python download_ncslgr.py --force      # re-download
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import paths

BASE_URL = "https://www.bu.edu/asllrp/ncslgr-for-download"

# name -> (url, what it is)
ARCHIVES = {
    "Archive-ssdb-2-2-12.zip": (
        f"{BASE_URL}/Archive-ssdb-2-2-12.zip",
        "SignStream 2 annotation database (the corpus annotations)",
    ),
    "video_index-20120129.zip": (
        f"{BASE_URL}/video_index-20120129.zip",
        "video index charts mapping utterances to media",
    ),
    "signstream-xmlparser.zip": (
        f"{BASE_URL}/signstream-xmlparser.zip",
        "reference Python parser for the SignStream XML export",
    ),
}

# served from the same page, documents the XML schema
EXTRA_FILES = {"old-dtd.xml": f"{BASE_URL}/old-dtd.xml"}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path, force: bool) -> bool:
    """Download url to destination. Returns True if it downloaded, False if skipped."""
    if destination.exists() and not force:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "segment-any-sign/research"})
    with urllib.request.urlopen(request, timeout=120) as response, open(destination, "wb") as out:
        shutil.copyfileobj(response, out)
    return True


def video_urls(dest: Path) -> list[tuple[str, str, str]]:
    """Read the extracted video index -> [(filename, camera_id, url), ...].

    The index lists a compressed .mov per (video sequence, camera perspective),
    plus uncompressed AVI mirrors we ignore. The CSV uses old Mac CR line
    endings, hence the newline normalisation.
    """
    index = (dest / "video_index-20120129" / "video_index-20120129"
             / "files_by_video_name.csv")
    if not index.exists():
        raise SystemExit(f"video index not found at {index} — run without --videos first")

    raw = index.read_bytes().decode("utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    import csv
    import io

    entries = []
    for row in csv.DictReader(io.StringIO(raw)):
        cell = (row.get("Compressed MOV file") or "").strip()
        if not cell:
            continue
        name = row["Video file name in XML file"].strip()
        camera = (row.get("Perspective/Camera id") or "?").strip()

        # A few long narratives are split across parts, given as several
        # semicolon-separated URLs in one cell. Keep each part under its own
        # upstream filename so they stay distinguishable.
        urls = [u.strip() for u in cell.split(";") if u.strip()]
        if len(urls) == 1:
            entries.append((name, camera, urls[0]))
        else:
            for url in urls:
                entries.append((url.rsplit("/", 1)[-1], camera, url))
    return entries


def download_videos(dest: Path, cameras: set[str] | None, force: bool, delay: float,
                    limit: int | None) -> None:
    import time

    videos_dir = dest / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    entries = video_urls(dest)
    if cameras:
        entries = [e for e in entries if e[1] in cameras]
    if limit:
        entries = entries[:limit]

    print(f"\nvideos -> {videos_dir}")
    print(f"  {len(entries)} files to consider"
          f"{' (cameras ' + ','.join(sorted(cameras)) + ')' if cameras else ''}")

    got = skipped = failed = 0
    total_bytes = 0
    started = time.time()

    for index, (name, camera, url) in enumerate(entries, start=1):
        target = videos_dir / name
        if target.exists() and not force:
            skipped += 1
        else:
            try:
                fetch(url, target, force=True)
                got += 1
                total_bytes += target.stat().st_size
                time.sleep(delay)  # be polite to csr.bu.edu
            except Exception as error:
                failed += 1
                if failed <= 5:
                    print(f"    failed {name}: {type(error).__name__}")

        if index % 200 == 0 or index == len(entries):
            elapsed = time.time() - started
            print(f"  ...{index}/{len(entries)}  got {got}, skipped {skipped}, "
                  f"failed {failed}  ({total_bytes / 1e6:.0f} MB, {elapsed:.0f}s)", flush=True)

    print(f"\n  downloaded {got}, skipped {skipped}, failed {failed}"
          f"  — {total_bytes / 1e9:.2f} GB in {time.time() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=None,
                        help="download directory (default: data.ncslgr.root of the paths file)")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--no-extract", action="store_true", help="skip unzipping")
    parser.add_argument("--videos", action="store_true",
                        help="also download the compressed .mov videos (~1.4 GB, 2612 files)")
    parser.add_argument("--cameras", default=None,
                        help="comma-separated camera ids to keep, e.g. 0 for the front view "
                             "(default: all of 0,1,2,3)")
    parser.add_argument("--delay", type=float, default=0.15,
                        help="seconds between video requests (politeness)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N videos")
    args = parser.parse_args()

    if args.dest is None:
        # the script makes the folder itself, so it need not exist yet
        args.dest = paths.data("ncslgr.root", must_exist=False)
    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"destination {args.dest}\n")

    for name, (url, description) in {**ARCHIVES,
                                     **{k: (v, "XML DTD") for k, v in EXTRA_FILES.items()}}.items():
        path = args.dest / name
        downloaded = fetch(url, path, args.force)
        size_mb = path.stat().st_size / 1e6
        print(f"  {'downloaded' if downloaded else 'present   '} {name:<28} "
              f"{size_mb:>6.2f} MB  md5 {md5(path)[:12]}")
        print(f"               {description}")

    if args.no_extract:
        return

    print("\nextracting")
    for name in ARCHIVES:
        archive = args.dest / name
        target = args.dest / archive.stem
        if target.exists() and not args.force:
            print(f"  present    {target.name}/")
            continue
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        members = sum(1 for _ in target.rglob("*") if _.is_file())
        print(f"  extracted  {target.name}/  ({members} files)")

    if args.videos:
        cameras = {c.strip() for c in args.cameras.split(",")} if args.cameras else None
        download_videos(args.dest, cameras, args.force, args.delay, args.limit)

    print(f"\ndone — {args.dest}")


if __name__ == "__main__":
    main()
