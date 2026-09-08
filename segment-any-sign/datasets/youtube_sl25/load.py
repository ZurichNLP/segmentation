"""Shared YouTube-SL-25 loading — subtitle timings as weak phrase boundaries.

Stage one of the staged design (see `../../experiments/README.md`). A subtitle
cue is a translation unit, so this corpus supervises the **phrase level only**;
the sign head gets no signal and must be masked, not given empty labels.

Deliberately unfiltered. The raw distribution is the point: 56% of cue
transitions are back-to-back and a third of videos are >95% covered, so O frames
are scarce — the mid- and post-training stages on DGS are what shift the domain.

Pose headers are cached under `/scratch`, since scanning 38,850 files costs about
an hour and a run cannot pay that each time.

    from datasets.youtube_sl25 import load as yt
    for spec in yt.clip_specs("train"):
        ...
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = "/shares/iict-sp2.ebling.cl.uzh/common/YouTube-SL-25_VGG"
POSES = f"{ROOT}/mediapipe"
SUBTITLES = f"{ROOT}/subtitles"

# headers are read once and cached here; /scratch, never /home
CACHE_DIR = Path("/scratch/zifjia/segment-any-sign/cache")
META_CACHE = CACHE_DIR / "youtube_sl25_pose_meta.json"
CUE_CACHE = CACHE_DIR / "youtube_sl25_cues.json"

#: videos held out for in-domain validation, per sign language. All 56 languages
#: in the corpus have at least 12 videos, so 3 each gives a balanced 168-video
#: dev set — enough to cover every language, small enough that validating on
#: whole videos (median 3.8 min) stays affordable.
DEV_PER_LANGUAGE = 3
METADATA = f"{ROOT}/youtube-sl-25_youtube-sl-25-metadata.csv"

_TIMESTAMP = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")


def _seconds(hours: str, minutes: str, secs: str, millis: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(millis) / 1000


def parse_vtt(path: str) -> list[dict]:
    """Cue spans in seconds, sorted. Text is ignored — only timings are labels.

    Empty cues and pure sound markers (`[Music]`, `♪ ... ♪`) are dropped: they
    mark audio events, not signing, and are about 1.3% of cues. Nothing else is
    filtered.
    """
    try:
        text = Path(path).read_text(errors="ignore")
    except OSError:
        return []

    cues = []
    for block in text.split("\n\n"):
        match = _TIMESTAMP.search(block)
        if not match:
            continue
        body = block[match.end():].strip()
        if not body or re.match(r"^\s*[\[(♪]", body):
            continue
        start = _seconds(*match.groups()[:4])
        end = _seconds(*match.groups()[4:])
        if end > start:
            cues.append({"start_time": start, "end_time": end, "glosses": []})
    cues.sort(key=lambda c: c["start_time"])
    return cues


def video_index() -> dict[str, dict]:
    """Map video id -> {pose, subtitle}, keeping only ids that have both."""
    poses = {p[:-len(".pose")]: os.path.join(POSES, p)
             for p in os.listdir(POSES) if p.endswith(".pose")}
    subtitles = {}
    for name in os.listdir(SUBTITLES):
        if name.endswith(".vtt"):
            subtitles.setdefault(name.split(".")[0], os.path.join(SUBTITLES, name))
    return {vid: {"pose": poses[vid], "subtitle": subtitles[vid]}
            for vid in sorted(poses.keys() & subtitles.keys())}


def _pose_meta(index: dict, rebuild: bool = False) -> dict:
    """fps and frame count per video, cached. Unreadable files are recorded as
    `null` so a corrupt pose is skipped without being re-opened every run."""
    cache = {}
    if META_CACHE.exists() and not rebuild:
        try:
            cache = json.loads(META_CACHE.read_text())
        except ValueError:
            cache = {}

    missing = [vid for vid in index if vid not in cache]
    if missing:
        from pose_format import Pose
        from pose_format.pose_body import EmptyPoseBody

        from concurrent.futures import ThreadPoolExecutor

        def read_one(vid: str):
            # header only: a few hundred bytes off a file that averages 72 MB
            try:
                with open(index[vid]["pose"], "rb") as handle:
                    pose = Pose.read(handle, pose_body=EmptyPoseBody)
                return vid, {"fps": float(pose.body.fps),
                             "total_frames": int(len(pose.body.data))}
            except Exception as error:            # truncated or unreadable
                return vid, None, type(error).__name__

        print(f"reading {len(missing):,} pose headers (cached afterwards)...")
        # header reads are IO-bound on a network share, so threads help a lot
        with ThreadPoolExecutor(max_workers=32) as pool:
            for done, result in enumerate(pool.map(read_one, missing), 1):
                cache[result[0]] = result[1]
                if result[1] is None:
                    print(f"  unreadable {result[0]}: {result[2]}")
                if done % 5000 == 0:
                    print(f"  {done:,}/{len(missing):,}", flush=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        META_CACHE.write_text(json.dumps(cache))
        broken = sum(1 for v in cache.values() if v is None)
        print(f"cached {len(cache):,} headers to {META_CACHE}"
              f"{f', {broken} unreadable' if broken else ''}")
    return cache


def _cues(index: dict, rebuild: bool = False) -> dict:
    """Cue spans per video, cached as integer milliseconds.

    Parsing 38,850 subtitle files off the share takes minutes; a run cannot pay
    that at every startup. Stored as ms ints rather than float seconds — a third
    the size, and exact.
    """
    cache = {}
    if CUE_CACHE.exists() and not rebuild:
        try:
            cache = json.loads(CUE_CACHE.read_text())
        except ValueError:
            cache = {}

    missing = [vid for vid in index if vid not in cache]
    if missing:
        from concurrent.futures import ThreadPoolExecutor

        def read_one(vid: str):
            return vid, [[int(c["start_time"] * 1000), int(c["end_time"] * 1000)]
                         for c in parse_vtt(index[vid]["subtitle"])]

        print(f"parsing {len(missing):,} subtitle files (cached afterwards)...")
        with ThreadPoolExecutor(max_workers=32) as pool:
            for done, (vid, cues) in enumerate(pool.map(read_one, missing), 1):
                cache[vid] = cues
                if done % 10000 == 0:
                    print(f"  {done:,}/{len(missing):,}", flush=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CUE_CACHE.write_text(json.dumps(cache, separators=(",", ":")))
        print(f"cached {sum(len(v) for v in cache.values()):,} cues to {CUE_CACHE}")
    return cache


def languages() -> dict[str, str]:
    """Video id -> sign-language code, from the corpus metadata.

    Note the file has CRLF line endings; values must be stripped or every
    comparison silently fails. 197 videos are labelled `???` and are kept — an
    unknown language is not a reason to drop otherwise good data.
    """
    import csv

    mapping = {}
    with open(METADATA, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                mapping[row[0].strip()] = row[1].strip()
    return mapping


def dev_ids(index: dict, per_language: int = DEV_PER_LANGUAGE) -> set[str]:
    """A language-balanced dev set: the same number of videos from each language.

    A hash split would follow the corpus distribution, which is 42% ASL — so
    in-domain validation would mostly measure ASL. Taking a fixed number per
    language instead means the selection metric reflects all 56 languages.
    Selection within a language is by hash, so membership never depends on file
    order, and `???` is treated as its own group rather than discarded.
    """
    import hashlib
    from collections import defaultdict

    by_language: dict[str, list[str]] = defaultdict(list)
    lang = languages()
    for vid in index:
        by_language[lang.get(vid, "???")].append(vid)

    chosen = set()
    for code, vids in sorted(by_language.items()):
        ordered = sorted(vids, key=lambda v: hashlib.sha256(
            f"ytsl25_{v}".encode()).hexdigest())
        chosen.update(ordered[:per_language])
    return chosen


def clip_specs(split: str = "train", per_language: int = DEV_PER_LANGUAGE,
               rebuild_cache: bool = False):
    """Yield clip metadata: `id`, `pose_path`, `fps`, `total_frames`, `sentences`.

    Same shape as the DGS loader, so one dataset adapter serves both. `sentences`
    carry empty `glosses`: there is no sign-level supervision here.
    """
    index = video_index()
    meta = _pose_meta(index, rebuild=rebuild_cache)
    all_cues = _cues(index, rebuild=rebuild_cache)
    held_out = dev_ids(index, per_language)

    for vid, paths in index.items():
        info = meta.get(vid)
        if info is None:                          # unreadable pose
            continue
        if split != "all" and (vid in held_out) != (split == "dev"):
            continue
        cues = [{"start_time": a / 1000, "end_time": b / 1000, "glosses": []}
                for a, b in all_cues.get(vid, [])]
        if not cues:
            continue
        yield {"id": vid, "pose_path": paths["pose"], "fps": info["fps"],
               "total_frames": info["total_frames"], "sentences": cues}
