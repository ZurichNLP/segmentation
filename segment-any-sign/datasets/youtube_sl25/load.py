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
               rebuild_cache: bool = False, dev_filter: bool = True):
    """Yield clip metadata: `id`, `pose_path`, `fps`, `total_frames`, `sentences`.

    Same shape as the DGS loader, so one dataset adapter serves both. `sentences`
    carry empty `glosses`: there is no sign-level supervision here.

    `dev_filter` keeps only the dev videos whose subtitle timings survive
    `DEV_FILTER` — 82 of 164, over 46 of the 56 sign languages. Pass False for
    the raw set, which is what every run before 2026-09-11 validated on.
    """
    index = video_index()
    meta = _pose_meta(index, rebuild=rebuild_cache)
    all_cues = _cues(index, rebuild=rebuild_cache)
    held_out = dev_ids(index, per_language)
    # The dev set is filtered by default and the training set never is: an
    # unreliable label is still worth learning from at this scale, but scoring
    # against one only adds noise to the number we steer by.
    keep = dev_keep(per_language) if (split == "dev" and dev_filter) else None

    for vid, paths in index.items():
        info = meta.get(vid)
        if info is None:                          # unreadable pose
            continue
        if split != "all" and (vid in held_out) != (split == "dev"):
            continue
        if keep is not None and vid not in keep:
            continue
        cues = [{"start_time": a / 1000, "end_time": b / 1000, "glosses": []}
                for a, b in all_cues.get(vid, [])]
        if not cues:
            continue
        yield {"id": vid, "pose_path": paths["pose"], "fps": info["fps"],
               "total_frames": info["total_frames"], "sentences": cues}


# -- dev-set quality: are a video's subtitle timings worth evaluating on? ------
#
# Subtitle timings here are nobody's ground truth. They are written against the
# audio, or the signing, or neither, and nothing in the corpus says which. Three
# cheap measures per video separate the ones worth scoring against, all computed
# from pose motion and the cue timings alone — no model, no manual labels.

QUALITY_CACHE = CACHE_DIR / "youtube_sl25_dev_quality.json"

#: Keep a dev video only if all three hold. `gap > 0` says the timings relate to
#: the signing at all; `|lag| < 2 s` allows the interpreting offset but excludes
#: videos whose best alignment is far away; `coverage < 0.95` excludes videos with
#: too few O frames for a segmentation metric to discriminate. Measured on 164 of
#: the 168 dev videos, this keeps 82 across 46 of the 56 sign languages — half the
#: set. Tightening `|lag|` to 0.5 s would leave 28 videos over 22 languages, which
#: is a noisier dev set than the one it replaced.
DEV_FILTER = {"min_gap": 0.0, "max_lag": 2.0, "max_coverage": 0.95}

#: frames analysed per video. A handful of dev videos run past an hour and the
#: measure is stable long before that.
QUALITY_CAP = 45_000


def _motion(pose) -> "np.ndarray":
    """Per-frame mean keypoint displacement, as z-scores within the video.

    Body and both hands, x and y. Face landmarks move constantly whether or not
    anyone is signing, and POSE_WORLD_LANDMARKS is a different coordinate space
    that would swamp the rest. z-scores because a close-up and a wide shot are
    otherwise incomparable.
    """
    import numpy as np

    offset, keep = 0, []
    for component in pose.header.components:
        if component.name in ("POSE_LANDMARKS", "LEFT_HAND_LANDMARKS",
                              "RIGHT_HAND_LANDMARKS"):
            keep += list(range(offset, offset + len(component.points)))
        offset += len(component.points)
    data = np.ma.filled(pose.body.data, 0.0)[:, 0][:, keep, :2]
    motion = np.linalg.norm(np.diff(data, axis=0), axis=-1).mean(-1)
    if motion.size < 200 or motion.std() == 0:
        return None
    return (motion - motion.mean()) / motion.std()


def dev_quality(per_language: int = DEV_PER_LANGUAGE, rebuild: bool = False) -> dict:
    """Alignment measures for the dev videos. See `measure_quality`."""
    return measure_quality(sorted(dev_ids(video_index(), per_language)),
                           rebuild=rebuild)


def measure_quality(ids, rebuild: bool = False, workers: int = 1) -> dict:
    """Per-video alignment measures, cached across calls. Keys are video ids.

    * `gap` — mean motion inside subtitle spans minus mean motion outside, in
      standard deviations of that video's own motion. Positive means the timings
      land on signing; at or below zero they carry no usable timing signal.
    * `lag` — the shift in seconds that maximises that gap, swept -5 s to +5 s.
      A consistent offset is a *correctable* label, not a bad video, which is why
      the filter tolerates two seconds of it.
    * `coverage` — fraction of frames inside any cue.

    A video that cannot be measured — fewer than four cues, an unreadable pose,
    no motion — is cached as `None` rather than left out, so it is not retried on
    every call, and `dev_keep` drops it.

    The gap is measured unshifted, so a video with a large true offset is
    penalised twice. Scoring it at each video's own best shift would be better and
    would move some videos back into the keep set.
    """
    import json

    import numpy as np
    from pose_format import Pose

    cache = {}
    if QUALITY_CACHE.exists() and not rebuild:
        try:
            cache = json.loads(QUALITY_CACHE.read_text())
        except ValueError:
            cache = {}

    index = video_index()
    meta = _pose_meta(index)
    cues = _cues(index)
    wanted = [v for v in ids if v not in cache]

    def measure(vid):
        info, spans = meta.get(vid), cues.get(vid) or []
        if info is None or len(spans) < 4:
            return vid, None
        fps = info["fps"]
        try:
            with open(index[vid]["pose"], "rb", buffering=4 * 1024 * 1024) as f:
                pose = Pose.read(f, start_frame=0,
                                 end_frame=min(info["total_frames"], QUALITY_CAP))
        except Exception:
            return vid, None
        motion = _motion(pose)
        if motion is None:
            return vid, None
        mask = np.zeros(len(motion), dtype=bool)
        for start, end in spans:
            lo = int(start / 1000 * fps)
            if lo < len(motion):
                mask[max(0, lo):min(len(motion), int(end / 1000 * fps))] = True
        if mask.all() or not mask.any():
            return vid, None

        def score(shift):
            rolled = np.roll(mask, shift)
            return float(motion[rolled].mean() - motion[~rolled].mean())

        step = max(1, int(fps // 5))
        best = max(range(-int(5 * fps), int(5 * fps) + 1, step), key=score)
        return vid, {"gap": score(0), "lag": best / fps,
                     "coverage": float(mask.mean())}

    if wanted:
        from concurrent.futures import ThreadPoolExecutor

        print(f"measuring subtitle alignment for {len(wanted):,} videos "
              f"on {workers} threads...", flush=True)
        # reads dominate and they are on a network share, so threads help well
        # past the core count
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for done, (vid, value) in enumerate(pool.map(measure, wanted), 1):
                cache[vid] = value
                if done % 500 == 0:
                    print(f"  {done:,}/{len(wanted):,}", flush=True)
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    QUALITY_CACHE.write_text(json.dumps(cache))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        QUALITY_CACHE.write_text(json.dumps(cache))
        print(f"cached {len(cache):,} measures to {QUALITY_CACHE}")
    return cache


def passes(quality: dict, rule: dict = None) -> bool:
    """Does one video's measure clear `DEV_FILTER`? An unmeasurable video does
    not: the measure usually fails because there are too few cues or the pose is
    unreadable, neither of which argues for trusting its timings."""
    rule = rule or DEV_FILTER
    return (quality is not None
            and quality["gap"] > rule["min_gap"]
            and abs(quality["lag"]) < rule["max_lag"]
            and quality["coverage"] < rule["max_coverage"])


def dev_keep(per_language: int = DEV_PER_LANGUAGE, rule: dict = None) -> set:
    """Dev ids passing `DEV_FILTER`."""
    return {vid for vid, q in dev_quality(per_language).items()
            if passes(q, rule)}


def keep_ids(ids, rule: dict = None, workers: int = 32) -> set:
    """Which of `ids` pass `DEV_FILTER`, measuring any that are not yet cached.

    Used for the training sampler, where the set is the whole corpus rather than
    the dev split. Measuring 38k videos takes a couple of hours on a many-core
    node and then caches, so this is cheap on every call but the first — but that
    first call happens *inside training startup*, so it warns rather than
    stalling silently.
    """
    # read the cache once, not once per id: it is a 3.7 MB JSON, and calling
    # this inside the loop hung a run for 26 h before anyone noticed
    cached = measure_quality([])
    missing = sum(1 for vid in ids if vid not in cached)
    if missing > 500:
        print(f"  {missing:,} videos have no alignment measure yet; this runs "
              f"once and caches, but it is not quick — see measure_quality")
    quality = measure_quality(sorted(ids), workers=workers)
    return {vid for vid, q in quality.items() if vid in ids and passes(q, rule)}
