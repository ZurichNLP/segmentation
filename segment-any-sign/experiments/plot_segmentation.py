"""Draw predicted segments against gold, for a handful of validation clips.

A number tells you a model segments badly; a strip tells you *how*. Each clip
becomes two ribbons over the same time axis — gold on top, prediction below —
so merging, splitting and offset are visible at a glance.

Written so the same function serves two callers: this CLI, and a W&B callback
that logs the panel once for the best checkpoint (never per validation, or the
run drowns in figures).

    python experiments/plot_segmentation.py --run 02_pretrain_youtube-2026.09.08 \
        --dataset youtube_25 --clips 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DIST = Path("/scratch/zifjia/segment-any-sign/dist")
LEVELS = {"sign": "sign", "sentence": "phrase"}

#: (record key, row label, colour). A row is drawn only if some record has it,
#: so the same function serves a plain gold-vs-ours panel and a comparison.
ROWS = (
    ("gold", "gold", "#d69e2e"),
    ("pred", "ours", "#2b6cb0"),
    ("ref_2023", "2023", "#38a169"),
    ("ref_2026", "2026", "#805ad5"),
)

CAPTION = {
    "gold": "gold — the reference annotation, split at every B.",
    "pred": "ours — our model's argmax, decoded by Moryossef & Jiang (2023) "
            "Algorithm 1, where a B closes the open segment and opens the next.",
    "ref_2023": "2023 — the published Moryossef & Jiang (2023) E4s model at its "
                "tuned decoding thresholds, exactly as benchmarked.",
    "ref_2026": "2026 — the shipped model at its own defaults and its own "
                "decoding, which ignores B, exactly as benchmarked.",
}

FOOTER = ("White hairlines separate touching segments. One panel per clip, "
          "first two minutes only.")

DETAIL_CAPTION = (
    "Keyframes are sampled uniformly, at least one per second, from the video "
    "where one exists and from a pose render otherwise, and run edge to edge in "
    "time order. Text above a span is the subtitle line overlapping it, thinned "
    "so a dense window stays readable. The window is the one holding the most "
    "gold spans, not the first.")


def newest_checkpoint(run_dir: Path, which: str = "best") -> Path:
    """Newest matching checkpoint. Two runs sharing a directory produce
    `best.ckpt` and `best-v1.ckpt`; picking by name alone silently scores the
    older one."""
    found = sorted(run_dir.glob(f"{which}*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no {which}*.ckpt under {run_dir}")
    return found[-1]


# one loader for every evaluation path, so a checkpoint that scores in the
# benchmark also plots here — see benchmark/predict_dgs_2026.py
from benchmark.predict_dgs_2026 import load_checkpoint  # noqa: E402,F401


def _segments(bio, b_id: int, i_id: int, split_on_b: bool = True):
    """Frame labels to spans.

    `split_on_b=True` is `metrics.bio_to_segments` — the 2023 paper's Algorithm 1
    at argmax, where a B closes the open segment and opens a new one. That is the
    decoder everything now uses.

    `split_on_b=False` reproduces upstream's `likeliest_probs_to_segments`, which
    never looks at B and merges any contiguous run of non-O. It is kept only so
    the plot can show, side by side, what the old metrics were scoring.
    """
    from metrics import bio_to_segments

    if split_on_b:
        return [(s["start"], s["end"] + 1) for s in bio_to_segments(bio, b_id, i_id)]

    spans, start = [], None
    for frame, value in enumerate(bio):
        if value in (b_id, i_id):
            if start is None:
                start = frame
        elif start is not None:
            spans.append((start, frame))
            start = None
    if start is not None:
        spans.append((start, len(bio)))
    return spans


def _to_seconds(spans, fps: float) -> list[tuple[float, float]]:
    """Frame spans to seconds. Every row is stored in seconds so a 25fps
    reference model can share an axis with a 50fps prediction."""
    return [(start / fps, end / fps) for start, end in spans]


def add_reference(records: list[dict], predictions: Path, key: str,
                  level_key: str = "pred"):
    """Merge a benchmark prediction JSON in as the row named `key`.

    The file is whatever `benchmark/predict_dgs_2023.py` or
    `predict_dgs_2026.py` wrote: each carries its own frame rate and its own
    decoding, so a row shows that model exactly as it was benchmarked. Clips the
    file does not cover simply get no row.
    """
    import json

    payload = json.loads(Path(predictions).read_text())
    # DGS ids come in two forms for the same clip, `1429910_a` and
    # `1429910-16075041-16115817_a`; index both so neither side has to know which
    def keys(clip_id: str):
        stem, _, person = clip_id.rpartition("_")
        return {clip_id, f"{stem.split('-')[0]}_{person}"}

    by_id = {key: clip for clip in payload["clips"] for key in keys(clip["id"])}
    matched = 0
    for record in records:
        clip = next((by_id[key] for key in keys(record["id"]) if key in by_id), None)
        if clip is None:
            continue
        fps = clip["fps"]
        record[key] = {
            our: _to_seconds([(s["start"], s["end"] + 1) for s in spans], fps)
            for our, spans in clip[level_key].items()}
        matched += 1
    print(f"reference {payload.get('model', predictions)}: "
          f"matched {matched}/{len(records)} clips")
    return matched


def predict_clip(model, pose, fps: float, velocity: bool, device: str):
    """Full-clip forward, returning per-level argmax BIO in upstream ids."""
    import torch

    from sign_language_segmentation.utils.pose import (compute_velocity,
                                                       preprocess_pose)

    pose = preprocess_pose(pose)
    data = pose.body.data.filled(0)[:, 0, :, :3].astype(np.float32)
    times = np.arange(len(data), dtype=np.float32) / fps
    if velocity:
        data = np.concatenate([data, compute_velocity(data, times)], axis=-1)

    with torch.no_grad():
        log_probs = model(torch.from_numpy(data).unsqueeze(0).to(device),
                          timestamps=torch.from_numpy(times).unsqueeze(0).to(device))
    return ({name: log_probs[name][0].cpu().numpy().argmax(-1)
             for name in log_probs}, len(data), times)


def plot_clips(records: list[dict], out_path: Path, title: str = "",
               level: str = "phrase", seconds: float = 120.0):
    """One row per clip: gold ribbon above, predicted ribbon below.

    Clips are cropped to the first `seconds` so segments stay wide enough to see;
    a 40-minute video drawn whole is a grey smear.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Rows are chosen per record, so a plot can carry a reference model or not.
    # Spans are already in seconds, which is what lets a 25fps reference sit
    # under a 50fps prediction on one axis.
    rows = [row for row in ROWS if any(row[0] in record for record in records)]
    height = max(2.5, 0.35 * len(rows) * len(records))
    figure, axes = plt.subplots(len(records), 1, figsize=(14, height),
                                squeeze=False, sharex=True)
    for axis, record in zip(axes[:, 0], records):
        for offset, (key, label, colour) in enumerate(rows):
            for start, end in record.get(key, {}).get(level, []):
                if start > seconds:
                    continue
                # a white edge is the only thing that separates two segments
                # that touch, and most of them do
                axis.barh(-offset, min(end, seconds) - start, left=start,
                          height=0.62, color=colour, edgecolor="white",
                          linewidth=0.5)
        axis.set_ylim(-len(rows) - 0.6, 0.6)
        axis.set_yticks([-i for i in range(len(rows))])
        axis.set_yticklabels([row[1] for row in rows], fontsize=6)
        axis.set_ylabel(record["id"][:18], fontsize=6, rotation=0,
                        ha="right", va="center")
        axis.set_xlim(0, seconds)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
    axes[-1, 0].set_xlabel("seconds")
    figure.suptitle(title or f"{level} segmentation", fontsize=10)
    # the three rows are easy to confuse, and the whole point is the difference
    # between them, so spell it out on the figure itself
    caption = "\n".join([CAPTION[row[0]] for row in rows] + [FOOTER])
    figure.text(0.5, 0.004, caption, ha="center", va="bottom", fontsize=7,
                color="#4a5568")
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=130)
    plt.close(figure)
    return out_path


def collect(model: "Path | object", dataset: str, clips: int, device: str,
            phrase: str = "sentence", quiet: bool = False):
    """Run the model over `clips` validation videos and return plot records.

    `model` is either a checkpoint path, which is loaded, or a live module, which
    is used as it stands — the callback passes the model mid-training so a panel
    costs one forward per clip rather than a checkpoint reload.
    """
    import torch
    from pose_format import Pose

    from sign_language_segmentation.metrics import bio_labels_to_segments
    from sign_language_segmentation.utils.bio import (BIO,
                                                      create_bio_from_times)

    if isinstance(model, (str, Path)):
        checkpoint = Path(model)
        model = load_checkpoint(checkpoint, device)
        if not quiet:
            print(f"checkpoint {checkpoint}")
    velocity = tuple(getattr(model.hparams, "pose_dims", (50, 6)))[1] >= 6
    if not quiet:
        print(f"velocity   {'on' if velocity else 'off'}")

    if dataset == "youtube_25":
        from datasets.youtube_sl25 import load as source
        specs = list(source.clip_specs("dev"))[:clips]
    else:
        from datasets.public_dgs_corpus import load as source
        specs = list(source.clip_specs_native("dev"))[:clips]

    records = []
    for spec in specs:
        with open(spec["pose_path"], "rb") as handle:
            pose = Pose.read(handle.read())
        try:
            bios, total, times = predict_clip(model, pose, spec["fps"],
                                              velocity, device)
        except Exception as error:
            print(f"  skipping {spec['id']}: {type(error).__name__}: {error}")
            continue

        if dataset == "youtube_25":
            spans = {"phrase": spec["sentences"], "sign": []}
        else:
            spans = source.sign_phrase_spans(spec["sentences"], phrase=phrase)

        fps = spec["fps"]
        record = {"id": spec["id"], "fps": fps, "frames": total, "units": "seconds",
                  # kept for the detail view, which renders a skeleton when the
                  # clip has no video
                  "pose_path": str(spec["pose_path"]),
                  "gold": {}, "pred": {}, "pred_upstream": {}}
        for upstream, ours in LEVELS.items():
            gold_ms = [{"start": s["start_time"] * 1000, "end": s["end_time"] * 1000}
                       for s in spans.get(ours, [])]
            gold = create_bio_from_times(gold_ms, times * 1000)
            # gold keeps upstream's own gold decoder, which splits at every B:
            # in gold, adjacent B frames mean adjacent one-frame segments
            record["gold"][ours] = _to_seconds(
                [(s["start"], s["end"] + 1) for s in bio_labels_to_segments(
                    torch.from_numpy(gold.astype("int64")))], fps)
            record["pred"][ours] = _to_seconds(
                _segments(bios[upstream], BIO["B"], BIO["I"]), fps)
            # what the metrics scored before the decoder was fixed
            record["pred_upstream"][ours] = _to_seconds(
                _segments(bios[upstream], BIO["B"], BIO["I"], split_on_b=False), fps)
        records.append(record)
        if quiet:
            continue
        print(f"  {record['id']:<20} {total:>7} frames  "
              f"gold {len(record['gold']['phrase']):>4}  "
              f"ours {len(record['pred']['phrase']):>5}  "
              f"upstream decode {len(record['pred_upstream']['phrase']):>4}",
              flush=True)

    if quiet or not records:
        return records
    counts = np.array([[len(r["gold"]["phrase"]), len(r["pred"]["phrase"]),
                        len(r["pred_upstream"]["phrase"])] for r in records])
    gold_n, ours, upstream = counts.sum(0)
    print(f"\n{len(records)} clips — phrase % (#pred/#gold), optimal 1:")
    print(f"  ours (2023 Algorithm 1)        {ours / gold_n:.3f}")
    print(f"  upstream decode (B ignored)    {upstream / gold_n:.3f}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run directory name under dist/")
    parser.add_argument("--datasets", default="youtube_25,dgs_corpus",
                        help="comma-separated validation datasets, one plot each")
    parser.add_argument("--clips", type=int, default=20,
                        help="clips per dataset")
    parser.add_argument("--which", default="best", choices=["best", "last"])
    parser.add_argument("--level", default="phrase", choices=["phrase", "sign"])
    parser.add_argument("--phrase", default="glosses",
                        choices=["glosses", "sentence"],
                        help="which DGS phrase definition the gold uses; ignored "
                             "for YouTube, whose phrase is the subtitle cue")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference", action="append", default=[],
                        metavar="ROW=PATH",
                        help="benchmark prediction JSON to draw as a reference "
                             "row; repeatable, e.g. "
                             "--reference ref_2023=benchmark/predictions/"
                             "dgs_validation_2023_E4s-1.json")
    parser.add_argument("--frames", type=int, default=1024,
                        help="window length of the detail view, in frames")
    parser.add_argument("--zoom-clips", type=int, default=4,
                        help="clips to draw in the detail view (0 = skip it)")
    parser.add_argument("--reuse", action="store_true",
                        help="replot from the cached predictions beside the figure")
    args = parser.parse_args()

    import json

    run_dir = DIST / args.run
    for dataset in [name.strip() for name in args.datasets.split(",") if name.strip()]:
        # named by what each view fixes: the overview spans a wall-clock window,
        # the detail a fixed number of frames, whose duration depends on the fps
        overview = run_dir / f"segments_{dataset}_{args.level}_{args.seconds:.0f}s.png"
        # inference over whole videos is minutes on CPU; restyling the figure
        # should not pay that again. `units` guards against an older cache that
        # stored frames instead of seconds.
        cache = overview.with_suffix(".json")
        records = None
        if args.reuse and cache.exists():
            cached = json.loads(cache.read_text())
            if cached and cached[0].get("units") == "seconds":
                records = cached
                print(f"reusing {len(records)} cached predictions from {cache}")
            else:
                print(f"ignoring stale cache {cache} (pre-seconds format)")
        if records is None:
            checkpoint = newest_checkpoint(run_dir, args.which)
            records = collect(checkpoint, dataset, args.clips, args.device,
                              phrase=args.phrase)
            cache.write_text(json.dumps(records))

        for spec in args.reference:
            key, _, path = spec.partition("=")
            if not path:
                raise SystemExit(f"--reference wants ROW=PATH, got {spec!r}")
            if key not in {row[0] for row in ROWS}:
                raise SystemExit(f"unknown row {key!r}; "
                                 f"pick from {[row[0] for row in ROWS]}")
            add_reference(records, Path(path), key)

        plot_clips(records, overview, level=args.level,
                   title=f"{args.run} / {args.which} — {dataset} dev ({args.level})",
                   seconds=args.seconds)
        print(f"wrote {overview}")

        if args.zoom_clips:
            detail = run_dir / f"segments_{dataset}_{args.level}_{args.frames}f.png"
            # real frames beat a skeleton, so clips that have an mp4 go first and
            # a pose render only appears when there are not enough of them. DGS
            # has no video at all, so every DGS panel is a render.
            chosen = sorted(records, key=lambda r: video_path(r["id"]) is None)
            plot_detail(chosen[:args.zoom_clips], detail, dataset,
                        level=args.level, frames=args.frames,
                        title=f"{args.run} / {args.which} — {dataset} dev "
                              f"({args.level}), {args.frames}-frame detail")
            print(f"wrote {detail}\n")




# --- detail view: keyframes + subtitle text over one 1024-frame window --------
#
# Styled after Figure 1 of Segment, Embed and Align (arXiv 2512.08094): a strip
# of keyframes sampled at the midpoint of each gold span, over the ribbon
# tracks. Theirs compares subtitle timings to signing; ours compares predicted
# segmentation to gold, so the tracks differ but the reading does not.

#: YouTube mp4s live in a second copy of the corpus, split by whether the video
#: is ASL. The VGG copy we train from carries poses and subtitles only.
VIDEO_ROOTS = ("/shares/iict-sp2.ebling.cl.uzh/common/YouTube-SL-25_Colin_Leong/"
               "ase/downloads",
               "/shares/iict-sp2.ebling.cl.uzh/common/YouTube-SL-25_Colin_Leong/"
               "non-ase/downloads")


def video_path(clip_id: str):
    """The mp4 for a clip, or None. About 85% of YouTube dev clips have one; the
    Public DGS Corpus archive holds no video at all, only `.pose` and `.eaf`."""
    import os

    for root in VIDEO_ROOTS:
        candidate = os.path.join(root, clip_id, f"{clip_id}.mp4")
        if os.path.exists(candidate):
            return candidate
    return None


def video_frames(path: str, times):
    """Decode one RGB frame per requested timestamp, seeking rather than scanning."""
    import av

    out = []
    container = av.open(path)
    stream = container.streams.video[0]
    try:
        for want in times:
            container.seek(int(want / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                if float(frame.pts * stream.time_base) >= want - 0.5:
                    out.append(frame.to_ndarray(format="rgb24"))
                    break
            else:
                out.append(None)
    finally:
        container.close()
    return out


def pose_frames(pose_path: str, fps: float, times):
    """Skeleton renders, for clips with no video — every DGS clip, and the ~15%
    of YouTube dev clips whose mp4 is missing.

    The whole window is read once and indexed, not reopened per keyframe: at one
    frame per second that is forty-odd reads of an 85 MB file over the share.

    Body and hands only. These files also carry POSE_WORLD_LANDMARKS, metric
    coordinates centred on the hips that share nothing with the image frame; drawn
    alongside the rest it appears as a second body floating beside the signer.
    """
    import numpy as np
    from pose_format import Pose

    wanted = [max(0, int(round(t * fps))) for t in times]
    first, last = min(wanted), max(wanted)
    with open(pose_path, "rb") as handle:
        pose = Pose.read(handle, start_frame=first, end_frame=last + 1)
    block = np.ma.filled(pose.body.data, np.nan)[:, 0]

    edges, offset = [], 0
    for component in pose.header.components:
        if component.name == "POSE_LANDMARKS":
            # 0-24 is head through hips. 0-10 are nose, eyes, ears and mouth, so
            # dropping them leaves a headless torso; only the legs (25+) go,
            # since `pose_hide_legs` zeroes them anyway and they would stretch
            # the thumbnail vertically for no gain.
            edges += [(a + offset, b + offset) for a, b in component.limbs
                      if a <= 24 and b <= 24]
            # MediaPipe connects the face points to each other and the shoulders
            # to each other, but never the two groups, so the head otherwise
            # floats above the torso. Nose to each shoulder stands in for a neck.
            edges += [(0 + offset, 11 + offset), (0 + offset, 12 + offset)]
        elif component.name in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
            edges += [(a + offset, b + offset) for a, b in component.limbs]
        offset += len(component.points)

    return [(block[min(index - first, len(block) - 1)][:, :2], edges)
            for index in wanted]


def keyframe_times(start: float, end: float, per_second: float = 1.0) -> list:
    """Uniform sample times, at least `per_second` of them per second of window.

    Uniform rather than at each gold span's midpoint: span midpoints cluster
    where the spans do, leaving gaps exactly where the model is disagreeing with
    the gold, which is the part worth looking at.
    """
    count = max(2, int(round((end - start) * per_second)))
    step = (end - start) / count
    return [start + (i + 0.5) * step for i in range(count)]


def draw_keyframes(axis, items, start: float, end: float) -> None:
    """Lay the keyframes edge to edge across the window, in time order.

    Each sits in its own inset rather than being drawn with a data extent,
    because an extent stretches the image to the box: a 640x360 frame across four
    seconds comes out badly distorted. An inset has a fixed shape in figure
    space, so `aspect="equal"` letterboxes and the signer keeps their proportions.
    """
    import numpy as np

    count = max(len(items), 1)
    width = 1.0 / count
    for position, item in enumerate(items):
        if item is None:
            continue
        inset = axis.inset_axes([position * width, 0.0, width, 1.0])
        inset.set_xticks([]); inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color("#e2e8f0"); spine.set_linewidth(0.4)
        if isinstance(item, tuple):
            points, edges = item
            good = ~np.isnan(points).any(axis=1)
            drawn = {i for edge in edges for i in edge}
            if good.any() and drawn:
                x, y = points[:, 0], -points[:, 1]
                for a, b in edges:
                    if good[a] and good[b]:
                        inset.plot([x[a], x[b]], [y[a], y[b]], color="#2d3748",
                                   linewidth=0.7, solid_capstyle="round")
                shown = [i for i in drawn if good[i]]
                if shown:
                    inset.set_xlim(min(x[shown]) - 12, max(x[shown]) + 12)
                    inset.set_ylim(min(y[shown]) - 12, max(y[shown]) + 12)
                inset.set_aspect("equal")
        else:
            inset.imshow(item, aspect="equal")

    axis.set_ylim(0, 1)
    axis.set_xticks([]); axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def cue_texts(dataset: str, clip_id: str) -> list:
    """Subtitle text with timings, for labelling gold spans.

    YouTube only. `datasets/youtube_sl25/load.py` deliberately drops the text
    when caching cues — the model never sees it — so it is re-parsed here, for
    one clip at a time, purely for the figure. The DGS loader carries no
    German text at all, so its spans go unlabelled.
    """
    if dataset != "youtube_25":
        return []
    import re

    from datasets.youtube_sl25 import load as yt

    path = yt.video_index().get(clip_id, {}).get("subtitle")
    if not path:
        return []
    out = []
    for block in Path(path).read_text(errors="ignore").split("\n\n"):
        match = yt._TIMESTAMP.search(block)
        if not match:
            continue
        body = block[match.end():].strip()
        if not body or re.match(r"^\s*[\[(♪]", body):
            continue
        out.append((yt._seconds(*match.groups()[:4]),
                    yt._seconds(*match.groups()[4:]),
                    " ".join(body.split())))
    return out


def pick_window(record, level: str, frames: int) -> tuple:
    """The `frames`-long window holding the most gold spans, in seconds.

    A window chosen by position usually lands on silence; one chosen by density
    shows the case the figure is meant to show — several boundaries close
    together, which is where segmentation is actually hard.
    """
    fps = record["fps"]
    length = frames / fps
    spans = record["gold"][level]
    if not spans:
        return 0.0, length
    best, count = spans[0][0], 0
    for start, _ in spans:
        inside = sum(1 for a, b in spans if a >= start and b <= start + length)
        if inside > count:
            best, count = start, inside
    return best, best + length


def plot_detail(records, out_path: Path, dataset: str, level: str = "phrase",
                frames: int = 1024, title: str = "", max_texts: int = 6,
                per_second: float = 1.0, inches_per_frame: float = 1.0):
    """One panel per clip: a keyframe strip above, ribbons below, text on gold.

    The canvas widens with the number of keyframes rather than the other way
    around, so a keyframe is always about `inches_per_frame` wide however long
    the window is. At one frame per second a 1024-frame window is 41 s on
    YouTube and 20 s on DGS, so the DGS figures come out half as wide.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in ROWS if any(row[0] in record for record in records)]
    shots, windows = [], []
    for record in records:
        start, end = pick_window(record, level, frames)
        times = keyframe_times(start, end, per_second)
        video = video_path(record["id"])
        frames_in = None
        if video:
            try:
                frames_in = video_frames(video, times)
            except Exception as error:
                # a few of the downloads are truncated or otherwise unreadable;
                # one bad file must not cost the whole figure
                print(f"  {record['id']}: {type(error).__name__} reading the "
                      f"video, falling back to a pose render")
                video, frames_in = None, None
        if frames_in is None:
            frames_in = pose_frames(record["pose_path"], record["fps"], times)
        shots.append(frames_in)
        windows.append((start, end, video))

    # The strip is sized to the frames themselves rather than to a fixed row
    # height: `aspect="equal"` letterboxes inside its box, so any spare height is
    # white space above and below the signer. One aspect per panel, since a panel
    # is one clip.
    def shape(items):
        for item in items:
            if item is not None and not isinstance(item, tuple):
                return item.shape[0] / item.shape[1]
        return 1.15                       # skeletons are drawn roughly portrait

    widest = max(len(shot) for shot in shots)
    strips = [inches_per_frame * shape(shot) for shot in shots]
    ribbon = 1.5
    figure = plt.figure(figsize=(max(12.0, widest * inches_per_frame),
                                 sum(strips) + ribbon * len(records) + 1.0))
    # Nested, because a flat gridspec spaces every gap alike. The strip and its
    # ribbons are one object and sit close together; separate clips need air
    # between them, or ten panels read as one wall.
    outer = figure.add_gridspec(len(records), 1,
                                height_ratios=[strip + ribbon for strip in strips],
                                hspace=0.55)
    grid = [outer[i].subgridspec(2, 1, height_ratios=[strips[i], ribbon],
                                 hspace=0.08)
            for i in range(len(records))]

    for index, record in enumerate(records):
        start, end, video = windows[index]
        top = figure.add_subplot(grid[index][0])
        top.set_xlim(start, end)
        draw_keyframes(top, shots[index], start, end)
        top.set_title(f"{record['id']}   {start:.0f}–{end:.0f} s   "
                      f"({'video' if video else 'pose render'}, "
                      f"{len(shots[index])} keyframes)", fontsize=8, loc="left")

        axis = figure.add_subplot(grid[index][1])
        for offset, (key, label, colour) in enumerate(rows):
            for a, b in record.get(key, {}).get(level, []):
                if b <= start or a >= end:
                    continue
                axis.barh(-offset, min(b, end) - max(a, start), left=max(a, start),
                          height=0.62, color=colour, edgecolor="white",
                          linewidth=0.5)
        spans = [(a, b) for a, b in record["gold"][level] if b > start and a < end]
        # two heights, alternating: neighbouring subtitle lines are long and
        # would otherwise print on top of each other
        for position, ((a, b), text) in enumerate(
                zip(spans, span_texts(dataset, record["id"], spans, max_texts))):
            if text:
                axis.text((a + b) / 2, 0.5 + 0.85 * (position % 2), text,
                          fontsize=6, ha="center", va="bottom", color="#4a5568")
        axis.set_ylim(-len(rows) - 0.6, 2.6)
        axis.set_yticks([-i for i in range(len(rows))])
        axis.set_yticklabels([row[1] for row in rows], fontsize=7)
        axis.set_xlim(start, end)
        axis.set_xlabel("seconds", fontsize=7)
        axis.tick_params(labelsize=6, labelbottom=True)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)

    figure.suptitle(title or f"{dataset} — {frames}-frame detail", fontsize=10)
    figure.text(0.5, 0.004, DETAIL_CAPTION, ha="center", va="bottom",
                fontsize=7, color="#4a5568")
    figure.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out_path


def span_texts(dataset: str, clip_id: str, spans, limit: int) -> list:
    """The subtitle line overlapping each gold span, truncated, thinned to
    `limit` labels so a dense window stays readable."""
    cues = cue_texts(dataset, clip_id)
    step = max(1, len(spans) // max(limit, 1))
    out = []
    for position, (a, b) in enumerate(spans):
        if position % step or not cues:
            out.append("")
            continue
        hit = max(cues, key=lambda c: min(b, c[1]) - max(a, c[0]), default=None)
        text = hit[2] if hit and min(b, hit[1]) > max(a, hit[0]) else ""
        out.append(text[:38] + ("…" if len(text) > 38 else ""))
    return out

if __name__ == "__main__":
    main()
