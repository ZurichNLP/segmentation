"""Frame-rate augmentation that keeps every training window at `num_frames`.

Replaces upstream's `fps_aug` (`sign_language_segmentation/datasets/common.py`)
when `--fps-aug on`; upstream's is never used. It had two problems:

  * it only lowers the frame rate, drawing 25-50 fps uniformly, so it never
    touches 24-25 fps video and resamples only 14% of 30 fps clips — close to a
    no-op on YouTube-SL-25
  * in 5% of clips its "tempo stretch" builds the BIO labels from rescaled
    timestamps while the poses stay in real time: on a 30 fps clip stretched to
    60, 296 of 1024 frames get a label from a different instant

Per training draw:

  1. a target rate from a log-normal centred on `PEAK_FPS`, truncated to
     `MIN_FPS`-`MAX_FPS`, so most draws stay near the corpus' own 25-30 fps and
     a few reach either end
  2. the video is cut into `k` equal tiles of `num_frames` frames at that rate,
     `k` a whole number, the rate nudged just enough for the tiles to fit the
     video exactly, and one tile is picked uniformly
  3. `num_frames` poses are sampled evenly across the tile, each interpolated
     linearly between the two nearest native frames. A keypoint missing in
     either neighbour takes the nearer frame instead, so a missing hand is never
     blended halfway towards the origin
  4. timestamps and BIO labels are both computed from the same real times, so
     each frame's label is what the gold says at the instant that pose shows

Tiles rather than a random start: a fixed-length window started uniformly at
random sees the first and last window of every video less often than its middle,
and 23% of YouTube-SL-25's frames sit in those margins. Tiles partition the
video, so each frame is inside the drawn window with probability exactly `1/k`;
`sampling_weights` then gives each video `num_frames / E[1/k]`, the expectation
over this same rate distribution, and every native frame in the corpus becomes
equally likely to be seen.

What stays fixed:

  * a video with at least `num_frames` native frames always yields exactly
    `num_frames`. A shorter one is resampled to fill `num_frames` if that needs
    no more than `MAX_FPS` (or its own rate, if higher), otherwise to that rate
    and padded as today — never below its native length
  * evaluation: `load_window` refuses any split but training

One inherited property: `preprocess_pose` centres and scales by shoulder
statistics averaged over whatever frames were read, so a window covering more
native frames is normalised over more of them. Upstream's downsampling has the
same property; interpolation reads at most one frame beyond the tile.

At a rate equal to the native rate and a video an exact multiple of `num_frames`
long, the output is identical to upstream's un-augmented window, poses, times and
labels alike — which is how this is tested.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

import numpy as np

#: target rates: log-normal with median `PEAK_FPS` — the peak on a log-fps axis,
#: so doubling and halving the rate are equally likely — truncated to the range
PEAK_FPS = 30.0
LOG_SIGMA = 0.2
MIN_FPS = 20.0
MAX_FPS = 60.0

#: rates at which `sampling_weights` evaluates the expectation over draws
QUADRATURE = 1024

#: range checks tolerate float error when a rate lands exactly on a bound
_TOLERANCE = 1e-9

#: tempo stretch, upstream's constants: in this share of training windows the
#: clock the model reads runs as if the frames came at one of these rates
TEMPO_PROBABILITY = 0.05
TEMPO_FPS = (24.0, 30.0, 60.0)


def stretch_clock(timestamps, rate: float, rng=random):
    """Upstream's tempo stretch, with the labels left where they belong.

    `timestamps` are a training window's real times in seconds and `rate` the
    frame rate they were taken at — the native rate, or the resampled one under
    `--fps-aug on`. With probability `TEMPO_PROBABILITY` they are rescaled to run
    at a rate from `TEMPO_FPS`, so consecutive frames sit `1 / tempo` apart on
    the clock, exactly as upstream's tempo branch sets them.

    Only the clock changes. Poses and labels are what the real times gave, so
    signing looks faster or slower to the attention layers, whose RoPE is the
    one place the model reads timestamps, while every boundary stays on its
    pose. Upstream built its labels from the rescaled clock instead, which is
    the bug this avoids. Velocity, computed before this from real times, is
    left alone too.
    """
    if rng.random() >= TEMPO_PROBABILITY:
        return timestamps
    return timestamps * (rate / rng.choice(TEMPO_FPS))


def draw_rate(rng=random) -> float:
    """One target rate. Rejection sampling; about 98% of draws are accepted.

    Uses Python's `random`, which PyTorch reseeds in every DataLoader worker —
    NumPy's global generator would repeat the same rates across workers.
    """
    while True:
        rate = PEAK_FPS * math.exp(rng.gauss(0.0, LOG_SIGMA))
        if MIN_FPS <= rate <= MAX_FPS:
            return rate


def rate_quantiles(count: int = QUADRATURE) -> np.ndarray:
    """`count` evenly spaced quantiles of the distribution `draw_rate` samples."""
    normal = NormalDist()
    low = normal.cdf(math.log(MIN_FPS / PEAK_FPS) / LOG_SIGMA)
    high = normal.cdf(math.log(MAX_FPS / PEAK_FPS) / LOG_SIGMA)
    probs = low + (np.arange(count) + 0.5) / count * (high - low)
    return PEAK_FPS * np.exp(LOG_SIGMA * np.array([normal.inv_cdf(p) for p in probs]))


def plan(total_frames, fps, rate, num_frames: int):
    """Number of tiles and the rate actually used, for a drawn target `rate`.

    Vectorised over any broadcastable arrays, so the sampler weights and each
    training draw go through this one function and cannot disagree.

    `k` is whichever of the two whole numbers around `duration x rate /
    num_frames` keeps the rate inside `MIN_FPS`-`MAX_FPS` and nearer the draw;
    one of them always does. A video too short for even one window at the drawn
    rate gets one tile, at the rate that fills `num_frames`, capped as described
    in the module docstring.
    """
    total = np.asarray(total_frames, dtype=np.float64)
    fps = np.asarray(fps, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)

    whole = num_frames * fps / total        # rate at which one window is the video
    windows = rate / whole
    below = np.maximum(np.floor(windows), 1.0)
    above = np.maximum(np.ceil(windows), 1.0)
    rate_below, rate_above = whole * below, whole * above

    def fits(candidate):
        return ((candidate >= MIN_FPS * (1 - _TOLERANCE))
                & (candidate <= MAX_FPS * (1 + _TOLERANCE)))

    nearer_above = np.abs(np.log(rate_above / rate)) < np.abs(np.log(rate_below / rate))
    use_above = fits(rate_above) & (~fits(rate_below) | nearer_above)
    tiles = np.where(use_above, above, below)
    actual = whole * tiles

    short = windows < 1.0
    tiles = np.where(short, 1.0, tiles)
    actual = np.where(short, np.minimum(np.maximum(MAX_FPS, fps), whole), actual)
    return tiles.astype(np.int64), actual


def window_length(total_frames: int, fps: float, tiles: int, actual: float,
                  num_frames: int) -> int:
    """Output frames: `num_frames`, unless a short video cannot fill it."""
    return int(min(num_frames, round(total_frames * actual / (fps * tiles))))


def sampling_weights(total_frames, fps, num_frames: int,
                     chunk: int = 2048) -> np.ndarray:
    """Per-video sampler weight making every native frame equally likely per draw.

    A frame is in the drawn window with probability `weight / sum x E[1/k]`, so
    the weight is `num_frames / E[1/k]`. The `num_frames` factor puts these on
    the same scale as the frame counts un-augmented datasets are weighted by —
    both then give every frame a chance of `num_frames / sum` — so the two can
    share one sampler.
    """
    total = np.asarray(total_frames, dtype=np.float64)
    fps = np.asarray(fps, dtype=np.float64)
    rates = rate_quantiles()
    inverse = np.empty(len(total))
    for start in range(0, len(total), chunk):
        stop = start + chunk
        tiles, _ = plan(total[start:stop, None], fps[start:stop, None],
                        rates[None, :], num_frames)
        inverse[start:stop] = (1.0 / tiles).mean(axis=1)
    return num_frames / inverse


def widened(length: int, drop_rate: float) -> int:
    """Frames to lay out so that dropping `drop_rate` of them leaves `length`.

    Upstream drops `int((n - 2) x rate)` of a window's `n` frames. Solving for
    the `n` that leaves `length` gives this, so the dropped share is upstream's.
    """
    if drop_rate <= 0.0 or length <= 2:
        return length
    return length + int((length - 2) * drop_rate / (1.0 - drop_rate))


def load_window(pose_path: str, fps: float, total_frames: int,
                signs: list[dict], sentences: list[dict], split,
                num_frames: int, velocity: bool, frame_dropout: float,
                body_part_dropout: float, tempo: bool = False,
                resample: bool = True, rng=random) -> dict:
    """One training window, in the shape `load_and_augment` returns.

    `resample` picks the window: a resampled tile as described above, or, when
    False, upstream's own un-augmented window — `num_frames` native frames from
    a uniformly random start — reproduced exactly.

    Frame dropout drops the same share of middle frames as upstream — a rate
    drawn uniformly from 0 to `frame_dropout`, never the first or last frame —
    with two differences. The window is widened first (`widened`), so what is
    left is still `length` frames rather than a shorter window padded back: the
    model has no attention mask, so padding would be attended to. And labels come
    from the kept frames' real times; upstream's un-augmented path labels the
    survivors as if still evenly spaced, so with 15% dropout its last frames are
    labelled from 15% earlier in the window.

    Body-part dropout and velocity follow upstream's code. `tempo` applies
    `stretch_clock` last, once labels and velocity have their real times.
    """
    import torch
    from pose_format import Pose

    from sign_language_segmentation.utils.bio import create_bio_from_times
    from sign_language_segmentation.utils.pose import (compute_velocity,
                                                       preprocess_pose)

    if str(split) != "train":
        raise ValueError(f"load_window builds training windows only, got {split!r}")

    # Frame dropout draws its rate first, because it widens the window: `count`
    # positions are laid out and `count - length` of them dropped, so the window
    # still holds `length` frames — never fewer than it would without dropout.
    drop_rate = rng.uniform(0.0, frame_dropout) if frame_dropout > 0.0 else 0.0

    if resample:
        tiles, actual = plan(total_frames, fps, draw_rate(rng), num_frames)
        tiles, actual = int(tiles), float(actual)
        length = window_length(total_frames, fps, tiles, actual, num_frames)
        count = widened(length, drop_rate)
        tile = rng.randrange(tiles)

        # Positions in native-frame units. The tile spans `span` frame slots,
        # frame m sitting at the centre of slot m, and samples sit at the centres
        # of `count` equal sub-slots. Widening only packs them denser inside the
        # same tile, so sampling stays frame-uniform. Times keep the unclamped
        # positions so they stay strictly increasing; only the pose lookup is
        # held inside the video.
        span = total_frames / tiles
        positions = tile * span - 0.5 + (np.arange(count) + 0.5) * (span / count)
    else:
        # upstream's window, same draw: `random.randint(0, total - num_frames)`,
        # widened by dropout where the video has room. A video shorter than
        # `num_frames` has none, and is left whole rather than made shorter.
        length = min(total_frames, num_frames)
        count = (min(total_frames, widened(length, drop_rate))
                 if total_frames >= num_frames else length)
        start = rng.randint(0, total_frames - count) if total_frames > count else 0
        span = float(count)
        positions = start + np.arange(count, dtype=np.float64)

    if count > length:
        # upstream's rule: never the first or last frame
        dropped = rng.sample(range(1, count - 1), count - length)
        keep = np.ones(count, dtype=bool)
        keep[dropped] = False
        positions = positions[keep]
    # the kept frames' average rate, which a rescaled clock is measured against
    rate = length * fps / span
    lookup = np.clip(positions, 0.0, total_frames - 1)
    first = int(np.floor(lookup[0]))
    last = int(np.ceil(lookup[-1]))

    with open(pose_path, "rb", buffering=4 * 1024 * 1024) as handle:
        pose = Pose.read(handle, start_frame=first, end_frame=last + 1)
    pose = preprocess_pose(pose)
    body = pose.body.data[:, 0, :, :3]
    data = body.filled(0).astype(np.float32)
    missing = np.ma.getmaskarray(body).any(axis=-1)

    local = lookup - first
    below = np.minimum(np.floor(local).astype(np.int64), len(data) - 1)
    above = np.minimum(below + 1, len(data) - 1)
    weight = (local - below).astype(np.float32)
    poses = (data[below] * (1 - weight[:, None, None])
             + data[above] * weight[:, None, None])
    nearest = np.where(weight < 0.5, below, above)
    gap = missing[below] | missing[above]
    pose_data = np.where(gap[:, :, None], data[nearest], poses).astype(np.float32)

    # float32 offsets over a Python float, as upstream computes `arange / fps`, so
    # an un-resampled window reproduces its times bit for bit
    frame_times = (positions - positions[0]).astype(np.float32) / fps
    frame_times_ms = frame_times * 1000

    if body_part_dropout > 0.0:
        if rng.random() < body_part_dropout:
            pose_data[:, 8:29, :] = 0
        if rng.random() < body_part_dropout:
            pose_data[:, 29:50, :] = 0

    if velocity:
        pose_data = np.concatenate(
            [pose_data, compute_velocity(pose_data, frame_times)], axis=-1)

    # spans relative to the first sample, clipped to the window as upstream does
    start_ms = float(positions[0]) / fps * 1000
    end_ms = (float(positions[0]) + span) / fps * 1000

    def clip(spans):
        return [{"start": max(0, s["start"] - start_ms),
                 "end": min(end_ms - start_ms, s["end"] - start_ms)}
                for s in spans if s["end"] > start_ms and s["start"] < end_ms]

    timestamps = torch.from_numpy(frame_times)
    return {
        "pose": torch.from_numpy(pose_data),
        "timestamps": stretch_clock(timestamps, rate, rng) if tempo else timestamps,
        "bio": {
            "sign": torch.from_numpy(create_bio_from_times(clip(signs), frame_times_ms)).long(),
            "sentence": torch.from_numpy(create_bio_from_times(clip(sentences), frame_times_ms)).long(),
        },
    }


def uses_ours(dataset) -> bool:
    """Should this dataset's windows come from `load_window`?

    Whenever a trick could move frames against their labels: resampling, a
    rescaled clock, or frame dropout, whose labels upstream gets wrong on its
    un-augmented path. Training only. Anything else — plain training, and every
    evaluation — keeps upstream's `load_and_augment` untouched.
    """
    return (str(dataset.split) == "train"
            and (getattr(dataset, "fps_resample", False)
                 or getattr(dataset, "tempo_stretch", False)
                 or getattr(dataset, "frame_dropout", 0.0) > 0.0))


def load_item(dataset, item: dict, rng=random) -> dict:
    """`load_window` for one of a dataset adapter's items, with its settings.

    Flags are read with `getattr` so a dataset built before a flag existed reads
    it as off.
    """
    return load_window(
        pose_path=item["pose_path"], fps=item["fps"],
        total_frames=item["total_frames"], signs=item["glosses"],
        sentences=item["sentences"], split=dataset.split,
        num_frames=dataset.num_frames, velocity=dataset.velocity,
        frame_dropout=dataset.frame_dropout,
        body_part_dropout=dataset.body_part_dropout,
        tempo=getattr(dataset, "tempo_stretch", False),
        resample=getattr(dataset, "fps_resample", False), rng=rng)
