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


def newest_checkpoint(run_dir: Path, which: str = "best") -> Path:
    """Newest matching checkpoint. Two runs sharing a directory produce
    `best.ckpt` and `best-v1.ckpt`; picking by name alone silently scores the
    older one."""
    found = sorted(run_dir.glob(f"{which}*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no {which}*.ckpt under {run_dir}")
    return found[-1]


def load_checkpoint(checkpoint: Path, device: str = "cpu"):
    """Load a checkpoint that our training wrote.

    `ValidationMetricsModel` keeps the class-frequency weights as buffers on its
    loss functions, so its checkpoints carry `sign_loss_fn.weight` and
    `phrase_loss_fn.weight` that plain `PoseTaggingModel` has never heard of.
    They matter to training and not at all to a forward pass, so they are dropped
    — but only those two. Any other unexpected key is still an error, because a
    silently half-loaded model would produce plausible nonsense.
    """
    import torch

    from sign_language_segmentation.model.model import PoseTaggingModel

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    dropped = {key for key in state["state_dict"]
               if key in ("sign_loss_fn.weight", "phrase_loss_fn.weight")}
    for key in dropped:
        state["state_dict"].pop(key)
    patched = checkpoint.parent / f".{checkpoint.stem}.plot.ckpt"
    torch.save(state, patched)
    try:
        model = PoseTaggingModel.load_from_checkpoint(
            checkpoint_path=str(patched), map_location=device)
    finally:
        patched.unlink(missing_ok=True)
    if dropped:
        print(f"dropped training-only keys: {', '.join(sorted(dropped))}")
    return model.to(device).eval()


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
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference", action="append", default=[],
                        metavar="ROW=PATH",
                        help="benchmark prediction JSON to draw as a reference "
                             "row; repeatable, e.g. "
                             "--reference ref_2023=benchmark/predictions/"
                             "dgs_validation_2023_E4s-1.json")
    parser.add_argument("--reuse", action="store_true",
                        help="replot from the cached predictions beside the figure")
    args = parser.parse_args()

    import json

    run_dir = DIST / args.run
    for dataset in [name.strip() for name in args.datasets.split(",") if name.strip()]:
        out = run_dir / f"segments_{dataset}_{args.level}.png"
        # inference over whole videos is minutes on CPU; restyling the figure
        # should not pay that again. `units` guards against an older cache that
        # stored frames instead of seconds.
        cache = out.with_suffix(".json")
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
            records = collect(checkpoint, dataset, args.clips, args.device)
            cache.write_text(json.dumps(records))

        for spec in args.reference:
            key, _, path = spec.partition("=")
            if not path:
                raise SystemExit(f"--reference wants ROW=PATH, got {spec!r}")
            if key not in dict((row[0], row) for row in ROWS):
                raise SystemExit(f"unknown row {key!r}; "
                                 f"pick from {[row[0] for row in ROWS]}")
            add_reference(records, Path(path), key)
        plot_clips(records, out, level=args.level,
                   title=f"{args.run} / {args.which} — {dataset} dev ({args.level})",
                   seconds=args.seconds)
        print(f"wrote {out}\n")


if __name__ == "__main__":
    main()
