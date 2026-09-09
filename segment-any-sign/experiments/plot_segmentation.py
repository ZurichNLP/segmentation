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

CAPTION = "\n".join([
    "gold (gold) — reference segments, split at every B.",
    "pred (B) (blue) — the model's argmax, split at every B: what the model actually says.",
    "pred (upstream) (pale blue) — the same argmax through likeliest_probs_to_segments, "
    "which ignores B and merges any run of non-O: what the logged metrics score.",
    "White hairlines separate touching segments. One panel per dev video, "
    "first two minutes only."])


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


def _segments(bio: np.ndarray, b_id: int, i_id: int,
              split_on_b: bool = True) -> list[tuple[int, int]]:
    """Frame labels to spans, with or without honouring B.

    `split_on_b=True` starts a new segment at every B, which is what upstream's
    *gold* decoder (`bio_labels_to_segments`) does. `split_on_b=False` groups any
    contiguous non-O run, which is what upstream's *prediction* decoder
    (`likeliest_probs_to_segments`) does — it never looks at B at all.

    The two are not interchangeable, and upstream pairs them: gold split on B,
    prediction not. Wherever phrases are back-to-back, that asymmetry caps the
    prediction count below the gold count no matter how good the model is.
    """
    spans, start = [], None
    for frame, value in enumerate(bio):
        if value == b_id and split_on_b:
            if start is not None:
                spans.append((start, frame))
            start = frame
        elif value in (b_id, i_id):
            if start is None:
                start = frame
        elif start is not None:
            spans.append((start, frame))
            start = None
    if start is not None:
        spans.append((start, len(bio)))
    return spans


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

    # segments are often back-to-back, so a white edge is the only thing that
    # shows where one ends and the next begins
    # gold in gold; the two decodings of one prediction share a hue so it reads
    # as one thing seen two ways, the upstream one muted because it is derived
    rows = (("gold", "#d69e2e"), ("pred", "#2b6cb0"), ("pred_upstream", "#90b4d6"))
    height = max(2.5, 1.0 * len(records))
    figure, axes = plt.subplots(len(records), 1, figsize=(14, height),
                                squeeze=False, sharex=True)
    for axis, record in zip(axes[:, 0], records):
        fps = record["fps"]
        limit = seconds * fps
        for offset, (key, colour) in enumerate(rows):
            for start, end in record[key][level]:
                if start > limit:
                    continue
                axis.barh(-offset, min(end, limit) / fps - start / fps,
                          left=start / fps, height=0.62, color=colour,
                          edgecolor="white", linewidth=0.5)
        axis.set_ylim(-2.6, 0.6)
        axis.set_yticks([0, -1, -2])
        axis.set_yticklabels(["gold", "pred (B)", "pred (upstream)"], fontsize=6)
        axis.set_ylabel(record["id"][:18], fontsize=6, rotation=0,
                        ha="right", va="center")
        axis.set_xlim(0, seconds)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
    axes[-1, 0].set_xlabel("seconds")
    figure.suptitle(title or f"{level} segmentation", fontsize=10)
    # the three rows are easy to confuse, and the whole point is the difference
    # between them, so spell it out on the figure itself
    figure.text(0.5, 0.004, CAPTION, ha="center", va="bottom", fontsize=7,
                color="#4a5568")
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=130)
    plt.close(figure)
    return out_path


def collect(checkpoint: Path, dataset: str, clips: int, device: str,
            phrase: str = "sentence"):
    """Run the model over `clips` validation videos and return plot records."""
    import torch  # noqa: F401
    from pose_format import Pose

    from sign_language_segmentation.utils.bio import (BIO,
                                                      create_bio_from_times)

    model = load_checkpoint(checkpoint, device)
    velocity = tuple(getattr(model.hparams, "pose_dims", (50, 6)))[1] >= 6
    print(f"checkpoint {checkpoint}\nvelocity   {'on' if velocity else 'off'}")

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

        record = {"id": spec["id"], "fps": spec["fps"], "frames": total,
                  "gold": {}, "pred": {}, "pred_upstream": {}}
        for upstream, ours in LEVELS.items():
            gold_ms = [{"start": s["start_time"] * 1000, "end": s["end_time"] * 1000}
                       for s in spans.get(ours, [])]
            gold = create_bio_from_times(gold_ms, times * 1000)
            record["gold"][ours] = _segments(gold, BIO["B"], BIO["I"])
            record["pred"][ours] = _segments(bios[upstream], BIO["B"], BIO["I"])
            # what the logged metrics actually score: B thrown away
            record["pred_upstream"][ours] = _segments(bios[upstream], BIO["B"],
                                                      BIO["I"], split_on_b=False)
        records.append(record)
        gold_n = len(record["gold"]["phrase"])
        print(f"  {record['id']:<20} {total:>7} frames  gold {gold_n:>4}  "
              f"pred(B) {len(record['pred']['phrase']):>5}  "
              f"pred(upstream) {len(record['pred_upstream']['phrase']):>4}",
              flush=True)

    counts = np.array([[len(r["gold"]["phrase"]), len(r["pred"]["phrase"]),
                        len(r["pred_upstream"]["phrase"])] for r in records])
    gold_n, with_b, without_b = counts.sum(0)
    print(f"\n{len(records)} clips — phrase % (#pred/#gold), the logged metric:")
    print(f"  upstream decoder (B ignored)   {without_b / gold_n:.3f}")
    print(f"  same argmax, B honoured        {with_b / gold_n:.3f}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run directory name under dist/")
    parser.add_argument("--dataset", default="youtube_25",
                        choices=["youtube_25", "dgs_corpus"])
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--which", default="best", choices=["best", "last"])
    parser.add_argument("--level", default="phrase", choices=["phrase", "sign"])
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    parser.add_argument("--reuse", action="store_true",
                        help="replot from the cached predictions beside --out")
    args = parser.parse_args()

    import json

    run_dir = DIST / args.run
    out = Path(args.out or run_dir / f"segments_{args.dataset}_{args.level}.png")
    # inference over whole videos is minutes on CPU; restyling the figure should
    # not pay that again
    cache = out.with_suffix(".json")
    if args.reuse and cache.exists():
        records = json.loads(cache.read_text())
        print(f"reusing {len(records)} cached predictions from {cache}")
    else:
        checkpoint = newest_checkpoint(run_dir, args.which)
        records = collect(checkpoint, args.dataset, args.clips, args.device)
        cache.write_text(json.dumps(records))
    plot_clips(records, out, level=args.level,
               title=f"{args.run} / {args.which} — {args.dataset} dev ({args.level})",
               seconds=args.seconds)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
