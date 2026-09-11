"""Score a checkpoint on the YouTube-SL-25 dev set, filtered or raw.

The DGS side has had `benchmark/predict_dgs_2026.py` plus `score.py` since the
start; YouTube has only ever been scored inside the training loop, which means a
finished checkpoint could not be re-scored without retraining. This closes that
gap, using the same [`../metrics/`](../metrics/) code and the same decoder as
everything else, so a row here is comparable to a row from either.

    python experiments/eval_youtube.py --run 02_pretrain_youtube_b80-2026.09.10
    python experiments/eval_youtube.py --run <run> --raw      # unfiltered dev

`--raw` scores the 167-video set every run before 2026-09-11 validated on.
Without it, the 82 videos whose subtitle timings survive `DEV_FILTER`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DIST = Path("/scratch/zifjia/segment-any-sign/dist")


def score(checkpoint: Path, dev_filter: bool, device: str, limit: int = None,
          level: str = "sentence") -> dict:
    """Run the model over each dev clip and pool the four metrics.

    Per-clip means for frame F1, IoU and `%`; mF1S from counts pooled over the
    corpus, never averaged per clip — that is how `score.py` does it and mixing
    the two conventions is how two numbers for one model start to disagree.
    """
    import torch
    from pose_format import Pose

    from sign_language_segmentation.metrics import bio_labels_to_segments
    from sign_language_segmentation.utils.bio import BIO, create_bio_from_times
    from sign_language_segmentation.utils.pose import (compute_velocity,
                                                       preprocess_pose)

    from benchmark.predict_dgs_2026 import load_checkpoint
    from datasets.youtube_sl25 import load as yt
    from metrics import (bio_to_segments, frame_f1, frame_f1_micro, global_iou,
                         mf1s_from_counts, segment_counts, segment_percentage)

    model = load_checkpoint(str(checkpoint), device=device)
    velocity = tuple(getattr(model.hparams, "pose_dims", (50, 6)))[1] >= 6

    specs = list(yt.clip_specs("dev", dev_filter=dev_filter))[:limit]
    pooled = {"frame_f1": [], "frame_f1_micro": [], "iou": [], "percentage": []}
    counts = None
    skipped = 0

    for done, spec in enumerate(specs, 1):
        try:
            with open(spec["pose_path"], "rb", buffering=4 * 1024 * 1024) as handle:
                pose = preprocess_pose(Pose.read(handle.read()))
            data = pose.body.data.filled(0)[:, 0, :, :3].astype(np.float32)
            times = np.arange(len(data), dtype=np.float32) / spec["fps"]
            if velocity:
                data = np.concatenate([data, compute_velocity(data, times)], axis=-1)
            with torch.no_grad():
                log_probs = model(
                    torch.from_numpy(data).unsqueeze(0).to(device),
                    timestamps=torch.from_numpy(times).unsqueeze(0).to(device))
        except Exception as error:
            print(f"  skipping {spec['id']}: {type(error).__name__}: {error}")
            skipped += 1
            continue

        gold = create_bio_from_times(
            [{"start": s["start_time"] * 1000, "end": s["end_time"] * 1000}
             for s in spec["sentences"]], times * 1000)
        gold_segments = bio_labels_to_segments(
            torch.from_numpy(gold.astype(np.int64)))
        if not gold_segments:
            skipped += 1
            continue

        predicted = log_probs[level][0].cpu().float().numpy().argmax(axis=1)
        pred_segments = bio_to_segments(predicted, b=BIO["B"], i=BIO["I"])
        # upstream ids to ours, the convention every metric here expects
        remap = {BIO["O"]: 0, BIO["B"]: 1, BIO["I"]: 2}
        gold_bio = np.array([remap.get(int(v), 0) for v in gold])
        pred_bio = np.array([remap.get(int(v), 0) for v in predicted])

        pooled["frame_f1"].append(frame_f1(pred_bio, gold_bio, labels=None))
        pooled["frame_f1_micro"].append(
            frame_f1_micro(pred_bio, gold_bio, labels=None))
        pooled["iou"].append(global_iou(pred_segments, gold_segments, len(data)))
        pooled["percentage"].append(
            segment_percentage(pred_segments, gold_segments))
        batch = segment_counts(pred_segments, gold_segments)
        counts = batch if counts is None else counts + batch
        if done % 20 == 0:
            print(f"  {done}/{len(specs)}", flush=True)

    result = {key: float(np.mean(values)) for key, values in pooled.items()}
    result["mf1s"] = float(mf1s_from_counts(counts)) if counts is not None else 0.0
    result["clips"] = len(pooled["frame_f1"])
    result["skipped"] = skipped
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run directory under dist/")
    parser.add_argument("--which", default="best", choices=["best", "last"])
    parser.add_argument("--raw", action="store_true",
                        help="score the unfiltered 167-video dev set")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from experiments.plot_segmentation import newest_checkpoint

    checkpoint = newest_checkpoint(DIST / args.run, args.which)
    print(f"{args.run} / {args.which}   dev set: "
          f"{'raw' if args.raw else 'filtered'}")
    result = score(checkpoint, dev_filter=not args.raw, device=args.device,
                   limit=args.limit)
    print(f"\n{result['clips']} clips scored"
          + (f", {result['skipped']} skipped" if result["skipped"] else ""))
    print("  F1-ma {frame_f1:.3f}   F1-mi {frame_f1_micro:.3f}   "
          "IoU {iou:.3f}   % {percentage:.3f}   mF1S {mf1s:.3f}".format(**result))


if __name__ == "__main__":
    main()
