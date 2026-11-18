"""Run the 2023 segmentation model over a Public DGS Corpus split.

**Drives the original v2023 code**, rather than reimplementing its pipeline.
Reproducing the published numbers turned out to depend on details that are easy
to get wrong from a reading — most importantly that `tfds_dataset.py` imports its
own `pose_utils`, whose `pose_hide_legs` zeroes exactly eight leg points *and*
their confidences. Substituting pose-format's same-named helper shifts the model
input enough to cost ~0.07 frame F1. So the 2023 source is vendored out of git
and called directly; only the outputs are ours.

Verified against Moryossef & Jiang (2023) on the test split, E1s:

    sign    frame F1 0.6378 (0.63)   accuracy 0.7540 (0.75)   IoU 0.6878 (0.69)
    phrase  frame F1 0.6615 (0.65)   accuracy 0.8799 (0.88)   IoU 0.8471 (0.82)

and the split loads as 9 documents / 17 videos, matching the paper.

Data loading is shared with every other model through
`../datasets/public_dgs_corpus/load.py`, which
also does the vendoring described below.

What is vendored, from the git history, cached under `.cache/v2023_src/`:

  * `data.py`, `pose_utils.py`, `probs_to_segments.py` from tag **v2023**
  * `tfds_dataset.py` from **e3a020b**, the commit that vendored it from the
    `_shared` package that v2023's `data.py` imports but does not contain

Three patches are applied, none of which touch behaviour:

  * `_shared.tfds_dataset` / `.pose_utils` imports are made absolute
  * `mediapipe` is stubbed — `tfds_dataset` imports it only to build
    `FACEMESH_CONTOURS_POINTS`, used solely when `reduce_face=True`
  * ELAN and CMDI paths are re-resolved by file name into the download archive,
    since the TFDS records point at a share which we can no longer read

The TFDS folder and the archive are the keys `data.public_dgs_corpus.tfds` and
`data.public_dgs_corpus.downloads` of the paths file (`../paths.py`).

Output is one JSON per run, holding gold and predicted segments plus frame-level
BIO labels, consumed by `score.py`.

    conda activate sas2023
    python benchmark/predict_dgs_2023.py --split test --model model_E4s-1.pth
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths
from datasets.public_dgs_corpus import load as dgs_data  # noqa: E402

CACHE = dgs_data.CACHE

# Components, fps and model settings from the 2023 job scripts (jobs/job_batch.sh):
#   E1s: hidden_dim=256 encoder_depth=4 bidirectional
#   E4s: E1s + --optical_flow=true --hand_normalization=true
COMPONENTS = dgs_data.COMPONENTS
FPS = 25

# Tuned decoding from the 2023 grid search (src/summary_decoding_E4s.csv,
# 82 configurations, selected on dev): sign b=60 o=40/50/60, phrase b=80 o=80/90.
# IoU saturates across many configurations, so percentage is what separates them.
# The shipped v2023 CLI hardcodes phrase 90/90, which is the *E1s* tuning applied
# regardless of checkpoint; for E4s the grid rates it worse, so it is not used here.
DEFAULT_SIGN = (60.0, 50.0)
DEFAULT_PHRASE = (80.0, 80.0)

LEVELS = {"sign": "sign", "sentence": "phrase"}  # their name -> ours


def rle(values) -> list[list[int]]:
    """Run-length encode frame labels, so they fit in the JSON."""
    out: list[list[int]] = []
    for value in np.asarray(values).tolist():
        if out and out[-1][0] == value:
            out[-1][1] += 1
        else:
            out.append([int(value), 1])
    return out


def to_frames(segments_seconds, fps: float, num_frames: int) -> list[dict]:
    """Seconds -> inclusive frame spans, exactly as v2023 `model.py::evaluate` did:

        segments_gold = [{'start': floor(s['start_time'] * fps),
                          'end':   floor(s['end_time']   * fps)} ...]

    and `segment_IoU` fills `segments_v[start:end + 1]`, so both ends are
    inclusive. This is *not* the mapping `build_bio` uses for the frame-level
    labels, which walks to the first frame at or after the start. The 2023
    evaluation genuinely uses two different golds; both are reproduced.
    """
    out = []
    for segment in segments_seconds:
        start = max(0, math.floor(segment["start_time"] * fps))
        end = min(num_frames - 1, math.floor(segment["end_time"] * fps))
        if end >= start:
            out.append({"start": start, "end": end})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--model", default="model_E4s-1.pth",
                        choices=["model_E1s-1.pth", "model_E4s-1.pth", "model.pth"])
    parser.add_argument("--sign-b", type=float, default=DEFAULT_SIGN[0])
    parser.add_argument("--sign-o", type=float, default=DEFAULT_SIGN[1])
    parser.add_argument("--phrase-b", type=float, default=DEFAULT_PHRASE[0])
    parser.add_argument("--phrase-o", type=float, default=DEFAULT_PHRASE[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    v2023 = dgs_data.vendored()

    import torch
    from utils.probs_to_segments import probs_to_segments  # noqa: E402

    is_e4 = "E4" in args.model  # E4 variants add optical flow + 3D hand normalisation
    tag = args.model.replace("model_", "").replace(".pth", "")
    out_path = args.out or (Path(__file__).parent / "predictions" /
                            f"dgs_{args.split}_2023_{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"model        {args.model} (optical flow + hand norm: {is_e4})")
    print(f"sign b/o     {args.sign_b:g} / {args.sign_o:g}")
    print(f"phrase b/o   {args.phrase_b:g} / {args.phrase_o:g}\n")

    started = time.time()
    dataset = v2023.get_dataset(name="dgs_corpus", poses="holistic", fps=FPS,
                                split=args.split, components=COMPONENTS,
                                data_dir=str(paths.data("public_dgs_corpus.tfds")),
                                hand_normalization=is_e4, optical_flow=is_e4,
                                classes="bio")

    model = torch.jit.load(str(CACHE / "v2023" / args.model))
    model.eval()

    thresholds = {"sign": (args.sign_b, args.sign_o),
                  "sentence": (args.phrase_b, args.phrase_o)}
    clips = []

    for item in dataset:
        with torch.no_grad():
            probs = model(item["pose"]["data"].unsqueeze(0))

        num_frames = len(item["bio"]["sign"])
        record = {"id": item["id"], "num_frames": num_frames, "fps": FPS,
                  "gold": {}, "pred": {}, "gold_bio": {}, "pred_bio": {}}

        for their_name, our_name in LEVELS.items():
            level_probs = probs[their_name][0]
            b, o = thresholds[their_name]
            record["gold"][our_name] = to_frames(item["segments"][their_name], FPS, num_frames)
            record["pred"][our_name] = [
                {"start": max(0, int(s["start"])), "end": min(num_frames - 1, int(s["end"]))}
                for s in probs_to_segments(level_probs, b, o)
                if min(num_frames - 1, int(s["end"])) >= max(0, int(s["start"]))
            ]
            # frame metrics use build_bio's labels against the argmax, never the
            # decoded segments — as v2023 model.py::evaluate does
            record["gold_bio"][our_name] = rle(item["bio"][their_name].numpy())
            record["pred_bio"][our_name] = rle(level_probs.numpy().argmax(axis=1))

        clips.append(record)
        print(f"  {record['id']:<24} {num_frames:>7} frames  "
              f"sign {len(record['pred']['sign']):>5}/{len(record['gold']['sign']):<5} "
              f"phrase {len(record['pred']['phrase']):>4}/{len(record['gold']['phrase']):<4} "
              f"({time.time() - started:.0f}s)", flush=True)

    out_path.write_text(json.dumps({
        "dataset": "public_dgs_corpus", "split": args.split,
        "model": f"2023 {tag}", "checkpoint": args.model,
        "thresholds": {"sign": [args.sign_b, args.sign_o],
                       "phrase": [args.phrase_b, args.phrase_o]},
        "pipeline": "v2023 original (vendored)",
        "clips": clips,
    }, indent=2))

    print(f"\n{len(clips)} clips in {time.time() - started:.0f}s")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
