"""Run the 2026 segmentation model over a Public DGS Corpus split.

Model: https://github.com/sign-language-processing/segmentation/blob/main/dist/2026/README.md

Stage 1 of 2, writing the same prediction JSON `score.py` already reads, so the
2026 model lands in the same table as the 2023 one with no new scoring code.

Data comes from [`../datasets/public_dgs_corpus/load.py`](../datasets/public_dgs_corpus/load.py) — the same clips, filters and gold
annotations the 2023 run uses. Everything below that is the 2026 model's own:

  * **Preprocessing** mirrors `sign_language_segmentation.datasets.common
    .load_and_augment` with `split != TRAIN`: `preprocess_pose` (body and hands
    only, legs hidden, holistic reduced, mean/std normalised), then velocity
    features appended, giving the 50 joints x 6 dims the checkpoint expects. No
    augmentation — fps_aug, frame dropout and body-part dropout are training-only.
  * **BIO labels** follow the same branch upstream takes at eval: the shipped
    config sets `fps_aug: true`, and `load_and_augment` reads that flag to pick
    `create_bio_from_times` (searchsorted over frame timestamps) over `create_bio`
    (floor/ceil on a fixed fps). The flag does no augmenting outside training, but
    it does select the label builder, so it is honoured here.
  * **Decoding** is `likeliest_probs_to_segments` — plain argmax. The 2026 README
    is explicit that threshold decoding was tried and rejected, so unlike the 2023
    row this one has no `b/o` to report.
  * **Chunking** is upstream's by default: the whole clip goes to
    `PoseTaggingModel`, whose CNN runs full length and whose transformer then
    cuts it into *non-overlapping* `num_frames` chunks. Attention cannot cross a
    cut, so a segment spanning one is seen as two halves.

    `--overlap 0.5` instead slides a `num_frames` window and keeps each frame's
    prediction from the window where it sits most centrally. Measured on the
    2026 baseline checkpoint it changes nothing (sign IoU 0.597 -> 0.599, phrase
    0.824 -> 0.825) for 1.6x the runtime, so it is off by default. The likely
    reason: upstream's CNN already runs full length, so features at a chunk edge
    carry neighbouring context, while sliding windows the CNN too — fixing the
    attention cut and adding a convolution cut.

Two conventions are translated on the way out:

  * BIO ids. Upstream 2026 is `UNK=0, O=1, B=2, I=3`; ours (and 2023's) is
    `O=0, B=1, I=2`. Frame labels are remapped, and `UNK` cannot occur since we
    never pad a batch of one.
  * Gold frames. Upstream derives gold segments from the BIO labels, which is
    kept here — it is that model's protocol. The 2023 row floors both bounds
    instead. See `README.md`; the two differ by at most a frame.
  * Gold phrases. Upstream takes a phrase to be the annotated sentence; v2023
    takes it to be the extent of that sentence's glosses. **We score against the
    gloss extent** (`--phrase glosses`, the default) so the benchmark's phrase
    column means one thing across every row. That is not this model's own target,
    and it costs it — see README.md. `--phrase sentence` scores it on its own
    terms instead.

**Source.** Defaults to `--source native`: the archived `.pose` downloads at their
original **50fps**, which is both what this model publishes numbers at and the
format its own loader reads. `--source tfds` serves the same clips from the TFDS
build instead, but that build baked in a 25fps downsample, so it costs the model
real accuracy. The clip list and gold annotations are identical either way.

    conda activate sas
    python benchmark/predict_dgs_2026.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.public_dgs_corpus import load as dgs_data  # noqa: E402

# upstream 2026 ids -> ours. UNK has no counterpart; it only appears as batch
# padding, which cannot happen at batch size 1.
BIO_2026_TO_OURS = {1: 0, 2: 1, 3: 2}

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


def prepare(pose, fps: float, velocity: bool = True):
    """Apply the 2026 preprocessing to a raw holistic pose.

    Mirrors `load_and_augment` outside training: preprocess, take xyz, and append
    fps-normalised velocity when the model was trained with it. `velocity` is read
    off the checkpoint rather than assumed — a model trained without it takes 3
    dims per joint, and feeding it 6 fails outright.
    """
    from sign_language_segmentation.utils.pose import compute_velocity, preprocess_pose

    pose = preprocess_pose(pose)
    pose_data = pose.body.data.filled(0)[:, 0, :, :3].astype(np.float32)
    frame_times = np.arange(len(pose_data), dtype=np.float32) / fps
    if velocity:
        pose_data = np.concatenate(
            [pose_data, compute_velocity(pose_data, frame_times)], axis=-1)
    return pose_data, frame_times


def overlapping_encode(overlap: float):
    """An `encode` that overlaps the *transformer* chunks, CNN untouched.

    Upstream runs the CNN over the whole clip and only then cuts the sequence
    into non-overlapping `num_frames` chunks for the transformer, so attention
    never crosses a cut. This keeps the CNN exactly as it is — full length, so
    its features still carry context across every boundary — and slides the
    transformer window instead, giving each frame its prediction from the chunk
    where it sits most centrally.

    It also ends the last chunk at the final frame instead of zero-padding it.
    Upstream pads, and since it removed the attention mask, real frames in that
    chunk attend to up to 1023 zeros. That is a bug, not a choice, so it is always
    fixed here — worth phrase % 0.751 -> 0.771 on the 2026 baseline, more than
    overlap itself buys.

    That is the difference from windowing the whole model from outside, which
    also cuts the CNN and measured no benefit.
    """
    def encode(self, pose_data, timestamps=None):
        import torch

        x = self.input_norm(self.frame_cnn(pose_data))  # full length, unchanged
        batch, total, _ = x.shape

        if timestamps is None:
            ts = (torch.arange(total, device=x.device, dtype=torch.float32)
                  / self.REFERENCE_FPS).unsqueeze(0).expand(batch, -1)
        else:
            ts = timestamps.to(x.device)
            if ts.dim() == 1:
                ts = ts.unsqueeze(0).expand(batch, -1)

        chunk = self.hparams.num_frames
        if total <= chunk:
            for layer in self.encoder_attn:
                x = layer(x, ts)
            return x

        stride = max(1, int(round(chunk * (1.0 - overlap))))
        starts = list(range(0, max(1, total - chunk + 1), stride))
        if starts[-1] + chunk < total:
            starts.append(total - chunk)

        # chunks are uniform, so run them all as one batch, as upstream does
        segments = torch.stack([x[0, s:s + chunk] for s in starts])
        seg_ts = torch.stack([ts[0, s:s + chunk] for s in starts])
        for layer in self.encoder_attn:
            segments = layer(segments, seg_ts)

        # claim strictly left to right: each chunk owns from where the previous
        # stopped to `margin` short of its own right edge, so a frame is never
        # taken from beside a cut, and the tail chunk cannot overwrite frames it
        # sees badly
        out = torch.zeros_like(x)
        margin, prev_hi = (chunk - stride) // 2, 0
        for i, s in enumerate(starts):
            hi = total if i == len(starts) - 1 else s + chunk - margin
            lo, hi = max(prev_hi, s), max(hi, prev_hi)
            if hi > lo:
                out[0, lo:hi] = segments[i, lo - s:hi - s]
            prev_hi = hi
        return out

    return encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--model", default=None,
                        help="path to a model dir, .safetensors or .ckpt "
                             "(default: the weights shipped in dist/2026)")
    parser.add_argument("--source", default="native", choices=["native", "tfds"],
                        help="native = archived 50fps .pose files (default); "
                             "tfds = the 25fps TFDS build")
    parser.add_argument("--fps", type=int, default=dgs_data.AVAILABLE_FPS,
                        help="only used by --source tfds")
    parser.add_argument("--phrase", default="glosses", choices=["glosses", "sentence"],
                        help="what counts as a phrase: 'glosses' = the extent of a "
                             "sentence's glosses (the 2023 definition, used for the "
                             "benchmark so the column is consistent); 'sentence' = "
                             "the annotated sentence bounds, this model's own target")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="overlap the transformer chunks by this fraction "
                             "(0-1), leaving the CNN full length. Default 0: no "
                             "overlap, measured to buy nothing. The last chunk "
                             "always ends at the final frame rather than being "
                             "zero-padded — that is a fix, not an option")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tfds-root", default=dgs_data.TFDS_ROOT)
    parser.add_argument("--backup", default=dgs_data.BACKUP)
    parser.add_argument("--label", default="2026",
                        help="model name for the results table; set it when "
                             "scoring a retrained checkpoint so the row is not "
                             "confused with the shipped one")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch
    from sign_language_segmentation.bin import _load_model_uncached, resolve_model_path
    from sign_language_segmentation.metrics import (bio_labels_to_segments,
                                                    likeliest_probs_to_segments)
    from sign_language_segmentation.utils.bio import create_bio_from_times

    model_path = args.model or resolve_model_path()
    model = _load_model_uncached(model_dir=model_path, device=args.device)
    num_frames = getattr(model.hparams, "num_frames", None)
    # (joints, dims); dims is 6 with velocity appended, 3 without
    pose_dims = tuple(getattr(model.hparams, "pose_dims", (50, 6)))
    velocity = pose_dims[1] >= 6

    out_path = args.out or (Path(__file__).parent / "predictions" /
                            f"dgs_{args.split}_2026.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    native = args.source == "native"
    print(f"model        {model_path}")
    print(f"chunk size   {num_frames} frames")
    print(f"decoding     likeliest (argmax)")
    print(f"source       {args.source} "
          f"({'50fps originals' if native else f'{args.fps}fps TFDS build'})")
    print(f"phrase gold  {args.phrase}")
    print(f"pose dims    {pose_dims}  (velocity {'on' if velocity else 'off'})")
    print(f"chunking     {num_frames}-frame transformer chunks, "
          f"{args.overlap:.0%} overlap, tail chunk unpadded\n")
    import types
    model.encode = types.MethodType(overlapping_encode(args.overlap), model)

    started = time.time()
    clips = []

    clip_source = (dgs_data.iter_clips_native(split=args.split, backup=args.backup)
                   if native else
                   dgs_data.iter_clips(split=args.split, fps=args.fps,
                                       tfds_root=args.tfds_root, backup=args.backup))

    for clip in clip_source:
        pose_data, frame_times = prepare(clip["pose"], clip["fps"], velocity=velocity)
        total_frames = len(pose_data)
        frame_times_ms = frame_times * 1000

        with torch.no_grad():
            log_probs = model(
                torch.from_numpy(pose_data).unsqueeze(0).to(args.device),
                timestamps=torch.from_numpy(frame_times).unsqueeze(0).to(args.device))

        record = {"id": clip["id"], "num_frames": total_frames, "fps": clip["fps"],
                  "gold": {}, "pred": {}, "gold_bio": {}, "pred_bio": {}}

        spans = dgs_data.sign_phrase_spans(clip["sentences"], phrase=args.phrase)

        for their_name, our_name in LEVELS.items():
            spans_ms = [{"start": s["start_time"] * 1000, "end": s["end_time"] * 1000}
                        for s in spans[our_name]]
            gold_bio = create_bio_from_times(spans_ms, frame_times_ms)
            probs = log_probs[their_name][0].cpu()

            record["gold"][our_name] = bio_labels_to_segments(torch.from_numpy(gold_bio.astype(np.int64)))
            record["pred"][our_name] = likeliest_probs_to_segments(probs)
            record["gold_bio"][our_name] = rle([BIO_2026_TO_OURS.get(int(v), 0) for v in gold_bio])
            record["pred_bio"][our_name] = rle(
                [BIO_2026_TO_OURS.get(int(v), 0) for v in probs.numpy().argmax(axis=1)])

        clips.append(record)
        print(f"  {record['id']:<24} {total_frames:>7} frames  "
              f"sign {len(record['pred']['sign']):>5}/{len(record['gold']['sign']):<5} "
              f"phrase {len(record['pred']['phrase']):>4}/{len(record['gold']['phrase']):<4} "
              f"({time.time() - started:.0f}s)", flush=True)

    out_path.write_text(json.dumps({
        "dataset": "public_dgs_corpus", "split": args.split,
        "model": args.label, "checkpoint": str(model_path),
        # argmax decoding has no thresholds; score.py renders the absence as "-"
        "thresholds": {},
        "source": args.source,
        "overlap": args.overlap,
        "velocity": velocity,
        "pose_dims": list(pose_dims),
        "phrase_gold": args.phrase,
        "pipeline": "sign_language_segmentation main (dist/2026)",
        "clips": clips,
    }, indent=2))

    print(f"\n{len(clips)} clips in {time.time() - started:.0f}s")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
