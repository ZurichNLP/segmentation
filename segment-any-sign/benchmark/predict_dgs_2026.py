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
  * **Decoding** is argmax, as the 2026 README requires — it is explicit that
    threshold decoding was tried and rejected, so unlike the 2023 row this one
    has no `b/o` to report. How the argmax labels become segments is the
    decoder, one of [`../metrics/decoders.py`](../metrics/decoders.py):
    `b-honouring` opens a segment only on a B (`metrics.bio_to_segments`: a B
    closes the open segment and opens the next; it equals the 2023 code at
    argmax only in a clip that has a B, since that code decodes a clip without
    B from I alone);
    `b-ignoring` never inspects B and merges any run of non-O (the rule of
    upstream's `likeliest_probs_to_segments`, which the shipped model is decoded
    with), so two touching segments can only be counted as one while the gold
    decoder splits them; `i-runs-kept` cuts those runs at a B. `2023` and
    `upstream` are the earlier names of the first two and are still accepted.

    Which decoder is used follows one order, the same in every script: the
    value of `--decode` in this run, then the decoder kept with the model, then
    `b-honouring`. A checkpoint of ours keeps the decoder of its training run
    as the hyperparameter `decode` (`experiments/validation_metrics.py`); one
    written before that keeps none. The shipped model keeps none either, since
    its `config.json` is upstream's: without `--model` it is decoded with
    `b-ignoring`, the decoder it ships with, unless `--decode` says otherwise.
    Which decoder a row is *reported* with is a rule of README.md, and the
    person who runs this applies it with `--decode`: no code does.
  * **Chunking** is upstream's: the whole clip goes to `PoseTaggingModel`,
    whose CNN runs full length and whose transformer then cuts it into
    *non-overlapping* `num_frames` chunks, the last one padded. Attention cannot
    cross a cut, so a segment spanning one is seen as two halves.

    Every model is loaded through `enable_masking`
    ([`../experiments/masked_model.py`](../experiments/masked_model.py)), so that
    padded last chunk is masked rather than attended to — the one change to the
    model as published, and a fix rather than a choice. An earlier version ended
    the last chunk at the final frame instead and could overlap chunks; overlap
    measured to buy nothing, and with the mask the padded chunk is exact, so both
    are gone.

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
from metrics import choose_decoder, decoder_names  # noqa: E402

# upstream 2026 ids -> ours. UNK has no counterpart; it only appears as batch
# padding, which cannot happen at batch size 1.
BIO_2026_TO_OURS = {1: 0, 2: 1, 3: 2}

LEVELS = {"sign": "sign", "sentence": "phrase"}  # their name -> ours


#: Class weights that `ValidationMetricsModel` used to register as *persistent*
#: buffers, so they sit in every checkpoint written before 2026-09-11 under keys
#: plain `PoseTaggingModel` has never heard of. Training-only either way: the
#: weight belongs to the loss, not to the trained model.
TRAINING_ONLY_KEYS = ("sign_loss_fn.weight", "phrase_loss_fn.weight")


#: The decoder of the shipped 2026 model: the rule of upstream's
#: `likeliest_probs_to_segments`. Its `config.json` is upstream's file and keeps
#: no decoder, so it is said here.
SHIPPED_DECODE = "b-ignoring"


def model_file(model_path: str | Path) -> Path:
    """The file upstream's loader reads for `model_path`.

    The same choice as `sign_language_segmentation.bin`: `model.safetensors`,
    given or in the folder given; else the `.ckpt` given; else the `best.ckpt`
    of the folder given, a run folder.
    """
    path = Path(model_path)
    if path.suffix == ".safetensors" or (path / "model.safetensors").exists():
        return path if path.suffix == ".safetensors" else path / "model.safetensors"
    return path if path.suffix == ".ckpt" else path / "best.ckpt"


def stored_decode(model_path: str | Path) -> str | None:
    """The name of the decoder kept in the configuration of a model, or None.

    A checkpoint keeps its configuration under `hyper_parameters`, and a
    safetensors model in `config.json` beside `model.safetensors`. Our training
    adds the key `decode` to the first (`experiments/validation_metrics.py`),
    and upstream's export copies it into the second. The file is read, not the
    loaded model: upstream's class takes the key and does not keep it.

    None when the configuration has no such key: every checkpoint written
    before the decoder was kept, and the shipped model.
    """
    import torch

    file = model_file(model_path)
    if file.suffix == ".safetensors":
        config = file.parent / "config.json"
        return json.loads(config.read_text()).get("decode") if config.exists() else None
    state = torch.load(file, map_location="cpu", weights_only=False)
    return state.get("hyper_parameters", {}).get("decode")


def decoder_of_run(given: str | None, model_given: bool, stored: str | None):
    """The decoder of a prediction run, in the order every script follows.

    `given` is `--decode`, `stored` the name kept with the model
    (`stored_decode`), and then comes the default (`metrics.choose_decoder`).
    The shipped model, which is what runs when no `--model` is given, keeps no
    decoder in its configuration and stands for `SHIPPED_DECODE`.
    """
    if stored is None and not model_given:
        stored = SHIPPED_DECODE
    return choose_decoder(given, stored)


def load_checkpoint(model_path: str, device: str = "cpu"):
    """Load a model for evaluation, tolerating those legacy keys.

    Anything that is not a `.ckpt` — the shipped safetensors directory — goes
    straight to upstream's loader. For a checkpoint of ours the two keys above
    are dropped and everything else is loaded **strictly**: a half-loaded model
    produces plausible nonsense rather than an error, so only the keys we can
    name are forgiven.

    Every model comes back with padding masked (`enable_masking`), whichever
    branch loaded it, so no evaluation path can score an unmasked model.

    It also comes back with `stored_decode`, the name of the decoder kept in its
    configuration, or None when it keeps none (`stored_decode` above). That is
    the second place a script looks for its decoder, after its own `--decode`.
    """
    import torch

    from sign_language_segmentation.bin import _load_model_uncached
    from sign_language_segmentation.model.model import PoseTaggingModel

    from experiments.masked_model import enable_masking

    path = Path(model_path)
    if path.suffix != ".ckpt":
        model = enable_masking(_load_model_uncached(model_dir=str(path), device=device))
        model.stored_decode = stored_decode(path)
        return model

    state = torch.load(path, map_location=device, weights_only=False)
    # read here, from the file: the loaded model does not keep the key
    stored = state.get("hyper_parameters", {}).get("decode")
    dropped = [key for key in TRAINING_ONLY_KEYS if key in state["state_dict"]]
    for key in dropped:
        state["state_dict"].pop(key)
    if not dropped:
        model = enable_masking(_load_model_uncached(model_dir=str(path), device=device))
        model.stored_decode = stored
        return model

    # load_from_checkpoint only takes a path, so the trimmed state goes via a
    # sibling file that is removed whether or not the load succeeds
    patched = path.parent / f".{path.stem}.load.ckpt"
    torch.save(state, patched)
    try:
        model = PoseTaggingModel.load_from_checkpoint(checkpoint_path=str(patched),
                                                      map_location=device)
    finally:
        patched.unlink(missing_ok=True)
    print(f"dropped training-only keys: {', '.join(dropped)}")
    model = enable_masking(model.to(device).eval())
    model.stored_decode = stored
    return model


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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--label", default="2026",
                        help="model name for the results table; set it when "
                             "scoring a retrained checkpoint so the row is not "
                             "confused with the shipped one")
    parser.add_argument("--decode", default=None, choices=decoder_names(aliases=True),
                        help="how the argmax labels become segments: b-honouring "
                             "opens a segment only on a B, b-ignoring makes one "
                             "segment of any run of labels that are not O, "
                             "i-runs-kept cuts those runs at a B. Not given: the "
                             "decoder kept with the model (a checkpoint of ours "
                             "keeps the one of its training run; the shipped "
                             "model stands for b-ignoring), else b-honouring. "
                             "'2023' and 'upstream' are the earlier names of "
                             "b-honouring and b-ignoring")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch
    from sign_language_segmentation.bin import resolve_model_path
    from sign_language_segmentation.metrics import bio_labels_to_segments
    from sign_language_segmentation.utils.bio import create_bio_from_times

    model_path = args.model or resolve_model_path()
    model = load_checkpoint(model_path, device=args.device)
    # One order in every script: --decode of this run, then the decoder kept
    # with the model, then the default. The shipped model is benchmarked on its
    # own terms, decoder included: that is the model as published.
    decoder = decoder_of_run(args.decode, model_given=args.model is not None,
                             stored=model.stored_decode)
    decoder_from = ("--decode" if args.decode
                    else "kept with the model" if model.stored_decode
                    else "the decoder of the shipped model" if args.model is None
                    else "the default: none given, none kept with the model")
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
    print(f"decoding     argmax, segments via {decoder.name}  ({decoder_from})")
    print(f"source       {args.source} "
          f"({'50fps originals' if native else f'{args.fps}fps TFDS build'})")
    print(f"phrase gold  {args.phrase}")
    print(f"pose dims    {pose_dims}  (velocity {'on' if velocity else 'off'})")
    print(f"chunking     {num_frames}-frame transformer chunks, "
          f"last one padded and masked\n")

    started = time.time()
    clips = []

    clip_source = (dgs_data.iter_clips_native(split=args.split)
                   if native else
                   dgs_data.iter_clips(split=args.split, fps=args.fps))

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

            # the argmax labels in our ids: what is stored, and what is decoded
            pred_bio = [BIO_2026_TO_OURS.get(int(v), 0) for v in probs.numpy().argmax(axis=1)]

            record["gold"][our_name] = bio_labels_to_segments(torch.from_numpy(gold_bio.astype(np.int64)))
            record["pred"][our_name] = decoder.decode(pred_bio)
            record["gold_bio"][our_name] = rle([BIO_2026_TO_OURS.get(int(v), 0) for v in gold_bio])
            record["pred_bio"][our_name] = rle(pred_bio)

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
        "chunking": "padded last chunk, masked",
        # the decoder of the stored segments, by its name in metrics/decoders.py
        "decode": decoder.name,
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
