"""Score prediction files and print the benchmark table.

Stage 2 of 2. Reads the JSON written by a `predict_*.py` script and applies
[`../metrics/`](../metrics/). Kept separate from inference so that changing a
metric only means re-scoring, never re-running a model.

Environment: **`sas`** — this needs only numpy and scikit-learn, no torch and no
TensorFlow.

    conda activate sas
    python benchmark/score.py benchmark/predictions/*.json
    python benchmark/score.py benchmark/predictions/*.json --per-clip

By default only clips carrying gold annotation are scored. Pass `--all-clips` for
the looser 17-clip set the 2023 paper used.

A prediction file stores the segments its model was decoded into, and those are
what is scored. `--decode NAME` scores the stored frame labels instead, decoded
again with that decoder of [`../metrics/decoders.py`](../metrics/decoders.py), so
one file serves every decoder. The file is not changed, and the row names the
decoder. Frame F1 is the same either way: it never reads segments.

    python benchmark/score.py benchmark/predictions/*.json --decode i-runs-kept
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from metrics import (aggregate, decoder_names, evaluate_level,  # noqa: E402
                     frame_f1, frame_f1_micro, get_decoder, global_iou,
                     segment_counts,
                     segment_percentage)  # evaluate_level is used by --per-clip

LEVELS = ("sign", "phrase")
COLUMNS = ("frame_f1", "frame_f1_micro", "iou", "percentage", "mf1s")


def unrle(runs) -> np.ndarray:
    """Expand run-length encoded frame labels."""
    return np.concatenate([np.full(count, value, dtype=np.int64) for value, count in runs])


def predicted_segments(clip: dict, level: str, decoder=None) -> list:
    """The predicted segments of a clip at one level.

    Without a decoder, the segments stored in the file. With one, the stored
    frame labels decoded again: what `--decode` asks for.
    """
    if decoder is None:
        return clip["pred"].get(level, [])
    if "pred_bio" not in clip:
        raise SystemExit(
            f"clip {clip.get('id')} has no frame labels to decode; re-run the predict step")
    return decoder.decode(unrle(clip["pred_bio"][level]))


def score_clip(clip: dict, level: str, decoder=None) -> dict:
    """Metrics for one clip at one level, matching how the 2023 eval computed them.

    Frame metrics come from the stored per-frame labels — gold from `build_bio`,
    prediction from the argmax of the probabilities — because that is what the
    2023 code scored. Segment metrics come from the decoded segments against
    `floor`-converted gold. The two use different golds; that asymmetry is the
    original's, reproduced rather than tidied away.

    Prediction files must carry frame labels. Deriving them from the decoded
    segments instead would silently compute a different quantity, so that is an
    error rather than a fallback.

    `decoder` replaces the stored segments by the stored labels decoded again
    (`predicted_segments`). The frame metrics do not move with it.
    """
    pred, gold = predicted_segments(clip, level, decoder), clip["gold"].get(level, [])
    num_frames = clip["num_frames"]

    if "gold_bio" not in clip or "pred_bio" not in clip:
        raise SystemExit(
            f"clip {clip.get('id')} has no frame labels; re-run the predict step")

    gold_bio, pred_bio = unrle(clip["gold_bio"][level]), unrle(clip["pred_bio"][level])
    # labels=None reproduces the 2023 evaluation, which called sklearn without a
    # label set and so averaged only over classes present. A clip with no
    # annotation, predicted empty, scores 1.0 rather than 0.33 — worth 0.12 on
    # the DGS test mean.
    result = {"frame_f1": frame_f1(pred_bio, gold_bio, labels=None),
              "frame_f1_micro": frame_f1_micro(pred_bio, gold_bio, labels=None)}

    result["iou"] = global_iou(pred, gold, num_frames)
    result["percentage"] = segment_percentage(pred, gold)
    result["mf1s_counts"] = segment_counts(pred, gold)
    result["mf1s"] = 0.0  # per-clip value unused; mF1S is aggregated micro
    return result


def score_file(path: Path, annotated_only: bool = True, decoder=None) -> dict:
    """Score one prediction file, per level.

    `decoder` is a decoder of `metrics.decoders`, or None for the segments stored
    in the file. The row says which under `decode`: the name of the decoder, or
    None.

    `annotated_only` (the default) drops clips with no gold at that level. Three
    of the 17 DGS test clips are the unannotated partner in a two-signer
    conversation: no gold, and every model correctly predicts nothing, so each
    scores ~1.0 on every metric and lifts the mean without testing anything.

    Dropping them is what the 2026 upstream eval does at the dataset level. The
    2023 evaluation kept them, so reproducing that paper needs
    `annotated_only=False` — see README.md.
    """
    payload = json.loads(path.read_text())
    clips = payload["clips"]

    results = {}
    for level in LEVELS:
        scored = [clip for clip in clips
                  if clip["gold"].get(level) or not annotated_only]
        # A level no clip annotates stays out of the table entirely, blank
        # rather than zero.
        if not any(clip["gold"].get(level) for clip in clips):
            results[level] = None
            continue
        results[level] = aggregate([score_clip(clip, level, decoder) for clip in scored])

    return {
        "model": payload.get("model", path.stem),
        "decode": decoder.name if decoder is not None else None,
        "dataset": payload.get("dataset", "?"),
        "split": payload.get("split", "?"),
        "thresholds": payload.get("thresholds", {}),
        "clips": len(clips),
        "annotated_only": annotated_only,
        "levels": results,
    }


def model_label(row: dict) -> str:
    """Model name carrying its decoding thresholds, e.g. `2023 E4s (60/50, 80/80)`.

    The thresholds belong to the model rather than to a level group — they differ
    per level (the tuned phrase pair is 90/90 for E1s but 80/80 for E4s) and the
    scores move with them, so a row is not interpretable without them. Written as
    `sign, phrase` in that order; a level the dataset does not annotate is
    omitted.

    When the stored labels were decoded again (`--decode`), the name of that
    decoder follows in brackets, e.g. `05_dgs_w1-5-1_s42 [i-runs-kept]`, so that
    such a row is never read as the stored segments of the file.
    """
    pairs = [f"{p[0]:g}/{p[1]:g}" for level in LEVELS
             if row["levels"].get(level) and (p := row["thresholds"].get(level))]
    label = f"{row['model']} ({', '.join(pairs)})" if pairs else row["model"]
    return f"{label} [{row['decode']}]" if row.get("decode") else label


def format_table(rows: list[dict]) -> str:
    """One row per model, with Sign and Phrase as column groups.

    Laid out like Table 2 of the 2023 paper, which groups the levels side by side
    rather than stacking them, so models are compared at a glance. Both frame F1
    averagings are named in full — `F1-ma` / `F1-mi` — because "acc" hid that the
    micro figure is the same metric under a different averaging.
    """
    metrics = ("frame_f1", "frame_f1_micro", "iou", "percentage", "mf1s")
    short = {"frame_f1": "F1-ma", "frame_f1_micro": "F1-mi", "iou": "IoU",
             "percentage": "%", "mf1s": "mF1S"}

    labels = [model_label(row) for row in rows]
    name_width = max([len("model")] + [len(label) for label in labels])

    group = " ".join(f"{short[m]:>6}" for m in metrics)
    width = len(group)
    head1 = f"{'model':<{name_width}} {'dataset':<18} {'split':<5} │ " \
            f"{'Sign'.center(width)} │ {'Phrase'.center(width)}"
    head2 = f"{'':<{name_width}} {'':<18} {'':<5} │ {group} │ {group}"
    rule = "─" * len(head1)

    lines = [rule, head1, head2, rule]
    for row, label in zip(rows, labels):
        cells = []
        for level in LEVELS:
            values = row["levels"].get(level)
            if values is None:
                # the dataset does not annotate this level: blank, not zero
                cells.append(" ".join(f"{'—':>6}" for _ in metrics))
                continue
            cells.append(" ".join(f"{values[m]:>6.3f}" for m in metrics))
        lines.append(f"{label:<{name_width}} {row['dataset']:<18} "
                     f"{row['split']:<5} │ {cells[0]} │ {cells[1]}")
    lines.append(rule)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--per-clip", action="store_true",
                        help="also print each clip, worst mF1S first")
    parser.add_argument("--all-clips", action="store_true",
                        help="also score clips with no gold at that level. Needed "
                             "to reproduce the 2023 paper, which included them; "
                             "the default drops them (14 of 17 DGS test clips)")
    parser.add_argument("--decode", default=None, choices=decoder_names(aliases=True),
                        help="decode the stored frame labels again with this "
                             "decoder and score those segments, in place of the "
                             "segments stored in the file. Not given: the stored "
                             "segments. '2023' and 'upstream' are the earlier "
                             "names of b-honouring and b-ignoring")
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    args = parser.parse_args()

    decoder = get_decoder(args.decode) if args.decode else None
    rows = [score_file(path, annotated_only=not args.all_clips, decoder=decoder)
            for path in args.predictions]
    print(format_table(rows))

    if args.per_clip:
        for path in args.predictions:
            payload = json.loads(path.read_text())
            print(f"\nper clip — {payload.get('model', path.stem)}"
                  + (f" [{decoder.name}]" if decoder else ""))
            scored = []
            for clip in payload["clips"]:
                if not clip["gold"].get("sign"):
                    continue
                pred = predicted_segments(clip, "sign", decoder)
                result = evaluate_level(pred, clip["gold"]["sign"], clip["num_frames"])
                scored.append((result["mf1s"], clip["id"], result,
                               len(clip["gold"]["sign"]), len(pred)))
            for mf1s_value, clip_id, result, n_gold, n_pred in sorted(scored):
                print(f"  {clip_id:<22} gold {n_gold:>5} pred {n_pred:>5}  "
                      f"F1 {result['frame_f1']:.4f}  IoU {result['iou']:.4f}  "
                      f"%={result['percentage']:.2f}  mF1S {mf1s_value:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
