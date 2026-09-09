"""Segmentation metrics for the benchmark. See README.md for the full rationale.

Reimplemented rather than imported from `sign_language_segmentation`, so scoring
is independent of any one model: plain segment lists in, no torch.

  * `frame_f1`, `frame_f1_micro`, `global_iou`, `segment_percentage`
    follow Moryossef & Jiang (2023) §4.4.
  * `mf1s` follows Renz et al. (ICASSP 2021) and their released code
    (`eval.py::get_sign_metric`) as closely as possible — segment level, greedy
    one-to-one matching, thresholds 0.40-0.75 step 0.05, strict `>`, and micro
    aggregation over the corpus — deviating only to clamp the intersection and
    use a true union, since theirs yields a negative IoU for disjoint segments.
    Their exclusive-end convention is not a bug and is not changed; ours are
    inclusive, hence the `+ 1`.

Not carried over: upstream `segment_f1` was `(p*r)/(p+r)`, half of an F1;
`segment_percentage` is in the paper but was dropped upstream, and is restored.

A **segment** is `{"start": int, "end": int}` in frames, both bounds inclusive.
BIO labels are `0 = O`, `1 = B`, `2 = I`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import f1_score

Segment = dict
BIO_LABELS = (0, 1, 2)  # O, B, I

# IoU thresholds for mF1S, from Renz et al.: 0.40 to 0.75 in steps of 0.05
MF1S_THRESHOLDS = tuple(round(0.40 + 0.05 * i, 2) for i in range(8))


def segments_to_bio(segments: Sequence[Segment], num_frames: int) -> np.ndarray:
    """Render segments as a frame-level BIO label array.

    The first frame of each segment is B, the rest I, everything else O.
    Segments are clipped to [0, num_frames); later segments overwrite earlier
    ones where they overlap.
    """
    labels = np.zeros(num_frames, dtype=np.int64)
    for segment in segments:
        start = max(0, int(segment["start"]))
        end = min(num_frames - 1, int(segment["end"]))
        if end < start:
            continue
        labels[start:end + 1] = 2  # I
        labels[start] = 1          # B
    return labels


def bio_to_segments(labels: Sequence[int], b: int = 1, i: int = 2) -> list[Segment]:
    """Frame labels to segments, following Moryossef & Jiang (2023) Algorithm 1.

    This is the argmax case of their greedy decoder (`threshold_likeliest`, where
    `b > max(i, o)` is just "the argmax is B"), with `restart_on_b` as their
    released `probs_to_segments` implements it:

      * a segment opens on the first B;
      * a **run** of B frames is one boundary, not one per frame — their
        `did_pass_start` flag suppresses restarts until a non-B frame is seen;
      * once past that, a B closes the open segment at `i - 1` and opens a new
        one **at that same frame**, so two back-to-back segments are both
        recovered;
      * anything that is neither B nor I — O, and 2026's UNK — closes the
        segment and opens nothing.

    Two deviations, both deliberate. The paper's Algorithm 1 sets `start ← None`
    when a B closes a segment, which consumes that B and loses the following
    segment; their released code restarts at `i`, and that is what produced the
    published numbers, so it is what we follow. And their tail case yields
    `len(probs)`, one past the last frame; ours ends at `len - 1`, since our
    segment bounds are inclusive.

    This decodes **predictions** only. Gold keeps `bio_labels_to_segments`,
    which splits at every B and has no `did_pass_start`: in gold, adjacent B
    frames mean adjacent one-frame segments and that is real, whereas in a noisy
    argmax a run of B is one boundary. Decoding gold both ways gives identical
    spans at phrase level on every split, and 90.4% identical at sign level,
    where one-frame segments are common — the counts still match to within 3 of
    8,004, but this decoder makes a `B O` pair one frame longer. That asymmetry
    is deliberate and is 2023's: their gold came from spans directly and was
    never decoded.

    What it replaces is upstream's `likeliest_probs_to_segments`, which ignores B
    entirely and merges any contiguous run of non-O. Paired with a gold decoder
    that does split on B, that caps `%` far below 1 however good the model is.
    Feeding gold labels through each decoder gives the attainable `%`:

    ==================  ========  =====
    dev set             upstream   ours
    ==================  ========  =====
    YouTube phrase         0.578  1.000
    DGS dev phrase         0.164  1.000
    DGS test phrase        0.246  1.000
    DGS test sign          0.962  1.000
    ==================  ========  =====

    Verified against the released v2023 `probs_to_segments` (with
    `threshold_likeliest`) on 20,000 random label sequences: zero mismatches.
    """
    segments: list[Segment] = []
    start = None
    passed_start = False

    for frame, label in enumerate(labels):
        label = int(label)
        if start is None:
            if label == b:
                start = frame
                passed_start = False
        elif not passed_start:
            # The frame right after the opening B never closes anything, not even
            # an O — it only marks that the opening run is over. That is what
            # `did_pass_start` does in their code, and it is why a lone B followed
            # by O still yields a segment rather than nothing.
            if label != b:
                passed_start = True
        elif label == b:
            segments.append({"start": start, "end": frame - 1})
            start = frame                            # restart on this very frame
            passed_start = False
        elif label != i:                             # O or UNK
            segments.append({"start": start, "end": frame - 1})
            start = None
            passed_start = False

    if start is not None:
        segments.append({"start": start, "end": len(labels) - 1})
    return segments


def frame_f1(pred_bio: np.ndarray, gold_bio: np.ndarray,
             labels: Iterable[int] | None = BIO_LABELS) -> float:
    """Macro-averaged per-class F1 over frame-level BIO labels.

    The paper's primary metric, chosen because it is independent of the segment
    decoding algorithm.

    `labels` defaults to the full set (O, B, I), so the value stays comparable
    across clips of different composition. Pass ``labels=None`` for sklearn's
    default — averaging only over classes actually present — which is what the
    2023 evaluation did: a clip with no annotation, predicted empty, scores 1.0
    there instead of 0.33. That choice moves the corpus mean by ~0.12 on the DGS
    test split, so it matters; see benchmark/.
    """
    pred_bio, gold_bio = np.asarray(pred_bio), np.asarray(gold_bio)
    if pred_bio.shape != gold_bio.shape:
        raise ValueError(f"shape mismatch: {pred_bio.shape} vs {gold_bio.shape}")
    if pred_bio.size == 0:
        return 1.0
    return float(f1_score(gold_bio, pred_bio,
                          labels=None if labels is None else list(labels),
                          average="macro", zero_division=0))


def frame_f1_micro(pred_bio: np.ndarray, gold_bio: np.ndarray,
                   labels: Iterable[int] | None = BIO_LABELS) -> float:
    """Micro-averaged F1 over frame-level BIO labels.

    Pools true/false positives across the three classes instead of averaging
    per-class scores, so frequent classes dominate. Since every frame carries
    exactly one label and all three classes are included, this is numerically
    identical to frame accuracy — it is reported next to the macro score to show
    how much of the macro number is carried by the rare B class.
    """
    pred_bio, gold_bio = np.asarray(pred_bio), np.asarray(gold_bio)
    if pred_bio.shape != gold_bio.shape:
        raise ValueError(f"shape mismatch: {pred_bio.shape} vs {gold_bio.shape}")
    if pred_bio.size == 0:
        return 1.0
    return float(f1_score(gold_bio, pred_bio,
                          labels=None if labels is None else list(labels),
                          average="micro", zero_division=0))


def global_iou(segments: Sequence[Segment], segments_gold: Sequence[Segment],
               num_frames: int) -> float:
    """Frame-level IoU pooled over all segments in a clip.

    The paper's definition, which deliberately does no one-to-one matching
    ("we do not perform a one-to-one mapping between the two using techniques
    like DTW"). It measures whether the right frames were covered and is blind
    to how they were grouped — which is why `segment_percentage` is reported
    alongside it.
    """
    pred_mask = np.zeros(num_frames, dtype=bool)
    gold_mask = np.zeros(num_frames, dtype=bool)
    for segment, mask in ((segments, pred_mask), (segments_gold, gold_mask)):
        for s in segment:
            start = max(0, int(s["start"]))
            end = min(num_frames - 1, int(s["end"]))
            if end >= start:
                mask[start:end + 1] = True

    union = np.logical_or(pred_mask, gold_mask).sum()
    if union == 0:
        return 1.0  # nothing predicted, nothing annotated
    return float(np.logical_and(pred_mask, gold_mask).sum() / union)


def segment_percentage(segments: Sequence[Segment],
                       segments_gold: Sequence[Segment]) -> float:
    """#predicted / #gold. Optimal value is 1.

    Guards against a model that scores well on IoU by emitting one enormous
    segment, or by fragmenting one sign into many.

    With no gold segments the ratio is undefined; following the 2023 code this
    returns 1.0 when nothing is predicted either, and otherwise the raw number
    of predicted segments — a count, not a ratio, so the no-signer case still
    registers as a failure rather than a division by zero.
    """
    if len(segments_gold) == 0:
        return 1.0 if len(segments) == 0 else float(len(segments))
    return len(segments) / len(segments_gold)


def iou_matrix(segments: Sequence[Segment],
               segments_gold: Sequence[Segment]) -> np.ndarray:
    """Pairwise IoU, shape (n_gold, n_pred).

    Standard interval IoU over inclusive frame bounds: the intersection is
    clamped at zero and the union is `len_gold + len_pred - intersection`.
    """
    if len(segments_gold) == 0 or len(segments) == 0:
        return np.zeros((len(segments_gold), len(segments)))

    gold = np.array([[s["start"], s["end"]] for s in segments_gold], dtype=float)
    pred = np.array([[s["start"], s["end"]] for s in segments], dtype=float)

    inter_start = np.maximum(gold[:, 0][:, None], pred[:, 0][None, :])
    inter_end = np.minimum(gold[:, 1][:, None], pred[:, 1][None, :])
    intersection = np.maximum(0.0, inter_end - inter_start + 1.0)

    gold_len = (gold[:, 1] - gold[:, 0] + 1.0)[:, None]
    pred_len = (pred[:, 1] - pred[:, 0] + 1.0)[None, :]
    union = gold_len + pred_len - intersection
    return np.where(union > 0, intersection / union, 0.0)


def greedy_match(matrix: np.ndarray) -> list[float]:
    """Greedily pair gold with predicted segments, returning the matched IoUs.

    Repeatedly takes the largest remaining IoU and removes its row and column,
    as Renz et al. do. This is one-to-one but not optimal — it maximises the
    single best pair at each step rather than the total — and is kept for
    comparability with their published numbers.
    """
    remaining = np.array(matrix, dtype=float, copy=True)
    matched: list[float] = []
    while remaining.size and min(remaining.shape) > 0:
        row, column = np.unravel_index(np.argmax(remaining), remaining.shape)
        matched.append(float(remaining[row, column]))
        remaining = np.delete(remaining, row, axis=0)
        remaining = np.delete(remaining, column, axis=1)
    return matched


def segment_counts(segments: Sequence[Segment], segments_gold: Sequence[Segment],
                   thresholds: Iterable[float] = MF1S_THRESHOLDS) -> np.ndarray:
    """Per-threshold (tp, fp, fn) for one clip. Shape (n_thresholds, 3).

    These are the quantities that must be summed across the corpus before
    computing F1 — see `mf1s_from_counts`.
    """
    thresholds = list(thresholds)
    matched = greedy_match(iou_matrix(segments, segments_gold))

    counts = np.zeros((len(thresholds), 3), dtype=np.int64)
    for index, threshold in enumerate(thresholds):
        # strict >, as in the paper ("IoU higher than a given threshold")
        true_positives = sum(1 for iou in matched if iou > threshold)
        counts[index] = (true_positives,
                         len(segments) - true_positives,
                         len(segments_gold) - true_positives)
    return counts


def mf1s_from_counts(counts: np.ndarray) -> float:
    """Mean F1 across thresholds from summed (tp, fp, fn) counts.

    Micro aggregation: pass the counts summed over every clip in the corpus, as
    Renz et al. do. Averaging per-clip F1 instead gives a macro average and a
    different, non-comparable number.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or counts.shape[1] != 3:
        raise ValueError(f"expected (n_thresholds, 3) counts, got {counts.shape}")

    scores = []
    for true_positives, false_positives, false_negatives in counts:
        denominator = 2 * true_positives + false_positives + false_negatives
        # nothing to score at this threshold: no gold, no predictions
        scores.append(1.0 if denominator == 0 else 2 * true_positives / denominator)
    return float(np.mean(scores))


def mf1s(segments: Sequence[Segment], segments_gold: Sequence[Segment],
         thresholds: Iterable[float] = MF1S_THRESHOLDS) -> float:
    """Single-clip mF1S. For a corpus, sum `segment_counts` and use
    `mf1s_from_counts` instead — per-clip values must not be averaged."""
    return mf1s_from_counts(segment_counts(segments, segments_gold, thresholds))


def evaluate_level(segments: Sequence[Segment], segments_gold: Sequence[Segment],
                   num_frames: int,
                   thresholds: Iterable[float] = MF1S_THRESHOLDS) -> dict:
    """All four metrics for one annotation level (sign or phrase) of one clip.

    `mf1s_counts` is included so the caller can aggregate correctly; the scalar
    `mf1s` is the value for this clip alone and is there for inspection, not for
    averaging.
    """
    pred_bio = segments_to_bio(segments, num_frames)
    gold_bio = segments_to_bio(segments_gold, num_frames)
    return {
        "frame_f1": frame_f1(pred_bio, gold_bio),
        "frame_f1_micro": frame_f1_micro(pred_bio, gold_bio),
        "iou": global_iou(segments, segments_gold, num_frames),
        "percentage": segment_percentage(segments, segments_gold),
        "mf1s": mf1s(segments, segments_gold, thresholds),
        "mf1s_counts": segment_counts(segments, segments_gold, thresholds),
    }


def aggregate(per_clip: Sequence[dict]) -> dict:
    """Combine per-clip results into the numbers that go in the table.

    `frame_f1`, `iou` and `percentage` are averaged over clips, as in the 2023
    evaluation. `mf1s` is aggregated **micro**, by summing counts across clips,
    as in Renz et al.
    """
    keys = ("frame_f1", "frame_f1_micro", "iou", "percentage")
    if not per_clip:
        return {**{key: float("nan") for key in keys}, "mf1s": float("nan")}

    totals = np.sum([result["mf1s_counts"] for result in per_clip], axis=0)
    return {
        **{key: float(np.mean([r[key] for r in per_clip])) for key in keys},
        "mf1s": mf1s_from_counts(totals),
    }
