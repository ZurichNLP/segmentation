"""Unit tests for the segmentation metrics.

Run from `segment-any-sign/`:
    pytest tests/
"""

import numpy as np
import pytest

from metrics.segmentation import (
    MF1S_THRESHOLDS,
    aggregate,
    evaluate_level,
    frame_f1,
    frame_f1_micro,
    global_iou,
    greedy_match,
    iou_matrix,
    mf1s,
    mf1s_from_counts,
    segment_counts,
    segment_percentage,
    segments_to_bio,
)


def seg(start, end):
    return {"start": start, "end": end}


# --- segments_to_bio ---------------------------------------------------------

def test_bio_marks_begin_then_inside():
    labels = segments_to_bio([seg(1, 3)], num_frames=6)
    # O B I I O O
    assert labels.tolist() == [0, 1, 2, 2, 0, 0]


def test_bio_single_frame_segment_is_just_b():
    assert segments_to_bio([seg(2, 2)], num_frames=4).tolist() == [0, 0, 1, 0]


def test_bio_clips_to_clip_length():
    # a segment running past the end is truncated, not an error
    assert segments_to_bio([seg(2, 99)], num_frames=4).tolist() == [0, 0, 1, 2]


def test_bio_empty_segments_is_all_o():
    assert segments_to_bio([], num_frames=3).tolist() == [0, 0, 0]


def test_bio_adjacent_segments_each_get_their_own_b():
    labels = segments_to_bio([seg(0, 1), seg(2, 3)], num_frames=4)
    assert labels.tolist() == [1, 2, 1, 2]


# --- frame_f1 ----------------------------------------------------------------

def test_frame_f1_perfect():
    gold = segments_to_bio([seg(1, 3)], 6)
    assert frame_f1(gold, gold) == pytest.approx(1.0)


def test_frame_f1_all_wrong_is_zero():
    gold = np.array([1, 2, 2, 0])
    pred = np.array([0, 0, 0, 1])
    assert frame_f1(pred, gold) == pytest.approx(0.0)


def test_frame_f1_averages_over_all_three_classes():
    # B is perfect, I is perfect, O is absent from both -> its F1 counts as 0,
    # so the macro average over three labels is 2/3 rather than 1.0
    gold = np.array([1, 2, 2])
    assert frame_f1(gold, gold) == pytest.approx(2 / 3)


def test_frame_f1_micro_equals_accuracy():
    # micro F1 over single-label frames is exactly accuracy
    gold = np.array([0, 1, 2, 2, 0])
    pred = np.array([0, 1, 2, 0, 0])
    assert frame_f1_micro(pred, gold) == pytest.approx(4 / 5)


def test_frame_f1_micro_hides_what_macro_exposes():
    # 100 O frames, one B, one I; the model misses both non-O frames.
    # micro stays high because O dominates; macro collapses.
    gold = np.array([0] * 100 + [1, 2])
    pred = np.array([0] * 102)
    assert frame_f1_micro(pred, gold) > 0.98
    assert frame_f1(pred, gold) < 0.35


def test_frame_f1_rejects_length_mismatch():
    with pytest.raises(ValueError):
        frame_f1(np.array([0, 1]), np.array([0, 1, 2]))


# --- global_iou --------------------------------------------------------------

def test_iou_identical_is_one():
    assert global_iou([seg(1, 3)], [seg(1, 3)], 10) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert global_iou([seg(0, 2)], [seg(5, 7)], 10) == pytest.approx(0.0)


def test_iou_half_overlap():
    # pred frames 0..3, gold frames 2..5 -> intersection 2, union 6
    assert global_iou([seg(0, 3)], [seg(2, 5)], 10) == pytest.approx(2 / 6)


def test_iou_pools_across_segments_ignoring_grouping():
    # one big prediction vs two gold segments covering the same frames:
    # IoU is blind to the grouping, which is why percentage is reported too
    fragmented = global_iou([seg(0, 1), seg(2, 3)], [seg(0, 3)], 8)
    merged = global_iou([seg(0, 3)], [seg(0, 3)], 8)
    assert fragmented == pytest.approx(merged) == pytest.approx(1.0)


def test_iou_empty_both_is_one():
    assert global_iou([], [], 10) == pytest.approx(1.0)


def test_iou_prediction_against_empty_gold_is_zero():
    assert global_iou([seg(0, 2)], [], 10) == pytest.approx(0.0)


# --- segment_percentage ------------------------------------------------------

def test_percentage_exact_match():
    assert segment_percentage([seg(0, 1), seg(3, 4)], [seg(0, 1), seg(3, 4)]) == 1.0


def test_percentage_over_segmentation():
    assert segment_percentage([seg(0, 0), seg(1, 1), seg(2, 2)], [seg(0, 2)]) == 3.0


def test_percentage_under_segmentation():
    assert segment_percentage([seg(0, 2)], [seg(0, 0), seg(1, 1)]) == 0.5


def test_percentage_catches_what_iou_misses():
    # the merged prediction scores a perfect IoU but a revealing percentage
    pred, gold = [seg(0, 3)], [seg(0, 1), seg(2, 3)]
    assert global_iou(pred, gold, 8) == pytest.approx(1.0)
    assert segment_percentage(pred, gold) == 0.5


def test_percentage_empty_gold():
    assert segment_percentage([], []) == 1.0
    # a count, not a ratio — the no-signer failure case
    assert segment_percentage([seg(0, 1), seg(3, 4)], []) == 2.0


# --- mf1s --------------------------------------------------------------------

def test_mf1s_perfect_is_one():
    segments = [seg(0, 9), seg(20, 29)]
    assert mf1s(segments, segments) == pytest.approx(1.0)


def test_mf1s_disjoint_is_zero():
    assert mf1s([seg(0, 4)], [seg(50, 54)]) == pytest.approx(0.0)


def test_mf1s_empty_both_is_one():
    assert mf1s([], []) == pytest.approx(1.0)


def test_mf1s_one_side_empty_is_zero():
    assert mf1s([], [seg(0, 4)]) == pytest.approx(0.0)
    assert mf1s([seg(0, 4)], []) == pytest.approx(0.0)


def test_mf1s_punishes_grouping_errors_unlike_iou():
    # perfect frame coverage, wrong grouping: IoU is 1.0, mF1S is not
    pred, gold = [seg(0, 3)], [seg(0, 1), seg(2, 3)]
    assert global_iou(pred, gold, 8) == pytest.approx(1.0)
    assert mf1s(pred, gold) < 0.5


def test_mf1s_threshold_sensitivity():
    # IoU of exactly 0.5 -> counted below 0.5, missed at or above it
    pred, gold = [seg(0, 9)], [seg(0, 4)]
    assert mf1s(pred, gold, thresholds=[0.4]) == pytest.approx(1.0)
    assert mf1s(pred, gold, thresholds=[0.75]) == pytest.approx(0.0)


def test_mf1s_matching_is_one_to_one():
    # two predictions both overlapping a single gold segment: only one can match
    pred = [seg(0, 4), seg(0, 4)]
    gold = [seg(0, 4)]
    # 1 TP, 1 FP, 0 FN -> F1 = 2/(2+1+0)
    assert mf1s(pred, gold, thresholds=[0.5]) == pytest.approx(2 / 3)


def test_mf1s_default_thresholds_are_the_published_range():
    assert MF1S_THRESHOLDS[0] == 0.40
    assert MF1S_THRESHOLDS[-1] == 0.75
    assert len(MF1S_THRESHOLDS) == 8


def test_mf1s_inclusive_bounds():
    # [0, 4] and [0, 9] share 5 frames of a 10-frame union -> IoU 0.5 exactly
    assert iou_matrix([seg(0, 9)], [seg(0, 4)])[0, 0] == pytest.approx(0.5)
    # comparison is strict, following the paper's "higher than"
    assert mf1s([seg(0, 9)], [seg(0, 4)], thresholds=[0.49]) == pytest.approx(1.0)
    assert mf1s([seg(0, 9)], [seg(0, 4)], thresholds=[0.5]) == pytest.approx(0.0)


# --- evaluate_level ----------------------------------------------------------

def test_evaluate_level_reports_all_four():
    segments = [seg(1, 3), seg(5, 7)]
    result = evaluate_level(segments, segments, num_frames=10)
    assert set(result) == {"frame_f1", "frame_f1_micro", "iou", "percentage",
                           "mf1s", "mf1s_counts"}
    for key in ("frame_f1", "frame_f1_micro", "iou", "percentage", "mf1s"):
        assert result[key] == pytest.approx(1.0)
    # counts are carried so the caller can aggregate micro
    assert result["mf1s_counts"].shape == (len(MF1S_THRESHOLDS), 3)


def test_evaluate_level_on_a_miss():
    result = evaluate_level([seg(0, 1)], [seg(8, 9)], num_frames=10)
    assert result["iou"] == pytest.approx(0.0)
    assert result["mf1s"] == pytest.approx(0.0)
    assert result["percentage"] == 1.0  # right count, wrong place


# --- iou_matrix and greedy matching -----------------------------------------

def test_iou_matrix_shape_is_gold_by_pred():
    m = iou_matrix([seg(0, 1), seg(5, 6), seg(9, 9)], [seg(0, 1)])
    assert m.shape == (1, 3)


def test_iou_matrix_clamps_disjoint_to_zero():
    # Renz's formula would give a negative value here; ours must not
    m = iou_matrix([seg(10, 12)], [seg(0, 2)])
    assert m[0, 0] == pytest.approx(0.0)
    assert (m >= 0).all()


def test_iou_matrix_uses_true_union_not_enclosing_span():
    # two 3-frame segments sharing 1 frame: intersection 1, union 5
    m = iou_matrix([seg(2, 4)], [seg(0, 2)])
    assert m[0, 0] == pytest.approx(1 / 5)


def test_greedy_match_takes_global_max_first():
    matrix = np.array([[0.9, 0.5],
                       [0.6, 0.4]])
    # 0.9 wins first, removing its row and column, leaving only 0.4
    assert greedy_match(matrix) == [pytest.approx(0.9), pytest.approx(0.4)]


def test_greedy_can_be_worse_than_optimal():
    # greedy takes 0.9 then is stuck with 0.1 (total 1.0);
    # the optimal assignment would take 0.8 + 0.85 = 1.65
    matrix = np.array([[0.9, 0.8],
                       [0.85, 0.1]])
    assert sum(greedy_match(matrix)) == pytest.approx(1.0)


def test_greedy_match_empty():
    assert greedy_match(np.zeros((0, 3))) == []
    assert greedy_match(np.zeros((2, 0))) == []


def _greedy_match_reference(matrix):
    """`greedy_match` as it was first written: the specification of its result.

    It took the largest IoU left with `np.argmax` and deleted its row and its
    column, which copies the matrix at every step and took seconds for a clip
    with 2,000 gold signs. Kept verbatim, and not to be changed: `greedy_match`
    must return exactly this list, for every input, ties and the zeros at the
    end included.
    """
    remaining = np.array(matrix, dtype=float, copy=True)
    matched: list[float] = []
    while remaining.size and min(remaining.shape) > 0:
        row, column = np.unravel_index(np.argmax(remaining), remaining.shape)
        matched.append(float(remaining[row, column]))
        remaining = np.delete(remaining, row, axis=0)
        remaining = np.delete(remaining, column, axis=1)
    return matched


def _same_list(result, expected):
    """The same list, bit for bit: Python floats with the same `repr`.

    `repr` gives back every float exactly, tells -0.0 from 0.0 and writes every
    NaN as `nan`, so it is exact where `==` is not: `==` takes -0.0 for 0.0, and
    never takes a NaN for a NaN.
    """
    return (all(type(value) is float for value in result)
            and [repr(value) for value in result] == [repr(value) for value in expected])


def _random_shape(rng, index):
    """The eight kinds of shape in turn: square, tall, wide, one row, one
    column, empty in either direction, and empty in both."""
    n, k = int(rng.integers(1, 13)), int(rng.integers(1, 13))
    return [(n, n), (n + k, n), (n, n + k), (1, n), (n, 1),
            (0, n), (n, 0), (0, 0)][index % 8]


def _random_segments(rng, count):
    starts = rng.integers(0, 60, size=count)
    return [seg(int(start), int(start + rng.integers(0, 8))) for start in starts]


#: seeded random matrices of each kind, for the comparison with the reference
RANDOM_MATRICES = {
    # distinct values, so no ties
    "continuous": lambda rng, shape: rng.random(shape),
    # four values only, so a tie decides most steps
    "many ties": lambda rng, shape: rng.integers(0, 4, size=shape) / 4,
    # mostly zero, with whole rows and columns of zeros
    "zero rows and columns": lambda rng, shape: (
        rng.integers(0, 3, size=shape) / 2 * (rng.random(shape) < 0.3)
        * (rng.random((shape[0], 1)) < 0.7) * (rng.random((1, shape[1])) < 0.7)),
    "all zeros": lambda rng, shape: np.zeros(shape),
    # what greedy_match is given: IoUs of segments, equal fractions tie exactly
    "iou of segments": lambda rng, shape: iou_matrix(
        _random_segments(rng, shape[1]), _random_segments(rng, shape[0])),
    # No IoU is below zero, NaN or infinite, but the list must be the same for
    # those too.
    "negative": lambda rng, shape: rng.integers(-3, 3, size=shape) / 2,
    "signed zeros and inf": lambda rng, shape: np.where(
        rng.random(shape) < 0.3, -0.0,
        np.where(rng.random(shape) < 0.05, np.inf, rng.integers(0, 3, size=shape) / 2)),
    "nan and inf": lambda rng, shape: np.where(
        rng.random(shape) < 0.1, np.nan,
        np.where(rng.random(shape) < 0.1, rng.choice([np.inf, -np.inf], size=shape),
                 rng.integers(-1, 3, size=shape) / 2)),
}


@pytest.mark.parametrize("kind", list(RANDOM_MATRICES))
def test_greedy_match_equals_the_reference_on_random_matrices(kind):
    rng = np.random.default_rng(0)
    for index in range(1000):
        matrix = RANDOM_MATRICES[kind](rng, _random_shape(rng, index))
        result, expected = greedy_match(matrix), _greedy_match_reference(matrix)
        assert _same_list(result, expected), \
            f"{kind} matrix {index}:\n{matrix!r}\ngives {result}, not {expected}"


#: inputs that the random matrices do not reach, for the comparison with the reference
FIXED_INPUTS = {
    # empty but not 2-D: the reference returns [] for every empty input
    "np.array([])": np.array([]),
    "[]": [],
    "np.zeros((0, 2, 2))": np.zeros((0, 2, 2)),
    # larger than the random shapes, with a tie at most steps
    "many ties, 100 x 100": RANDOM_MATRICES["many ties"](np.random.default_rng(0), (100, 100)),
}


@pytest.mark.parametrize("name", list(FIXED_INPUTS))
def test_greedy_match_equals_the_reference_on_these_inputs(name):
    matrix = FIXED_INPUTS[name]
    result, expected = greedy_match(matrix), _greedy_match_reference(matrix)
    assert _same_list(result, expected), f"{name} gives {result}, not {expected}"


@pytest.mark.parametrize("matrix", [[0.5], [0.5, -1.0], np.ones((1, 1, 1))],
                         ids=["1-D", "1-D with a negative entry", "3-D"])
def test_greedy_match_raises_on_input_not_2d_and_not_empty(matrix):
    # as the reference does; np.atleast_2d would make a 1-D input succeed
    with pytest.raises(ValueError):
        _greedy_match_reference(matrix)
    with pytest.raises(ValueError):
        greedy_match(matrix)


def test_greedy_match_breaks_ties_by_row_then_column():
    # Three 1s tie. (0, 0) comes first, as np.argmax has it, and blocks the
    # other two, which leaves 0.5. Taking (0, 1) first would leave (1, 0) free
    # and give [1.0, 1.0].
    assert greedy_match(np.array([[1.0, 1.0], [1.0, 0.5]])) == [1.0, 0.5]


def test_greedy_match_ends_with_the_zeros_left():
    # one value for each gold or each predicted segment, whichever are fewer
    matrix = np.zeros((3, 5))
    matrix[1, 2] = 0.4
    assert greedy_match(matrix) == [0.4, 0.0, 0.0]
    assert greedy_match(np.zeros((3, 5))) == [0.0, 0.0, 0.0]
    assert greedy_match(np.zeros((5, 3))) == [0.0, 0.0, 0.0]
    # the entries themselves, not a 0.0 written in their place
    assert repr(greedy_match(np.array([[-0.0, 0.0]]))) == "[-0.0]"


def test_greedy_match_one_row_and_one_column():
    assert greedy_match(np.array([[0.2, 0.7, 0.7]])) == [0.7]
    assert greedy_match(np.array([[0.2], [0.7], [0.7]])) == [0.7]


def test_greedy_match_negative_and_nan_entries():
    # No IoU is either, but the list is still the reference's: a negative value
    # is taken when nothing larger is left, and a NaN first, since np.argmax
    # takes it for the largest value.
    assert greedy_match(np.array([[-1.0]])) == [-1.0]
    assert repr(greedy_match(np.array([[np.nan, 1.0], [2.0, 3.0]]))) == "[nan, 3.0]"


# --- counts and micro aggregation -------------------------------------------

def test_segment_counts_shape_and_values():
    counts = segment_counts([seg(0, 9)], [seg(0, 9)], thresholds=[0.5, 0.9])
    assert counts.shape == (2, 3)
    assert counts[0].tolist() == [1, 0, 0]  # perfect match at both thresholds


def test_segment_counts_unmatched_are_fp_and_fn():
    counts = segment_counts([seg(0, 4), seg(50, 54)], [seg(0, 4)], thresholds=[0.5])
    # one true positive, one spurious prediction, no missed gold
    assert counts[0].tolist() == [1, 1, 0]


def test_micro_and_macro_differ():
    """Clips of unequal size are exactly where the two aggregations diverge.

    Averaging per-clip F1 lets a tiny clip count as much as a large one; micro
    weights by the number of segments, which is what Renz et al. report.
    """
    # one segment, perfect
    small = evaluate_level([seg(0, 9)], [seg(0, 9)], 200)
    # ten segments, all wrong
    pred = [seg(i * 10, i * 10 + 4) for i in range(10)]
    gold = [seg(i * 10 + 100, i * 10 + 104) for i in range(10)]
    large = evaluate_level(pred, gold, 400)

    assert small["mf1s"] == pytest.approx(1.0)
    assert large["mf1s"] == pytest.approx(0.0)

    macro = float(np.mean([small["mf1s"], large["mf1s"]]))
    micro = aggregate([small, large])["mf1s"]

    assert macro == pytest.approx(0.5)
    # micro: tp=1, fp=10, fn=10 -> 2/(2+10+10)
    assert micro == pytest.approx(2 / 22)
    assert micro < macro


def test_mf1s_from_counts_rejects_bad_shape():
    with pytest.raises(ValueError):
        mf1s_from_counts(np.zeros((3,)))


def test_aggregate_reports_all_metrics():
    results = [evaluate_level([seg(1, 3)], [seg(1, 3)], 10) for _ in range(4)]
    out = aggregate(results)
    assert out["mf1s"] == pytest.approx(1.0)
    assert set(out) == {"frame_f1", "frame_f1_micro", "iou", "percentage", "mf1s"}


def test_aggregate_empty_is_nan_not_crash():
    out = aggregate([])
    assert np.isnan(out["mf1s"])
    assert np.isnan(out["frame_f1_micro"])
