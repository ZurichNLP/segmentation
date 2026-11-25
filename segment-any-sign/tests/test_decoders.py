"""The decoders: from the label of each frame to segments.

Run from `segment-any-sign/`:
    pytest tests/test_decoders.py

A sequence is written as a string of `O`, `B` and `I`, one letter for each frame,
and a segment as `(first frame, last frame)`, both inside the segment.
"""

import json
import random
from pathlib import Path

import numpy as np
import pytest
from metrics import bio_to_segments
from metrics.decoders import (
    DECODERS,
    DEFAULT,
    BHonouring,
    BIgnoring,
    IRunsKept,
    choose_decoder,
    decoder_names,
    get_decoder,
)

from tests.test_locations import needs

ROOT = Path(__file__).resolve().parents[1]   # segment-any-sign/

LETTERS = {"O": 0, "B": 1, "I": 2}


def labels(sequence: str) -> list[int]:
    return [LETTERS[letter] for letter in sequence]


def spans(segments) -> list[tuple[int, int]]:
    return [(segment["start"], segment["end"]) for segment in segments]


def random_sequences(count: int, values, seed: int = 0, longest: int = 40):
    """`count` sequences of 0 to `longest` labels drawn from `values`, the same on every run."""
    generator = random.Random(seed)
    return [[generator.choice(values) for _ in range(generator.randint(0, longest))]
            for _ in range(count)]


# --- bio_to_segments, the B-honouring rule -----------------------------------

# A segment opens only on a B. The frame after the opening B never closes the
# segment, as in the 2023 code, so `OBOO` is 1-2 and `OBOIIO` is one segment.
HONOURING = {
    "": [],
    "O": [],
    "B": [(0, 0)],
    "I": [],
    "OOO": [],
    "OIIO": [],                       # a run of I without a B: no segment
    "IIOI": [],
    "OBIIO": [(1, 3)],
    "OBII": [(1, 3)],                 # open at the end of the clip
    "OBOO": [(1, 2)],                 # the O after the opening B is inside
    "OBOIIO": [(1, 4)],               # and so is the run of I behind that O
    "OBOBOO": [(1, 2), (3, 4)],
    "BOBO": [(0, 1), (2, 3)],
    "OBIBIO": [(1, 2), (3, 4)],       # a B closes the open segment and opens the next
    "OBIIB": [(1, 3), (4, 4)],
    "OBBIO": [(1, 3)],                # a run of B is one boundary
    "BBB": [(0, 2)],
    "OIIBIO": [(3, 4)],               # the I before the first B are lost
    "IBIO": [(1, 2)],
    "OBIOIIO": [(1, 2)],              # and the run of I after an O
    "OBIOIBI": [(1, 2), (5, 6)],
}


@pytest.mark.parametrize("sequence", HONOURING)
def test_bio_to_segments_on_short_sequences(sequence):
    assert spans(bio_to_segments(labels(sequence))) == HONOURING[sequence]


def test_bio_to_segments_gives_no_segment_without_b():
    # here it differs from the released 2023 function, which decodes a clip
    # without B from I alone: see the docstring of bio_to_segments
    for sequence in random_sequences(500, values=(0, 2)):
        assert bio_to_segments(sequence) == []


def test_bio_to_segments_reads_upstream_ids_like_remapped_labels():
    # upstream: UNK=0, O=1, B=2, I=3. UNK closes a segment as O does, so
    # decoding the upstream ids equals decoding the labels remapped to ours
    to_ours = {1: 0, 2: 1, 3: 2}
    for sequence in random_sequences(2000, values=(0, 1, 2, 3)):
        remapped = [to_ours.get(label, 0) for label in sequence]
        assert bio_to_segments(sequence, b=2, i=3) == bio_to_segments(remapped)


# --- the three decoders on the same short sequences ---------------------------

# Any run of labels that are not O is one segment. B is not read.
IGNORING = {
    "": [],
    "O": [],
    "B": [(0, 0)],
    "I": [(0, 0)],
    "OOO": [],
    "OIIO": [(1, 2)],
    "IIOI": [(0, 1), (3, 3)],
    "OBIIO": [(1, 3)],
    "OBII": [(1, 3)],
    "OBOO": [(1, 1)],
    "OBOIIO": [(1, 1), (3, 4)],
    "OBOBOO": [(1, 1), (3, 3)],
    "BOBO": [(0, 0), (2, 2)],
    "OBIBIO": [(1, 4)],               # two segments that touch are one
    "OBIIB": [(1, 4)],
    "OBBIO": [(1, 3)],
    "BBB": [(0, 2)],
    "OIIBIO": [(1, 4)],
    "IBIO": [(0, 2)],
    "OBIOIIO": [(1, 2), (4, 5)],
    "OBIOIBI": [(1, 2), (4, 6)],
}

# The same runs as the B-ignoring decoder, cut where a B follows a label that
# is not B. So a run of I without a B is a segment, and an O closes at once.
I_RUNS_KEPT = {
    "": [],
    "O": [],
    "B": [(0, 0)],
    "I": [(0, 0)],
    "OOO": [],
    "OIIO": [(1, 2)],
    "IIOI": [(0, 1), (3, 3)],
    "OBIIO": [(1, 3)],
    "OBII": [(1, 3)],
    "OBOO": [(1, 1)],                 # not 1-2: the O is not in the segment
    "OBOIIO": [(1, 1), (3, 4)],
    "OBOBOO": [(1, 1), (3, 3)],
    "BOBO": [(0, 0), (2, 2)],
    "OBIBIO": [(1, 2), (3, 4)],       # a B closes the open segment and opens the next
    "OBIIB": [(1, 3), (4, 4)],
    "OBBIO": [(1, 3)],                # a run of B is one boundary
    "BBB": [(0, 2)],
    "OIIBIO": [(1, 2), (3, 4)],       # the I before the first B are kept
    "IBIO": [(0, 0), (1, 2)],
    "OBIOIIO": [(1, 2), (4, 5)],      # and the run of I after an O
    "OBIOIBI": [(1, 2), (4, 4), (5, 6)],
}

EXPECTED = {"b-honouring": HONOURING, "b-ignoring": IGNORING, "i-runs-kept": I_RUNS_KEPT}


def test_every_decoder_has_its_table_and_the_tables_hold_the_same_sequences():
    assert set(EXPECTED) == set(DECODERS)
    assert set(IGNORING) == set(HONOURING) == set(I_RUNS_KEPT)


@pytest.mark.parametrize("name, sequence", [(name, sequence) for name in EXPECTED
                                            for sequence in EXPECTED[name]])
def test_decoder_on_short_sequences(name, sequence):
    assert spans(DECODERS[name].decode(labels(sequence))) == EXPECTED[name][sequence]


def keep_i_runs(labels):
    """The definition of "I runs kept", as it was given for this decoder."""
    segments, start, prev = [], None, 0
    for frame, label in enumerate(labels):
        if label == 0:
            if start is not None:
                segments.append({"start": start, "end": frame - 1}); start = None
        elif label == 1 and prev != 1:
            if start is not None:
                segments.append({"start": start, "end": frame - 1})
            start = frame
        elif start is None:
            start = frame
        prev = label
    if start is not None:
        segments.append({"start": start, "end": len(labels) - 1})
    return segments


def test_i_runs_kept_is_its_definition():
    for sequence in random_sequences(2000, values=(0, 1, 2)):
        assert IRunsKept().decode(sequence) == keep_i_runs(sequence)


def test_i_runs_kept_equals_b_ignoring_on_any_sequence_without_b():
    for sequence in random_sequences(2000, values=(0, 2)):
        assert IRunsKept().decode(sequence) == BIgnoring().decode(sequence)


@pytest.mark.parametrize("name", ["b-ignoring", "i-runs-kept"])
def test_segments_cover_the_frames_that_are_not_o_and_no_other(name):
    # so these two decoders give the same IoU, and differ only in where they cut
    for sequence in random_sequences(2000, values=(0, 1, 2)):
        segments = DECODERS[name].decode(sequence)
        covered = [frame for segment in segments
                   for frame in range(segment["start"], segment["end"] + 1)]
        assert covered == [frame for frame, label in enumerate(sequence) if label != 0]


def test_b_honouring_is_bio_to_segments():
    for sequence in random_sequences(2000, values=(0, 1, 2)):
        assert BHonouring().decode(sequence) == bio_to_segments(sequence)


@needs("sign_language_segmentation", "torch")
def test_b_ignoring_is_the_function_of_the_upstream_package():
    """`BIgnoring` is written on labels, so that scoring needs no torch.

    Upstream's function reads log-probabilities in upstream's ids. Each label
    becomes a one-hot row, so its argmax is that label. If upstream changes its
    function, this test fails, and the two must be brought together again.
    """
    import torch

    from sign_language_segmentation.metrics import likeliest_probs_to_segments

    to_ours = {1: 0, 2: 1, 3: 2}      # upstream: UNK=0, O=1, B=2, I=3
    hand_made = [[], [0], [3], [2], [1, 3, 3, 1], [1, 2, 3, 2, 3, 1], [3, 0, 3], [2, 0, 2, 1]]
    for sequence in hand_made + random_sequences(1000, values=(0, 1, 2, 3)):
        log_probs = torch.full((len(sequence), 4), -20.0)
        log_probs[torch.arange(len(sequence)), torch.tensor(sequence, dtype=torch.long)] = 0.0
        remapped = [to_ours.get(label, 0) for label in sequence]
        assert BIgnoring().decode(remapped) == likeliest_probs_to_segments(log_probs)


# --- what a decoder takes and gives -------------------------------------------

@pytest.mark.parametrize("name", list(EXPECTED))
def test_decoder_takes_an_array_and_gives_plain_integers(name):
    sequence = labels("OBIOIBI")
    from_array = DECODERS[name].decode(np.array(sequence, dtype=np.int64))
    assert from_array == DECODERS[name].decode(sequence)
    # segments are written into prediction files: numpy integers would not be
    assert json.loads(json.dumps(from_array)) == from_array


@pytest.mark.parametrize("name", list(EXPECTED))
@pytest.mark.parametrize("sequence", [[0, 1, 3], [0, -1, 2], [1, 2, 4, 0]])
def test_a_label_outside_o_b_i_is_an_error(name, sequence):
    # upstream ids that were not remapped must not give a number
    with pytest.raises(ValueError, match="O=0, B=1, I=2"):
        DECODERS[name].decode(sequence)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_labels_must_be_one_row(name):
    with pytest.raises(ValueError, match="one label for each frame"):
        DECODERS[name].decode(np.zeros((4, 3), dtype=np.int64))


# --- names ----------------------------------------------------------------------

def test_names_and_the_earlier_names():
    assert decoder_names() == ["b-honouring", "b-ignoring", "i-runs-kept"]
    assert decoder_names(aliases=True) == ["b-honouring", "b-ignoring", "i-runs-kept",
                                           "2023", "upstream"]
    for name in decoder_names():
        assert get_decoder(name).name == name
    # the values of --decode before the decoders had names, kept in committed files
    assert isinstance(get_decoder("2023"), BHonouring)
    assert isinstance(get_decoder("upstream"), BIgnoring)
    assert get_decoder("2023").name == "b-honouring"
    assert get_decoder("upstream").name == "b-ignoring"


def test_an_unknown_name_is_an_error_that_lists_the_names():
    with pytest.raises(ValueError) as error:
        get_decoder("likeliest")
    for name in decoder_names(aliases=True):
        assert name in str(error.value)
    assert "likeliest" in str(error.value)


# --- the order: --decode of the run, the decoder stored with the model, the default

def test_default_decoder_is_b_honouring():
    assert DEFAULT == "b-honouring"
    assert choose_decoder(None).name == "b-honouring"
    assert choose_decoder(None, None).name == "b-honouring"


def test_stored_decoder_comes_before_the_default():
    assert choose_decoder(None, "b-ignoring").name == "b-ignoring"
    assert choose_decoder(None, "upstream").name == "b-ignoring"


def test_given_decoder_comes_before_the_stored_one():
    assert choose_decoder("i-runs-kept", "b-ignoring").name == "i-runs-kept"
    assert choose_decoder("b-honouring", "b-ignoring").name == "b-honouring"
    assert choose_decoder("2023", "b-ignoring").name == "b-honouring"


def test_a_stored_name_that_is_not_a_decoder_is_an_error():
    with pytest.raises(ValueError):
        choose_decoder(None, "likeliest")


# --- the committed prediction files ---------------------------------------------

# Every file that is there, with no list of them here: a new file is tested too.
PREDICTIONS = sorted((ROOT / "experiments" / "predictions").glob("*.json"))


def test_there_are_committed_prediction_files():
    assert PREDICTIONS, "no file in experiments/predictions/: the next test would test nothing"


@pytest.mark.parametrize("path", PREDICTIONS, ids=lambda path: path.stem)
def test_committed_prediction_file_is_reproduced_by_the_decoder_that_made_it(path):
    """The stored segments of a file are its stored labels, decoded by the decoder it names.

    `decode` is that name. A file without it was written before 2026-09-09, when
    upstream's function was the only decoder, so it stands for `upstream`.
    """
    from benchmark.score import unrle

    payload = json.loads(path.read_text())
    decoder = get_decoder(payload.get("decode", "upstream"))
    for clip in payload["clips"]:
        for level, stored in clip["pred"].items():
            decoded = decoder.decode(unrle(clip["pred_bio"][level]))
            assert decoded == stored, f"{path.name}, clip {clip['id']}, {level}, {decoder.name}"
