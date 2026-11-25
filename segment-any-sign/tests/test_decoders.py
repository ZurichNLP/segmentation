"""The decoders: from the label of each frame to segments.

Run from `segment-any-sign/`:
    pytest tests/test_decoders.py

A sequence is written as a string of `O`, `B` and `I`, one letter for each frame,
and a segment as `(first frame, last frame)`, both inside the segment.
"""

import random

import pytest
from metrics import bio_to_segments

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
