"""The decoders: from the label of each frame to segments.

A decoder reads the label of each frame, after argmax, and returns segments. IoU,
`%` and mF1S are computed on those segments, so they depend on the decoder. Frame
F1 does not. Every script that decodes takes its decoder from here, by name.

  * `b-honouring` (`BHonouring`): a segment opens only on a B. A run of I
    without a B gives no segment. This is `bio_to_segments`.
  * `b-ignoring` (`BIgnoring`): any run of labels that are not O is one segment.
    B is not read. This is the rule of upstream's `likeliest_probs_to_segments`.
  * `i-runs-kept` (`IRunsKept`): the runs of the B-ignoring decoder, cut at a B.
    A B opens a segment and closes the one that is open, where a run of B is
    one boundary; an O closes it; and a run of I with no B before it is a
    segment too.

Labels are ours: `0 = O`, `1 = B`, `2 = I`. A caller that holds upstream's ids
(`UNK = 0`, `O = 1`, `B = 2`, `I = 3`) remaps them first, UNK to O. A label
outside 0 to 2 is an error, so that a caller which forgets to remap fails
instead of giving a number. A segment is `{"start": int, "end": int}` in frames,
both bounds inclusive, as everywhere in `metrics`.

Which decoder a script uses is always decided in the same order
(`choose_decoder`): the one named in that run, then the one stored with the
model, then `DEFAULT`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from metrics.segmentation import Segment, bio_to_segments

O, B, I = 0, 1, 2

#: the decoder of a script when none is named and none is stored with the model
DEFAULT = "b-honouring"


class Decoder:
    """One rule that turns the labels of the frames of a clip into segments."""

    #: what `--decode` takes, and what is stored with a prediction and a model
    name: str = ""
    #: earlier names of the same rule, still accepted and still read
    aliases: tuple[str, ...] = ()

    def decode(self, labels: Sequence[int]) -> list[Segment]:
        """The segments of `labels`, in the order of their first frame."""
        return self._segments(_checked(labels))

    def _segments(self, labels: list[int]) -> list[Segment]:
        raise NotImplementedError


class BHonouring(Decoder):
    """A segment opens only on a B: `bio_to_segments`, whose docstring has the detail.

    A run of I without a B before it gives no segment, and the frame after an
    opening B is inside the segment whatever its label. It was called `2023`,
    after the code it follows; it equals that code only in a clip that has a B.
    """

    name = "b-honouring"
    aliases = ("2023",)

    def _segments(self, labels: list[int]) -> list[Segment]:
        return bio_to_segments(labels, b=B, i=I)


class BIgnoring(Decoder):
    """Any run of labels that are not O is one segment. B is not read.

    The rule of `likeliest_probs_to_segments` of the upstream package, which the
    shipped 2026 model is decoded with: two segments that touch are one. It is
    written here on labels, because upstream's function takes log-probabilities
    in a torch tensor and scoring must not need torch.
    `tests/test_decoders.py` keeps the two equal.
    """

    name = "b-ignoring"
    aliases = ("upstream",)

    def _segments(self, labels: list[int]) -> list[Segment]:
        segments: list[Segment] = []
        start = None
        for frame, label in enumerate(labels):
            if label != O:
                if start is None:
                    start = frame
            elif start is not None:
                segments.append({"start": start, "end": frame - 1})
                start = None
        if start is not None:
            segments.append({"start": start, "end": len(labels) - 1})
        return segments


class IRunsKept(Decoder):
    """A B opens a segment, and a run of I without a B is a segment too.

      * a B opens a segment, and closes the one that is open;
      * a run of B is one boundary, not one for each frame;
      * an O closes the open segment, at once: unlike `BHonouring`, the frame
        after an opening B is tested like any other;
      * an I with no segment open opens one, so a run of I that has no B before
        it is kept.

    Every segment holds only frames labelled B or I, and every such frame is in
    a segment. So this decoder covers the same frames as `BIgnoring` and gives
    the same IoU; the two differ in where the segments are cut, and are equal on
    a sequence without B.
    """

    name = "i-runs-kept"

    def _segments(self, labels: list[int]) -> list[Segment]:
        segments: list[Segment] = []
        start = None
        previous = O
        for frame, label in enumerate(labels):
            if label == O:
                if start is not None:
                    segments.append({"start": start, "end": frame - 1})
                    start = None
            elif label == B and previous != B:
                if start is not None:
                    segments.append({"start": start, "end": frame - 1})
                start = frame
            elif start is None:
                start = frame
            previous = label
        if start is not None:
            segments.append({"start": start, "end": len(labels) - 1})
        return segments


def _checked(labels: Sequence[int]) -> list[int]:
    """`labels` as a list of plain integers, after checking that they are ours."""
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(f"a decoder takes one label for each frame of one clip, "
                         f"got an array of shape {array.shape}")
    if array.size and not np.isin(array, (O, B, I)).all():
        found = sorted({int(value) for value in np.unique(array)})
        raise ValueError(f"a decoder takes the labels O=0, B=1, I=2, got {found}: "
                         f"remap upstream's ids (UNK=0, O=1, B=2, I=3) first")
    return [int(value) for value in array.tolist()]


#: every decoder, by its name, in the order they are listed
DECODERS: dict[str, Decoder] = {
    decoder.name: decoder for decoder in (BHonouring(), BIgnoring(), IRunsKept())}

_ALIASES: dict[str, Decoder] = {
    alias: decoder for decoder in DECODERS.values() for alias in decoder.aliases}


def decoder_names(aliases: bool = False) -> list[str]:
    """The names of the decoders, and with `aliases` their earlier names after them."""
    return list(DECODERS) + (list(_ALIASES) if aliases else [])


def get_decoder(name: str) -> Decoder:
    """The decoder of that name, or of that earlier name."""
    decoder = DECODERS.get(name) or _ALIASES.get(name)
    if decoder is None:
        raise ValueError(f"no decoder is called {name!r}: the decoders are "
                         f"{', '.join(DECODERS)}, and {', '.join(_ALIASES)} are "
                         f"earlier names that are still read")
    return decoder


def choose_decoder(given: str | None, stored: str | None = None) -> Decoder:
    """The decoder of a run, in the one order every script follows.

    `given` is the value of `--decode` in that run, or None when the option was
    not given. `stored` is the name kept in the configuration of the model, or
    None when it keeps none. Then comes `DEFAULT`.
    """
    return get_decoder(given or stored or DEFAULT)
