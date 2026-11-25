"""Change the decoder that a prediction file is stored with.

What a prediction file holds
    For each clip, the label of each frame (`pred_bio`) and the segments that one
    decoder made of those labels (`pred`). For the whole file, the name of that
    decoder (`decode`).

What this script does
    It decodes the stored labels again with the decoder you name, writes the result
    as `pred`, and sets `decode`. No model is loaded and no data is read. Nothing
    else in the file changes: not the labels, not the gold, no other field.

Why it exists
    `score.py <file>`, with no option, prints the row of that run in the README. So
    a file must hold the segments of the decoder its row is reported with. When that
    decoder changes for a run that is already recorded, the file has to change too.
    The labels are already in the file, so the model does not need to run again.

Use it when
    the decoder that a recorded row is reported with changes. Example: a run was
    trained without class weights and its file was written with `b-honouring`, but
    its row is reported with `b-ignoring` (README.md, Rules).

Do not use it when
    * You only want to see the scores of a file under another decoder. Use
      `score.py <file> --decode <name>`. It prints the scores and leaves the file
      alone. That is the tool for comparing decoders.
    * The run is new. Launch it with the right `--decode`, and its file is written
      with the right segments from the start.
    * The segments of the file did not come from its labels. The files of the 2023
      models are like that: their segments come from thresholds on the
      probabilities, and the file does not keep the probabilities. Such segments
      cannot be built again from the labels.

How it protects a file
    Before it writes, the script checks that the stored segments equal the stored
    labels decoded with the decoder the file names (`upstream` when it names none:
    such a file was written before 2026-09-09). Only then can the old segments be
    built again later, and only then is the file changed. For any other file it
    writes nothing and says why. A file whose `thresholds` are not empty is refused
    for that alone, and so is a file that is not laid out as `predict_dgs_2026.py`
    writes it, because writing it back would change every line. When several files
    are given and one is refused, none is written. `--dry-run` says what would
    change and writes nothing. Files under `experiments/predictions/` are in git and
    can be restored; files under `benchmark/predictions/` are not.

Environment: **`sas`**, numpy and scikit-learn only, as for `score.py`.

    conda activate sas
    python benchmark/redecode.py <file> --decode b-ignoring --dry-run   # look first
    python benchmark/redecode.py <file> --decode b-ignoring
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.score import unrle
from metrics import decoder_names, get_decoder


def refusal(payload: dict) -> str | None:
    """Why the segments of this file must not be written again, or None when they may.

    They may when the file itself shows that nothing would be lost: its stored
    segments are its stored labels decoded with the decoder it names. Then the
    same segments can be made again at any time, with that decoder.
    """
    if payload.get("thresholds"):
        return ("its `thresholds` are not empty, so its segments were decoded from "
                "probabilities with thresholds, as predict_dgs_2023.py does, and not "
                "from its stored labels, which are the argmax. They cannot be made "
                "again from this file")
    # a file without the field was written before 2026-09-09, by upstream's function
    named = payload.get("decode", "upstream")
    try:
        decoder = get_decoder(named)
    except ValueError as error:
        return f"it names a decoder that does not exist: {error}"
    said = (f"the decoder it names, {named!r}" if "decode" in payload else
            "it names no decoder, which stands for 'upstream'")
    for clip in payload["clips"]:
        if "pred_bio" not in clip:
            return (f"clip {clip.get('id')} has no frame labels to decode; "
                    f"re-run the predict step")
        for level, runs in clip["pred_bio"].items():
            if decoder.decode(unrle(runs)) != clip["pred"].get(level):
                return (f"its stored segments are not its stored labels decoded with "
                        f"{decoder.name} ({said}): they differ first at clip "
                        f"{clip.get('id')}, {level}. Segments that the labels do not "
                        f"give cannot be made again from this file")
    return None


def scores_hint(path: Path, payload: dict) -> str:
    """How to see the scores of a refused file under another decoder, or "".

    `score.py --decode` prints them and leaves the file alone. It is not offered
    for a file it would refuse or could not decode: one with thresholds, or one
    without frame labels.
    """
    if payload.get("thresholds") or any("pred_bio" not in clip for clip in payload["clips"]):
        return ""
    return (f" To see its scores under another decoder without changing it: "
            f"python benchmark/score.py {path} --decode <name>")


def redecode(payload: dict, decoder) -> tuple[dict, int, int]:
    """`payload` with the segments of `decoder`, and how many clip levels changed.

    Returns the new content of the file, the number of clip levels whose
    segments differ from the stored ones, and the number of clip levels.
    `payload` itself is changed too: its clips are not copied.
    """
    changed = total = 0
    for clip in payload["clips"]:
        if "pred_bio" not in clip:
            raise SystemExit(f"clip {clip.get('id')} has no frame labels to decode; "
                             f"re-run the predict step")
        for level, runs in clip["pred_bio"].items():
            segments = decoder.decode(unrle(runs))
            total += 1
            changed += segments != clip["pred"].get(level)
            clip["pred"][level] = segments

    if "decode" in payload:
        payload["decode"] = decoder.name
        return payload, changed, total
    # a file written before the decoder was named: the name goes among the
    # fields of the file, before the clips, as predict_dgs_2026.py writes it
    fields = {key: value for key, value in payload.items() if key != "clips"}
    return {**fields, "decode": decoder.name, "clips": payload["clips"]}, changed, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--decode", required=True, choices=decoder_names(aliases=True),
                        help="the decoder of the new segments. '2023' and 'upstream' "
                             "are the earlier names of b-honouring and b-ignoring; "
                             "the file gets the name of today")
    parser.add_argument("--dry-run", action="store_true",
                        help="say what would change and write nothing")
    args = parser.parse_args()

    decoder = get_decoder(args.decode)
    # Every file is read and checked before any is written, so that one file
    # which is refused leaves all of them as they were.
    payloads = []
    for path in args.predictions:
        text = path.read_text()
        payload = json.loads(text)
        # Only `pred` and `decode` may change. The file is written back whole, so
        # it must be laid out as it will be written, or every line would change.
        if json.dumps(payload, indent=2) != text:
            raise SystemExit(
                f"{path} is refused: it is not laid out as predict_dgs_2026.py writes "
                f"it (json.dumps with indent=2), so writing it back would change lines "
                f"that have nothing to do with the decoder. Nothing was written."
                + scores_hint(path, payload))
        # The stored segments are about to be overwritten: only when the file
        # shows that they can be made again.
        if reason := refusal(payload):
            raise SystemExit(f"{path} is refused: {reason}. Nothing was written."
                             + scores_hint(path, payload))
        payloads.append((path, payload))

    for path, payload in payloads:
        before = payload.get("decode", "none")
        payload, changed, total = redecode(payload, decoder)
        if not args.dry_run:
            path.write_text(json.dumps(payload, indent=2))
        print(f"{path}: decode {before} -> {decoder.name}, segments changed in "
              f"{changed} of {total} clip levels"
              + ("  (dry run: not written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
