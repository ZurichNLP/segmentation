"""Write the segments of a prediction file again, with another decoder.

A prediction file stores the frame labels of a model (`pred_bio`), the segments
one decoder made of them (`pred`), and the name of that decoder (`decode`).
`score.py --decode` scores the labels with another decoder and leaves the file
alone. This script changes the file: it decodes `pred_bio` of every clip with
the decoder named, writes the result as `pred`, and sets `decode`.

Use it when the decoder a row is reported with changes (README.md, Rules). After
it the file alone says how its row is scored, and `score.py <file>` with no
option prints that row.

Nothing else in the file changes: not the labels, not the gold, no other field,
and the file keeps its layout. No model is loaded and no data is read, so the
segments are those of the labels that are already there.

Environment: **`sas`** — numpy and scikit-learn only, as for `score.py`.

    conda activate sas
    python benchmark/redecode.py experiments/predictions/<run>.json --decode b-ignoring
    python benchmark/redecode.py <file> --decode b-ignoring --dry-run   # write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.score import unrle
from metrics import decoder_names, get_decoder


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
    for path in args.predictions:
        text = path.read_text()
        payload = json.loads(text)
        # Only `pred` and `decode` may change. The file is written back whole, so
        # it must be laid out as it will be written, or every line would change.
        if json.dumps(payload, indent=2) != text:
            raise SystemExit(
                f"{path} is not laid out as predict_dgs_2026.py writes it "
                f"(json.dumps with indent=2): writing it back would change lines "
                f"that have nothing to do with the decoder. Nothing was written.")
        before = payload.get("decode", "none")
        payload, changed, total = redecode(payload, decoder)
        if not args.dry_run:
            path.write_text(json.dumps(payload, indent=2))
        print(f"{path}: decode {before} -> {decoder.name}, segments changed in "
              f"{changed} of {total} clip levels"
              + ("  (dry run: not written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
