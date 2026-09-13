"""Append a finished pretraining run to the Runs table in experiments/README.md.

Called by `train.py` as the last step of the post-training evaluation, so a run
that completes lands in the table without anyone copying numbers by hand. Every
number comes from a file the evaluation just wrote — nothing is re-scored here:

  * step         `global_step` stored in the selected checkpoint
  * YouTube dev  the JSON `eval_youtube.py --json` writes, raw and filtered
  * DGS          the prediction file, through `benchmark/score.py`

    python experiments/add_result_row.py --run 02_pretrain_youtube-2026.09.14 \\
        --note "weights 2,80,1, 40k steps"

The `change` column is the one thing a script cannot know, so it takes `--note`;
without one the row says so, to be filled in. A run already in the table is left
alone, so re-running is safe. Two runs finishing at once take a file lock rather
than each writing over the other's row.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HERE = Path(__file__).resolve().parent
README = HERE / "README.md"
DIST = Path("/scratch/zifjia/segment-any-sign/dist")

#: the header cell that identifies the pretraining table, and nothing else
TABLE_ANCHOR = "| Filtered YouTube dev |"
METRICS = ("frame_f1", "iou", "percentage", "mf1s")


def _cells(scores: dict | None) -> list[str]:
    if not scores:
        return ["—"] * len(METRICS)
    return [f"{scores[key]:.3f}" for key in METRICS]


def build_row(number: int, run: str, note: str, step: int,
              youtube: dict, dgs: dict | None) -> str:
    cells = [str(number), f"`{run}`", note, f"{step:,}",
             *_cells(youtube.get("filtered")), *_cells(youtube.get("raw")),
             *_cells(dgs)]
    return "| " + " | ".join(cells) + " |"


def insert_row(text: str, run: str, make_row) -> tuple[str, str | None]:
    """Put the row after the last numbered row of the table.

    Returns the new text and the row, or the text unchanged and None when the
    run is already there.
    """
    lines = text.split("\n")
    try:
        start = next(i for i, line in enumerate(lines) if TABLE_ANCHOR in line)
    except StopIteration:
        raise SystemExit(f"no pretraining table (a line containing "
                         f"{TABLE_ANCHOR!r}) in the README")
    end = start
    while end + 1 < len(lines) and lines[end + 1].startswith("|"):
        end += 1
    table = lines[start:end + 1]
    if any(f"`{run}`" in line for line in table):
        return text, None
    numbers = [int(m.group(1)) for line in table
               if (m := re.match(r"\|\s*(\d+)\s*\|", line))]
    row = make_row(max(numbers, default=0) + 1)
    lines.insert(end + 1, row)
    return "\n".join(lines), row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run directory under dist/")
    parser.add_argument("--note", default="",
                        help="the `change` column: what this run does differently")
    parser.add_argument("--youtube", type=Path, default=None,
                        help="eval_youtube.py JSON (default: <run>/youtube_dev.json)")
    parser.add_argument("--dgs", type=Path, default=None,
                        help="DGS validation predictions (default: "
                             "experiments/predictions/<run id>.json)")
    parser.add_argument("--readme", type=Path, default=README)
    args = parser.parse_args()

    import torch

    from benchmark.score import score_file
    from experiments.plot_segmentation import newest_checkpoint

    run_dir = DIST / args.run
    run_id = args.run.rsplit("-", 1)[0]
    checkpoint = newest_checkpoint(run_dir, "best")
    step = int(torch.load(checkpoint, map_location="cpu",
                          weights_only=False)["global_step"])

    youtube_path = args.youtube or run_dir / "youtube_dev.json"
    youtube = json.loads(youtube_path.read_text())
    dgs_path = args.dgs or HERE / "predictions" / f"{run_id}.json"
    dgs = score_file(dgs_path)["levels"].get("phrase") if dgs_path.exists() else None

    note = args.note.strip() or "*(describe the change)*"
    with open(args.readme, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text, row = insert_row(
            handle.read(), args.run,
            lambda number: build_row(number, args.run, note, step, youtube, dgs))
        if row is None:
            print(f"{args.run} is already in {args.readme.name} — left as is")
            return
        handle.seek(0)
        handle.write(text)
        handle.truncate()
    print(f"added to {args.readme}:\n{row}")


if __name__ == "__main__":
    main()
