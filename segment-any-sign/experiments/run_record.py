"""The record of a training run, read from one of its checkpoints.

Every checkpoint that `train.py` writes keeps the record of its run, the content
of its run_config.json, under the key `run_config`
(`ValidationMetricsModel.on_save_checkpoint` in `validation_metrics.py`). So a
checkpoint copied away from its run folder still says how it was trained. A
checkpoint written before keeps none; its run folder may still hold
run_config.json.

    python -m experiments.run_record dist/<run>/last.ckpt
    python -m experiments.run_record dist/<run>/best.ckpt --key command_as_typed

The first prints the whole record as JSON. The second prints one key of it: a
text as it is, so that `command_as_typed` can be pasted back into a shell, and
anything else as JSON. Run from `segment-any-sign/`. Exit code 1 when the
checkpoint keeps no record, 2 when it has no such key.

The file is read as every loader of ours reads a checkpoint, with
`weights_only=False`, so only open checkpoints you trust.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def record_of(checkpoint: str | Path) -> dict | None:
    """The record of the run that wrote `checkpoint`, or None when it keeps none."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return state.get("run_config")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="a .ckpt written by train.py")
    parser.add_argument("--key", default=None,
                        help="print only this key of the record, e.g. command_as_typed "
                             "or options")
    args = parser.parse_args(argv)

    record = record_of(args.checkpoint)
    if record is None:
        print(f"{args.checkpoint} keeps no record of its run: it was written before "
              f"checkpoints kept one. Its run folder may still hold run_config.json.",
              file=sys.stderr)
        return 1
    if args.key is None:
        value = record
    elif args.key in record:
        value = record[args.key]
    else:
        print(f"the record has no key {args.key!r}; its keys are {', '.join(record)}",
              file=sys.stderr)
        return 2
    print(value if isinstance(value, str) else json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
