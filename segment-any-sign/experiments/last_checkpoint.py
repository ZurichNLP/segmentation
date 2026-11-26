"""The last checkpoint of a run: `last.ckpt`, written again at every validation.

A run folder keeps two checkpoints. `best.ckpt` is written by upstream's
ModelCheckpoint whenever the selection metric improves. `last.ckpt` is the run
at its last validation, so that a job that is killed leaves the weights of its
last validation, and the end of a run can be read beside its best.

Upstream asks its ModelCheckpoint for `save_last=True`. In the Lightning of
this project (2.6.6) that writes `last.ckpt` only at a step where a checkpoint
was saved, so with `save_top_k=1` it was a copy of `best.ckpt`: in every run
folder checked, the two held the same step and the same weights. `train.py`
therefore turns that off (`turn_off_save_last`) and gives the trainer
`LastCheckpoint`, so that one writer makes `last.ckpt`.

Why not a second, stock ModelCheckpoint with `monitor=None`; measured on
2026-11-26:

  * Without a filename it writes a new name at every validation, from
    `epoch=0-step=2.ckpt` to `epoch=1-step=12.ckpt` in a run of twelve steps,
    and deletes the one before, so no `last*.ckpt` of a reader finds it.
  * With `filename="last"` it writes `last.ckpt` in place. Lightning saves to a
    file in /tmp and then moves it, and on the UZH cluster /tmp and /scratch
    are different mounts, so the move is a copy into `last.ckpt` itself. A
    SIGKILL during that copy left `last.ckpt` with 81,920 of 600,001,641 bytes,
    which `torch.load` refuses ("failed finding central directory"), and the
    checkpoint of the validation before was gone with it.

`LastCheckpoint` writes `last.ckpt.part` and renames it over `last.ckpt`, which
is atomic on one file system: the same kill left `last.ckpt` whole, with the
validation before, beside a broken `last.ckpt.part`. No reader of a run folder
matches the `.part`, and the next validation writes over it. Either way the
kill also leaves Lightning's whole temporary copy in /tmp. A SIGTERM, from a
time limit or `scancel`, does no harm: Lightning only notes it, and SLURM waits
KillWait (200 s on that cluster) before it sends SIGKILL, so a write finishes.

It is a ModelCheckpoint with nothing of its own to save, not a plain Callback,
for the order. Lightning runs every ModelCheckpoint after all other callbacks,
in the order they were given, so this one runs after upstream's, and
`last.ckpt` keeps upstream's state of the same validation. A plain Callback
runs before it: when the last validation was also the best, its `last.ckpt`
said that the best score was the one before (0.2 where it was 0.3, in a
check). The weights are the same either way; that state is what a run resumed
from `last.ckpt` would read.
"""

from __future__ import annotations

import os
from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import TrainerFn

NAME = "last.ckpt"
#: what a save writes first; renamed to NAME when it is whole
PART = NAME + ".part"


class LastCheckpoint(ModelCheckpoint):
    """Write `last.ckpt` in `dirpath` at the end of every validation of a fit.

    It keeps the whole state of the training, optimizer included, not only the
    weights.
    """

    def __init__(self, dirpath):
        # nothing of the parent's to save: no top-k checkpoints and no save_last
        super().__init__(dirpath=dirpath, save_top_k=0, save_last=False)

    def on_validation_end(self, trainer, pl_module) -> None:
        if (trainer.sanity_checking or trainer.fast_dev_run
                or trainer.state.fn != TrainerFn.FITTING):
            return
        folder = Path(self.dirpath)
        # The whole state of the training: weights, optimizer, schedulers, loops
        # and callbacks, as upstream's ModelCheckpoint saves best.ckpt. Said
        # explicitly: left unset, Lightning saves the same and logs a line about
        # it at every validation.
        trainer.save_checkpoint(folder / PART, weights_only=False)
        if trainer.is_global_zero:
            os.replace(folder / PART, folder / NAME)
        trainer.strategy.barrier()


def turn_off_save_last(callbacks: list) -> ModelCheckpoint:
    """Turn off `save_last` of upstream's ModelCheckpoint in `callbacks`, and return it.

    `callbacks` are those upstream's `train()` gives the trainer. `LastCheckpoint`
    writes `last.ckpt` instead, and two writers of one name would leave
    Lightning's copy of `best.ckpt` as `last-v1.ckpt` beside it. Stops the run,
    changing nothing, unless exactly one ModelCheckpoint other than a
    LastCheckpoint is there: then upstream keeps its checkpoints in another way
    than this module was written for, and going on could leave a run folder
    with no `best.ckpt` or with two writers of `last.ckpt`.
    """
    found = [callback for callback in callbacks
             if isinstance(callback, ModelCheckpoint)
             and not isinstance(callback, LastCheckpoint)]
    if len(found) != 1:
        raise SystemExit(
            f"upstream's training was to give the trainer one ModelCheckpoint, which "
            f"writes best.ckpt; found {len(found)}: {[type(c).__name__ for c in callbacks]}. "
            f"experiments/last_checkpoint.py turns off its save_last, because "
            f"LastCheckpoint writes last.ckpt; see its docstring before changing either.")
    found[0].save_last = False
    return found[0]
