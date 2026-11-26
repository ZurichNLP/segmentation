"""How the two checkpoints of a run, `best.ckpt` and `last.ckpt`, are written.

`best.ckpt` is written by upstream's ModelCheckpoint when the selection metric
improves. `last.ckpt` is the last validation, written again at each by
`LastCheckpoint`, so that a job that is killed leaves the weights it reached.
Upstream's `save_last` wrote `last.ckpt` only together with `best.ckpt`, as a
copy of it, so `train.py` turns it off (`turn_off_save_last`).

Lightning saves a checkpoint to a file in /tmp and then moves it. Where /tmp
is another mount than the run folder, as on the cluster, the move is a copy
into the checkpoint, and a kill during it truncates the file. So `train.py`
gives the trainer `WholeFileCheckpointIO`, through which Lightning writes
every checkpoint: to `<name>.part`, then renamed over `<name>`. A kill
leaves both checkpoints whole, beside a `.part` that no reader matches. A
SIGTERM, from a time limit or `scancel`, stops the run at the next training
batch without saving anything: what is left is `best.ckpt` and the `last.ckpt`
of the last finished validation. SLURM waits KillWait before it sends SIGKILL,
so a write in progress finishes.

`LastCheckpoint` is a ModelCheckpoint with nothing of its own to save, not a
plain Callback, for the order. Lightning runs every ModelCheckpoint after all
other callbacks, in the order they were given, so this one runs after
upstream's, and `last.ckpt` keeps upstream's state of the same validation. A plain Callback
runs before it: when the last validation was also the best, its `last.ckpt`
holds the best score of the validation before. The weights are the same either
way; that state is what a run resumed from `last.ckpt` would read. `train.py`
has no option to resume yet, and a resume into a new run folder makes
upstream's callback forget its best score.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import TrainerFn

NAME = "last.ckpt"


class WholeFileCheckpointIO(TorchCheckpointIO):
    """Write a checkpoint to `<name>.part`, then rename it to `<name>`.

    A kill at any moment leaves a whole file under that name, the old or the
    new. `torch.save` is given a file object, as in TorchCheckpointIO, so the
    bytes are the same as its. Not covered, as with Lightning's own save, which
    does not sync either: a crash of the node in the seconds after a save.
    """

    def save_checkpoint(self, checkpoint: dict, path, storage_options=None) -> None:
        if storage_options is not None:
            raise TypeError(f"{type(self).__name__} takes no storage_options")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        with open(part, "wb") as handle:
            torch.save(checkpoint, handle)
        os.replace(part, target)


class LastCheckpoint(ModelCheckpoint):
    """Write `last.ckpt` in `dirpath` at the end of every validation of a fit.

    It keeps the whole state of the training, optimizer included, not only the
    weights. It is written whole by the trainer's `WholeFileCheckpointIO`, and a
    fit without it is stopped.
    """

    def __init__(self, dirpath):
        # nothing of the parent's to save: no top-k checkpoints and no save_last
        super().__init__(dirpath=dirpath, save_top_k=0, save_last=False)

    def setup(self, trainer, pl_module, stage) -> None:
        super().setup(trainer, pl_module, stage)
        if not isinstance(trainer.strategy.checkpoint_io, WholeFileCheckpointIO):
            raise SystemExit(
                "LastCheckpoint needs a trainer given WholeFileCheckpointIO in its "
                "plugins: without it a kill during a write would truncate last.ckpt. "
                "See experiments/checkpoints.py.")

    def on_validation_end(self, trainer, pl_module) -> None:
        if (trainer.sanity_checking or trainer.fast_dev_run
                or trainer.state.fn != TrainerFn.FITTING):
            return
        # The whole state of the training: weights, optimizer, schedulers, loops
        # and callbacks, as upstream's ModelCheckpoint saves best.ckpt. Said
        # explicitly: left unset, Lightning saves the same and logs a line about
        # it at every validation.
        trainer.save_checkpoint(Path(self.dirpath) / NAME, weights_only=False)


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
            f"experiments/checkpoints.py turns off its save_last, because "
            f"LastCheckpoint writes last.ckpt; see its docstring before changing either.")
    found[0].save_last = False
    return found[0]
