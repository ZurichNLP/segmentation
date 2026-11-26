"""How the two checkpoints of a run, `best.ckpt` and `last.ckpt`, are written.

`best.ckpt` is written by upstream's ModelCheckpoint when the selection metric
improves. `last.ckpt` is the last validation, written again at each by the
callback of `last_checkpoint`, so that a killed job leaves the weights it reached.
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

Both callbacks are ModelCheckpoints, which Lightning runs after all other
callbacks and in their order, so `last.ckpt` keeps upstream's state of the
same validation: what a run resumed from it would read. `train.py` has no
option to resume yet, and a resume into a new run folder makes upstream's
callback forget its best score.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO
from pytorch_lightning.callbacks import ModelCheckpoint


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


def last_checkpoint(dirpath) -> ModelCheckpoint:
    """The callback that writes `last.ckpt` in `dirpath` at every validation of a fit.

    A stock ModelCheckpoint that monitors nothing keeps one file, the latest,
    under the name `last.ckpt`; the trainer's WholeFileCheckpointIO writes it whole.
    """
    return ModelCheckpoint(dirpath=dirpath, filename="last", monitor=None, save_top_k=1,
                           save_last=False, enable_version_counter=False)


def turn_off_save_last(callbacks: list) -> ModelCheckpoint:
    """Turn off `save_last` of upstream's ModelCheckpoint in `callbacks`, and return it.

    `callbacks` are those upstream's `train()` gives the trainer. The callback
    of `last_checkpoint` writes `last.ckpt` instead, and two writers of one name
    would leave Lightning's copy of `best.ckpt` as `last-v1.ckpt` beside it.
    Stops the run, changing nothing, unless exactly one ModelCheckpoint is
    there: then upstream keeps its checkpoints in another way than this module
    was written for, and going on could leave a run folder with no `best.ckpt`
    or with two writers of `last.ckpt`.
    """
    found = [callback for callback in callbacks if isinstance(callback, ModelCheckpoint)]
    if len(found) != 1:
        raise SystemExit(
            f"upstream's training was to give the trainer one ModelCheckpoint, which "
            f"writes best.ckpt; found {len(found)}: {[type(c).__name__ for c in callbacks]}. "
            f"experiments/checkpoints.py turns off its save_last, because the callback "
            f"of last_checkpoint writes last.ckpt; see its docstring before changing either.")
    found[0].save_last = False
    return found[0]
