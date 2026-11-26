"""How the checkpoints of a run are written (`experiments/checkpoints.py`).

Run from `segment-any-sign/`:
    pytest tests/test_checkpoints.py

A run folder keeps `best.ckpt`, which upstream's ModelCheckpoint writes when
the selection metric improves, and `last.ckpt`, which the callback of
`last_checkpoint` writes at every validation. The trainings here are of a
module with one weight, on CPU, in pytest's tmp_path, validated every two
steps across epochs as `train.py` validates. Its metric at each validation is given by the test. The
tests that need Lightning are marked with `needs` and skipped, with the
missing packages named, where it is not installed.
"""

from pathlib import Path

import pytest

from tests.test_locations import needs

LIGHTNING = ("pytorch_lightning", "torch")


def upstream_checkpoint(folder, monitor="m"):
    """Upstream's ModelCheckpoint, with the arguments of its `train()`."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    return ModelCheckpoint(dirpath=folder, filename="best", verbose=True, save_top_k=1,
                           save_last=True, monitor=monitor, every_n_epochs=1, mode="max")


def train(folder, metrics: list[float], callbacks: list, plugins=None):
    """Train for two steps per value of `metrics`, logged as `m` at each validation."""
    import pytorch_lightning as pl
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    class OneWeight(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Linear(1, 1)
            self.validations = 0

        def training_step(self, batch, index):
            return self.weight(batch[0]).pow(2).mean()

        def validation_step(self, batch, index):
            pass

        def on_validation_epoch_end(self):
            self.log("m", metrics[self.validations])
            self.validations += 1

        def configure_optimizers(self):
            # with momentum, so that the optimizer keeps a state of its own
            return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    model = OneWeight()
    trainer = pl.Trainer(max_steps=2 * len(metrics), val_check_interval=2,
                         check_val_every_n_epoch=None, num_sanity_val_steps=0,
                         logger=False, enable_progress_bar=False,
                         enable_model_summary=False, accelerator="cpu",
                         default_root_dir=folder, callbacks=callbacks, plugins=plugins)
    trainer.fit(model, DataLoader(TensorDataset(torch.randn(8, 1))),
                DataLoader(TensorDataset(torch.randn(1, 1))))
    return model


def stored(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def best_score_inside(checkpoint: dict) -> float:
    """The best score that upstream's ModelCheckpoint had when `checkpoint` was written."""
    [state] = [state for key, state in checkpoint["callbacks"].items()
               if "'monitor': 'm'" in key]
    return float(state["best_model_score"])


@needs(*LIGHTNING)
@pytest.mark.parametrize("metrics, best_step, best_score", [
    ([0.1, 0.3, 0.5, 0.4, 0.2, 0.1], 6, 0.5),     # the best validation comes before the end
    ([0.1, 0.2, 0.3], 6, 0.3),                    # the last validation is the best
])
def test_a_run_keeps_its_best_and_its_last_validation(metrics, best_step, best_score,
                                                       tmp_path):
    import logging

    import torch
    from experiments.checkpoints import WholeFileCheckpointIO, last_checkpoint, turn_off_save_last

    # Lightning's own logger does not pass its records on to the root logger,
    # where caplog listens, so they are kept here
    logged = []
    keep = logging.Handler()
    keep.emit = lambda record: logged.append(record.getMessage())
    lightning = logging.getLogger("pytorch_lightning")
    lightning.addHandler(keep)
    callbacks = [upstream_checkpoint(tmp_path)]
    turn_off_save_last(callbacks)
    try:
        model = train(tmp_path, metrics, callbacks + [last_checkpoint(tmp_path)],
                      plugins=[WholeFileCheckpointIO()])
    finally:
        lightning.removeHandler(keep)

    assert sorted(path.name for path in tmp_path.glob("*.ckpt*")) == ["best.ckpt", "last.ckpt"]
    best, last = stored(tmp_path / "best.ckpt"), stored(tmp_path / "last.ckpt")
    assert best["global_step"] == best_step
    assert last["global_step"] == 2 * len(metrics)           # the last validation
    for key, value in model.state_dict().items():            # the weights of the end
        assert torch.equal(last["state_dict"][key], value), key
    # written after upstream's callback has seen the same validation
    assert best_score_inside(last) == pytest.approx(best_score)
    # the whole state of the training, the momentum of the weight included
    assert last["optimizer_states"] and last["optimizer_states"][0]["state"]
    assert "lr_schedulers" in last and "loops" in last
    # and no line at each validation about how much of it was saved
    assert not [line for line in logged if "weights_only" in line]


@needs(*LIGHTNING)
def test_save_last_of_upstream_is_turned_off(tmp_path):
    from experiments.checkpoints import turn_off_save_last
    from pytorch_lightning.callbacks import EarlyStopping

    upstream = upstream_checkpoint(tmp_path)
    callbacks = [EarlyStopping(monitor="m", mode="max"), upstream]
    assert turn_off_save_last(callbacks) is upstream
    assert upstream.save_last is False


@needs(*LIGHTNING)
@pytest.mark.parametrize("found", [0, 2])
def test_the_run_stops_unless_upstream_gives_exactly_one_checkpoint_callback(found, tmp_path):
    from experiments.checkpoints import turn_off_save_last
    from pytorch_lightning.callbacks import EarlyStopping

    upstreams = [upstream_checkpoint(tmp_path / str(n)) for n in range(found)]
    callbacks = [EarlyStopping(monitor="m", mode="max"), *upstreams]
    with pytest.raises(SystemExit) as stop:
        turn_off_save_last(callbacks)
    message = str(stop.value)
    assert f"found {found}" in message and "checkpoints.py" in message
    # and nothing was changed
    assert all(callback.save_last is True for callback in upstreams)


# --- WholeFileCheckpointIO: every checkpoint written whole ----------------------

def a_checkpoint(step: int) -> dict:
    """A checkpoint as Lightning hands it to its CheckpointIO, cut to a few keys."""
    import torch

    return {"global_step": step, "state_dict": {"weight": torch.full((3, 2), float(step))},
            "optimizer_states": [{"state": {0: {"momentum_buffer": torch.ones(2)}}}]}


@needs(*LIGHTNING)
def test_a_save_writes_the_bytes_that_lightning_writes(tmp_path):
    """Only where and how the file is written changes, not what it holds."""
    from experiments.checkpoints import WholeFileCheckpointIO
    from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO

    WholeFileCheckpointIO().save_checkpoint(a_checkpoint(6), tmp_path / "ours" / "best.ckpt")
    TorchCheckpointIO().save_checkpoint(a_checkpoint(6), tmp_path / "lightning" / "best.ckpt")

    ours = (tmp_path / "ours" / "best.ckpt").read_bytes()
    assert ours == (tmp_path / "lightning" / "best.ckpt").read_bytes()
    assert sorted(path.name for path in (tmp_path / "ours").iterdir()) == ["best.ckpt"]


@needs(*LIGHTNING)
def test_a_save_that_dies_midway_leaves_the_checkpoint_before_whole(tmp_path, monkeypatch):
    import torch
    from experiments.checkpoints import WholeFileCheckpointIO

    target = tmp_path / "best.ckpt"
    WholeFileCheckpointIO().save_checkpoint(a_checkpoint(2), target)
    real_save = torch.save

    def killed(checkpoint, handle, *arguments, **options):
        handle.write(b"half of th")
        raise RuntimeError("killed")

    monkeypatch.setattr(torch, "save", killed)
    with pytest.raises(RuntimeError):
        WholeFileCheckpointIO().save_checkpoint(a_checkpoint(8), target)
    monkeypatch.setattr(torch, "save", real_save)

    assert stored(target)["global_step"] == 2
    # what is left beside it is matched by no reader of a run folder
    assert (tmp_path / "best.ckpt.part").read_bytes() == b"half of th"
    for pattern in ("best*.ckpt", "last*.ckpt", "*.ckpt"):
        assert [path.name for path in tmp_path.glob(pattern) if path.name != "best.ckpt"] == []


@needs(*LIGHTNING)
def test_a_save_with_storage_options_is_refused_as_lightning_refuses_it(tmp_path):
    from experiments.checkpoints import WholeFileCheckpointIO

    with pytest.raises(TypeError):
        WholeFileCheckpointIO().save_checkpoint(a_checkpoint(6), tmp_path / "best.ckpt",
                                                storage_options={"anything": 1})
    assert list(tmp_path.glob("best.ckpt*")) == []


def dies_at_the_second_write_of(name: str):
    """`torch.save` as it is, except that its second write of `<name>.part` dies midway."""
    import torch

    real_save, writes = torch.save, []

    def save(checkpoint, handle, *arguments, **options):
        if str(getattr(handle, "name", "")).endswith(f"/{name}.part"):
            writes.append(name)
            if len(writes) == 2:
                handle.write(b"half of th")
                raise RuntimeError("killed")
        return real_save(checkpoint, handle, *arguments, **options)

    return save


@needs(*LIGHTNING)
@pytest.mark.parametrize("killed, best_step, last_step", [
    # the case of the review: the best of step 2, then a better one at step 8
    ("best.ckpt", 2, 6),
    # the second write of last.ckpt, at step 4
    ("last.ckpt", 2, 2),
])
def test_a_kill_during_a_write_leaves_both_checkpoints_whole(killed, best_step, last_step,
                                                             tmp_path, monkeypatch):
    import torch
    from experiments.checkpoints import WholeFileCheckpointIO, last_checkpoint, turn_off_save_last

    callbacks = [upstream_checkpoint(tmp_path)]
    turn_off_save_last(callbacks)
    monkeypatch.setattr(torch, "save", dies_at_the_second_write_of(killed))
    with pytest.raises(RuntimeError):
        train(tmp_path, [0.3, 0.1, 0.2, 0.5], callbacks + [last_checkpoint(tmp_path)],
              plugins=[WholeFileCheckpointIO()])

    best, last = stored(tmp_path / "best.ckpt"), stored(tmp_path / "last.ckpt")
    assert best["global_step"] == best_step and best_score_inside(best) == pytest.approx(0.3)
    assert last["global_step"] == last_step              # the validation before the kill
    assert (tmp_path / f"{killed}.part").exists()
