"""A run that starts from the weights of another (`--finetune_from` of `experiments/train.py`).

Run from `segment-any-sign/`:
    pytest tests/test_finetune.py

Upstream's training builds the model of `--finetune_from` with its own loader,
`sign_language_segmentation.bin.load_model`; `main()` gives that loader our
model class. The checkpoints are written in pytest's tmp_path by a small model,
as Lightning writes them during training. The tests are marked with `needs`
and skipped, with the missing packages named, where these are not installed.
"""

import hashlib

import pytest

# `project`, the fixture of the working directory, for the tests of `main()`
from tests.test_decoder_use import SMALL, checkpoint_of, project
from tests.test_locations import UPSTREAM, needs
from tests.test_train_outputs import RECORD, checkpoint_with, start_training, stored

#: the model options upstream's `train()` gives the model (`model_kwargs`), for a
#: small model: every hyperparameter but `num_classes`
RUN = {**SMALL, "learning_rate": 1e-3, "lr_scale_backbone": 1.0, "dice_loss_weight": 0.0,
       "optimizer": "adamw-onecycle", "attn_ff_mult": 2, "attn_dropout": 0.1, "fps_aug": False,
       "frame_dropout": 0.15, "num_frames": 1024}


def generators() -> tuple:
    """The states of the random generators of torch, numpy and python."""
    import random

    import numpy as np
    import torch

    return torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate()


def seed() -> None:
    import random

    import numpy as np
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


@needs(*UPSTREAM)
def test_upstream_loader_builds_the_model_of_the_run_with_every_tensor_of_the_source(
        tmp_path, monkeypatch):
    """As upstream's `train()` calls it, with the model options of the run."""
    import torch
    from experiments.validation_metrics import ValidationMetricsModel
    from metrics import get_decoder

    import sign_language_segmentation.bin as upstream_bin

    # the source keeps another decoder and another learning rate than the run
    monkeypatch.setattr(ValidationMetricsModel, "decoder", get_decoder("i-runs-kept"))
    source = checkpoint_of(ValidationMetricsModel(**SMALL, learning_rate=0.5),
                           tmp_path / "best.ckpt")
    # the run, as main() sets the class up for it
    monkeypatch.setattr(ValidationMetricsModel, "decoder", get_decoder("b-honouring"))
    monkeypatch.setattr(ValidationMetricsModel, "class_weights",
                        {"sign": [0.0, 1.0, 9.0, 1.0], "sentence": [0.0, 1.0, 5.0, 1.0]})
    monkeypatch.setattr(upstream_bin, "PoseTaggingModel", ValidationMetricsModel)

    seed()
    control = ValidationMetricsModel(**RUN)            # a run without --finetune_from
    after_control = generators()
    seed()
    tuned = upstream_bin.load_model(model_dir=str(source), device="cpu",
                                    config_overrides=dict(RUN), eval_mode=False)
    after_tuned = generators()

    # the model of the run: its class, its settings and decoder, its loss
    assert type(tuned) is ValidationMetricsModel
    assert dict(tuned.hparams) == dict(control.hparams)
    assert tuned.hparams["decode"] == "b-honouring" and tuned.learning_rate == 1e-3
    assert torch.equal(tuned.phrase_loss_fn.weight, control.phrase_loss_fn.weight)
    # every tensor of the source
    weights = stored(source)["state_dict"]
    assert tuned.state_dict().keys() == weights.keys()
    assert all(torch.equal(tuned.state_dict()[key], value) for key, value in weights.items())
    # and the random generators where building the model leaves them
    assert torch.equal(after_tuned[0], after_control[0])
    assert (after_tuned[1] == after_control[1]).all() and after_tuned[2] == after_control[2]


@needs(*UPSTREAM)
@pytest.mark.parametrize("record", [RECORD, None])
def test_source_of_a_checkpoint_gives_its_file_hash_step_and_record(record, tmp_path,
                                                                   monkeypatch):
    import torch
    from experiments.run_record import source_of

    path = checkpoint_with(record, tmp_path, monkeypatch)
    state = stored(path)
    state["global_step"] = 7
    torch.save(state, path)

    assert source_of(path) == {"file": str(path.resolve()),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "global_step": 7, "record": record}


@needs(*UPSTREAM, "safetensors")
def test_source_of_an_export_gives_its_file_and_hash(tmp_path, monkeypatch):
    """Upstream's export keeps no step and no record."""
    from experiments.run_record import source_of

    from sign_language_segmentation.publish.utils import convert_to_safetensors

    (tmp_path / "export").mkdir()
    convert_to_safetensors(str(checkpoint_with(RECORD, tmp_path, monkeypatch)), tmp_path / "export")
    file = tmp_path / "export" / "model.safetensors"

    assert source_of(file) == {"file": str(file.resolve()),
                               "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                               "global_step": None, "record": None}


@needs(*UPSTREAM)
@pytest.mark.usefixtures("project")
@pytest.mark.parametrize("options, kept, expected", [
    ([], None, "b-honouring"),                                 # none kept: the default
    ([], "i-runs-kept", "b-honouring"),                        # the source's is not used
    (["--decode", "b-ignoring"], "i-runs-kept", "b-ignoring"),  # the one given
])
def test_main_with_finetune_from_records_the_source_and_the_decoder(
        options, kept, expected, tmp_path, monkeypatch):
    from benchmark.predict_dgs_2026 import load_checkpoint
    from experiments.run_record import source_of
    from experiments.validation_metrics import ValidationMetricsModel
    from metrics import get_decoder

    from sign_language_segmentation.model.model import PoseTaggingModel

    # the source: a run of ours that kept a decoder, or upstream's class, which keeps none
    with monkeypatch.context() as source_run:
        if kept:
            source_run.setattr(ValidationMetricsModel, "decoder", get_decoder(kept))
        model = ValidationMetricsModel(**SMALL) if kept else PoseTaggingModel(**SMALL)
        source = checkpoint_of(model, tmp_path / "source.ckpt")
    written = []

    def training(monitor_metric):
        # upstream's train() with --finetune_from: its loader builds the model
        import sign_language_segmentation.train as upstream_train
        from sign_language_segmentation.args import args

        model = upstream_train.load_model(model_dir=args.finetune_from, device="cpu",
                                          config_overrides=dict(RUN), eval_mode=False)
        written.append((model, checkpoint_of(model, tmp_path / "best.ckpt")))

    record = start_training(monkeypatch, "--finetune_from", str(source), *options,
                            training=training)

    [(built, path)] = written
    assert type(built) is ValidationMetricsModel
    checkpoint = stored(path)
    assert record["args"]["decode"] == checkpoint["hyper_parameters"]["decode"] == expected
    assert record["args"]["finetune_source"] == source_of(source)
    assert checkpoint["run_config"] == record
    # and the checkpoint is read back as every script reads one
    assert load_checkpoint(str(path)).stored_decode == expected
