"""How the scripts take a decoder.

Run from `segment-any-sign/`:
    pytest tests/test_decoder_use.py

The decoders themselves are tested in test_decoders.py. Each test here imports
what it tests, so one missing function cannot stop the others.
"""

import json
import sys

import pytest

# the model is built on the upstream package: see test_locations.py
from tests.test_locations import UPSTREAM, needs

# --- a prediction file written by hand ------------------------------------------

O, B, I = 0, 1, 2


def rle(values) -> list[list[int]]:
    runs: list[list[int]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return runs


def spans(*pairs) -> list[dict]:
    return [{"start": start, "end": end} for start, end in pairs]


#: One clip of 8 frames. The stored segments are what no decoder gives from the
#: stored labels, so a score says which of the two it was computed from.
#:
#:              frame   0 1 2 3 4 5 6 7
#:   sign    gold       O B I I O B I O      1-3, 5-6
#:           predicted  O I I I O B I O      b-honouring 5-6; the other two 1-3, 5-6
#:   phrase  gold       O B I I I I I O      1-6
#:           predicted  O B I I B I I O      b-ignoring 1-6; the other two 1-3, 4-6
CLIP = {
    "id": "by_hand", "num_frames": 8, "fps": 50.0,
    "gold": {"sign": spans((1, 3), (5, 6)), "phrase": spans((1, 6))},
    "pred": {"sign": spans((0, 7)), "phrase": spans((2, 5))},
    "gold_bio": {"sign": rle([O, B, I, I, O, B, I, O]),
                 "phrase": rle([O, B, I, I, I, I, I, O])},
    "pred_bio": {"sign": rle([O, I, I, I, O, B, I, O]),
                 "phrase": rle([O, B, I, I, B, I, I, O])},
}

#: IoU and % of that clip, sign then phrase, by where the segments come from
STORED = {"sign": (5 / 8, 0.5), "phrase": (4 / 6, 1.0)}
DECODED = {
    "b-honouring": {"sign": (2 / 5, 0.5), "phrase": (1.0, 2.0)},
    "b-ignoring": {"sign": (1.0, 1.0), "phrase": (1.0, 1.0)},
    "i-runs-kept": {"sign": (1.0, 1.0), "phrase": (1.0, 2.0)},
}


@pytest.fixture
def prediction_file(tmp_path):
    path = tmp_path / "by_hand.json"
    path.write_text(json.dumps({"dataset": "by_hand", "split": "validation",
                                "model": "model by hand", "thresholds": {},
                                "decode": "2023", "clips": [CLIP]}, indent=2))
    return path


def iou_and_percentage(row: dict) -> dict:
    return {level: (pytest.approx(values["iou"]), pytest.approx(values["percentage"]))
            for level, values in row["levels"].items()}


# --- score.py ---------------------------------------------------------------------

def test_score_file_scores_the_stored_segments_when_no_decoder_is_given(prediction_file):
    from benchmark.score import score_file

    row = score_file(prediction_file)
    assert iou_and_percentage(row) == STORED
    assert row["decode"] is None


@pytest.mark.parametrize("name", list(DECODED))
def test_score_file_with_a_decoder_decodes_the_stored_labels(prediction_file, name):
    from benchmark.score import score_file
    from metrics import get_decoder

    row = score_file(prediction_file, decoder=get_decoder(name))
    assert iou_and_percentage(row) == DECODED[name]
    assert row["decode"] == name


def test_frame_f1_does_not_depend_on_the_decoder(prediction_file):
    from benchmark.score import score_file
    from metrics import DECODERS

    stored = score_file(prediction_file)["levels"]
    for decoder in DECODERS.values():
        decoded = score_file(prediction_file, decoder=decoder)["levels"]
        for level in ("sign", "phrase"):
            for metric in ("frame_f1", "frame_f1_micro"):
                assert decoded[level][metric] == stored[level][metric]


def test_the_prediction_file_is_not_changed_by_scoring_with_a_decoder(prediction_file):
    from benchmark.score import score_file
    from metrics import get_decoder

    before = prediction_file.read_bytes()
    score_file(prediction_file, decoder=get_decoder("b-ignoring"))
    assert prediction_file.read_bytes() == before


def run_score(monkeypatch, capsys, *arguments) -> str:
    from benchmark import score

    monkeypatch.setattr(sys, "argv", ["score.py", *[str(argument) for argument in arguments]])
    score.main()
    return capsys.readouterr().out


def numbers(line: str) -> list[str]:
    return [word for word in line.split() if word.replace(".", "").isdigit() and "." in word]


def test_command_line_without_decode_prints_the_stored_segments(prediction_file,
                                                                monkeypatch, capsys):
    out = run_score(monkeypatch, capsys, prediction_file)
    row = next(line for line in out.splitlines() if line.startswith("model by hand"))
    assert "[" not in row
    assert numbers(row)[2:4] == ["0.625", "0.500"]          # sign IoU and %


@pytest.mark.parametrize("option", ["b-ignoring", "upstream"])
def test_command_line_with_decode_names_the_decoder_in_the_row(prediction_file, option,
                                                               monkeypatch, capsys):
    out = run_score(monkeypatch, capsys, prediction_file, "--decode", option)
    row = next(line for line in out.splitlines() if line.startswith("model by hand"))
    # the earlier name is read, and the row shows the name of today
    assert row.startswith("model by hand [b-ignoring]")
    assert numbers(row)[2:4] == ["1.000", "1.000"]


def test_command_line_refuses_a_name_that_is_not_a_decoder(prediction_file, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run_score(monkeypatch, capsys, prediction_file, "--decode", "likeliest")
    assert "b-honouring" in capsys.readouterr().err


def test_per_clip_lines_use_the_decoder(prediction_file, monkeypatch, capsys):
    stored = run_score(monkeypatch, capsys, prediction_file, "--per-clip")
    decoded = run_score(monkeypatch, capsys, prediction_file, "--per-clip",
                        "--decode", "b-honouring")
    line = next(line for line in stored.splitlines() if line.strip().startswith("by_hand"))
    assert "pred     1" in line and "IoU 0.6250" in line
    line = next(line for line in decoded.splitlines() if line.strip().startswith("by_hand"))
    assert "pred     1" in line and "IoU 0.4000" in line


def test_out_file_records_the_decoder(prediction_file, tmp_path, monkeypatch, capsys):
    out = tmp_path / "scores.json"
    run_score(monkeypatch, capsys, prediction_file, "--decode", "i-runs-kept", "--out", out)
    assert json.loads(out.read_text())[0]["decode"] == "i-runs-kept"


# --- training: the decoder of a run, and where it is kept -----------------------

#: a model small enough to build in a test; the arguments are upstream's
SMALL = {"pose_dims": (50, 3), "hidden_dim": 64, "encoder_depth": 1,
         "steps_per_epoch": 5, "max_epochs": 2, "attn_nhead": 4}


def checkpoint_of(model, path):
    """Write `model` to `path` as Lightning writes a checkpoint during training."""
    import pytorch_lightning as pl

    trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator="cpu",
                         enable_progress_bar=False, enable_model_summary=False,
                         max_epochs=1)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(path)
    return path


@needs(*UPSTREAM)
def test_model_decodes_with_b_honouring_when_nothing_sets_its_decoder():
    from experiments.validation_metrics import ValidationMetricsModel

    assert ValidationMetricsModel.decoder.name == "b-honouring"
    assert ValidationMetricsModel(**SMALL).hparams["decode"] == "b-honouring"


@needs(*UPSTREAM)
@pytest.mark.parametrize("name", ["b-honouring", "b-ignoring", "i-runs-kept"])
def test_checkpoint_keeps_the_name_of_the_decoder_of_the_run(name, tmp_path, monkeypatch):
    import torch
    from experiments.validation_metrics import ValidationMetricsModel
    from metrics import get_decoder

    from sign_language_segmentation.model.model import PoseTaggingModel

    monkeypatch.setattr(ValidationMetricsModel, "decoder", get_decoder(name))
    path = checkpoint_of(ValidationMetricsModel(**SMALL), tmp_path / "best.ckpt")

    stored = torch.load(path, map_location="cpu", weights_only=False)["hyper_parameters"]
    assert stored["decode"] == name
    # only the decoder is added to what upstream's model keeps
    assert set(stored) == set(PoseTaggingModel(**SMALL).hparams) | {"decode"}


@needs(*UPSTREAM)
def test_upstream_class_still_loads_a_checkpoint_that_keeps_a_decoder(tmp_path, monkeypatch):
    import torch
    from experiments.validation_metrics import ValidationMetricsModel
    from metrics import get_decoder

    from sign_language_segmentation.model.model import PoseTaggingModel

    monkeypatch.setattr(ValidationMetricsModel, "decoder", get_decoder("b-ignoring"))
    model = ValidationMetricsModel(**SMALL)
    path = checkpoint_of(model, tmp_path / "best.ckpt")

    loaded = PoseTaggingModel.load_from_checkpoint(path, map_location="cpu")
    # the other hyperparameters come back, and so do the weights
    assert loaded.hparams["hidden_dim"] == 64
    assert tuple(loaded.hparams["pose_dims"]) == (50, 3)
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key


def one_hot_log_probs(upstream_ids):
    import torch

    log_probs = torch.full((1, len(upstream_ids), 4), -20.0)
    log_probs[0, torch.arange(len(upstream_ids)), torch.tensor(upstream_ids)] = 0.0
    return log_probs


@needs(*UPSTREAM)
@pytest.mark.parametrize("name, iou, percentage", [("b-honouring", 0.0, 0.0),
                                                   ("b-ignoring", 1.0, 1.0),
                                                   ("i-runs-kept", 1.0, 1.0)])
def test_validation_scores_the_segments_of_the_decoder_of_the_run(name, iou, percentage,
                                                                  monkeypatch):
    """Gold `O B I I O`, predicted `O I I I O`: the B is missing.

    With `b-honouring` there is no segment, so IoU and `%` are 0. The other two
    decoders make the run of I a segment, which is the gold one.
    """
    import torch
    from experiments.validation_metrics import ValidationMetricsModel, _empty
    from metrics import get_decoder

    monkeypatch.setattr(ValidationMetricsModel, "decoder", get_decoder(name))
    model = ValidationMetricsModel(**SMALL)
    gold = torch.tensor([[1, 2, 3, 3, 1]])           # upstream ids: UNK=0, O=1, B=2, I=3
    batch = {"pose": torch.zeros(1, 5, 50, 3), "bio": {"sign": gold, "sentence": gold}}
    predicted = one_hot_log_probs([1, 3, 3, 3, 1])

    collected = _empty()
    model._accumulate(batch, collected, log_probs={"sign": predicted, "sentence": predicted})
    for level in ("sign", "phrase"):
        assert collected[level]["iou"] == [iou]
        assert collected[level]["percentage"] == [percentage]
        # the frame metrics do not see the decoder
        assert collected[level]["frame_f1_micro"] == [pytest.approx(4 / 5)]


def missing_b_batch():
    """Gold `O B I I O` and prediction `O I I I O`, at both levels: see the test above."""
    import torch

    gold = torch.tensor([[1, 2, 3, 3, 1]])
    predicted = one_hot_log_probs([1, 3, 3, 3, 1])
    return ({"pose": torch.zeros(1, 5, 50, 3), "bio": {"sign": gold, "sentence": gold}},
            {"sign": predicted, "sentence": predicted})


@needs(*UPSTREAM)
def test_validation_also_scores_the_other_decoders(monkeypatch):
    from experiments.validation_metrics import ValidationMetricsModel, _empty
    from metrics import get_decoder, mf1s_from_counts

    monkeypatch.setattr(ValidationMetricsModel, "decoder", get_decoder("b-honouring"))
    model = ValidationMetricsModel(**SMALL)
    batch, log_probs = missing_b_batch()

    collected = _empty()
    model._accumulate(batch, collected, log_probs=log_probs, others=True)
    for level in ("sign", "phrase"):
        # the decoder of the run, as before
        assert collected[level]["iou"] == [0.0]
        # the two others, and not the decoder of the run a second time
        others = collected[level]["others"]
        assert set(others) == {"b-ignoring", "i-runs-kept"}
        for name in others:
            assert others[name]["iou"] == [1.0]
            assert others[name]["percentage"] == [1.0]
            assert mf1s_from_counts(others[name]["counts"]) == 1.0


@needs(*UPSTREAM)
def test_train_batches_score_the_decoder_of_the_run_only():
    from experiments.validation_metrics import ValidationMetricsModel, _empty

    model = ValidationMetricsModel(**SMALL)
    batch, log_probs = missing_b_batch()
    collected = _empty()
    model._accumulate(batch, collected, log_probs=log_probs)
    assert collected["sign"]["others"] == {} and collected["phrase"]["others"] == {}


@needs(*UPSTREAM)
def test_other_decoders_are_logged_under_decoders_with_their_names(monkeypatch):
    from experiments.validation_metrics import ValidationMetricsModel, _empty
    from metrics import get_decoder

    monkeypatch.setattr(ValidationMetricsModel, "decoder", get_decoder("b-ignoring"))
    model = ValidationMetricsModel(**SMALL)
    batch, log_probs = missing_b_batch()
    collected = _empty()
    model._accumulate(batch, collected, log_probs=log_probs, others=True)

    logged = {}
    monkeypatch.setattr(model, "log", lambda name, value, **_: logged.__setitem__(name, value))
    model._log_other_decoders(collected, "validation_dgs_corpus")

    stem = "decoders/validation_dgs_corpus"
    assert set(logged) == {f"{stem}_{level}_{metric}_{name}"
                           for level in ("sign", "phrase")
                           for metric in ("iou", "percentage", "mf1s")
                           for name in ("b-honouring", "i-runs-kept")}
    # b-honouring finds no segment where the B is missing; i-runs-kept finds the gold one
    assert logged[f"{stem}_sign_iou_b-honouring"] == 0.0
    assert logged[f"{stem}_sign_percentage_b-honouring"] == 0.0
    assert logged[f"{stem}_sign_mf1s_b-honouring"] == 0.0
    assert logged[f"{stem}_phrase_iou_i-runs-kept"] == 1.0
    assert logged[f"{stem}_phrase_percentage_i-runs-kept"] == 1.0
    assert logged[f"{stem}_phrase_mf1s_i-runs-kept"] == 1.0


@needs(*UPSTREAM)
def test_a_validation_pass_logs_the_other_decoders_beside_the_metrics_of_the_run():
    """One validation pass of Lightning over one clip written by hand, on a new model."""
    import pytorch_lightning as pl
    import torch
    from experiments.validation_metrics import ValidationMetricsModel
    from torch.utils.data import DataLoader

    frames = 32
    gold = torch.tensor([[1, 2, 3, 3, 1] + [1] * (frames - 5)])   # upstream ids
    batch = {"pose": torch.zeros(1, frames, 50, 3), "bio": {"sign": gold, "sentence": gold}}
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator="cpu",
                         enable_progress_bar=False, enable_model_summary=False)
    logged = trainer.validate(ValidationMetricsModel(**SMALL),
                              dataloaders=DataLoader([batch], batch_size=None),
                              verbose=False)[0]

    # the validation set is called "dev" when train.py has not named it
    others = {name for name in logged if name.startswith("decoders/")}
    assert others == {f"decoders/validation_dev_{level}_{metric}_{name}"
                      for level in ("sign", "phrase")
                      for metric in ("iou", "percentage", "mf1s")
                      for name in ("b-ignoring", "i-runs-kept")}
    # the metrics of the decoder of the run keep their names, the selection metric too
    for name in ("validation_dev_sign_iou", "validation_dev_phrase_mf1s",
                 "validation_sign_percentage", "validation_mean_mf1s"):
        assert name in logged


def test_train_refuses_a_decode_that_is_not_a_decoder(monkeypatch, capsys):
    from experiments import train

    monkeypatch.setattr(sys, "argv", ["train.py", "--decode", "likeliest"])
    with pytest.raises(SystemExit) as stop:
        train.main()
    assert stop.value.code == 2
    assert "b-honouring" in capsys.readouterr().err


# --- training: the warning ---------------------------------------------------------

@pytest.mark.parametrize("weighting, decoder", [
    ("none", "b-ignoring"), ("none", "i-runs-kept"),
    ("inverse", "b-honouring"), ("inverse-b", "b-honouring"), ("1,5,1", "b-honouring"),
    ("inverse", "b-ignoring"), ("2,80,1", "i-runs-kept"),
])
def test_no_warning_with_class_weights_or_with_another_decoder(weighting, decoder):
    from experiments.train import decoder_warning

    assert decoder_warning(weighting, decoder) is None


def test_warning_for_a_run_without_class_weights_that_decodes_with_b_honouring():
    from experiments.train import decoder_warning

    warning = decoder_warning("none", "b-honouring")
    # one line of the log, which says what is the matter and what to pass instead
    assert warning.startswith("warning: ") and "\n" not in warning
    assert "without class weights" in warning and "b-honouring" in warning
    assert "--decode b-ignoring" in warning
