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


# --- reading the decoder back from a model -------------------------------------------

@needs(*UPSTREAM)
@pytest.mark.parametrize("name", ["b-ignoring", "i-runs-kept"])
def test_load_checkpoint_gives_the_decoder_that_the_checkpoint_keeps(name, tmp_path,
                                                                     monkeypatch):
    from benchmark.predict_dgs_2026 import load_checkpoint
    from experiments.validation_metrics import ValidationMetricsModel
    from metrics import get_decoder

    monkeypatch.setattr(ValidationMetricsModel, "decoder", get_decoder(name))
    path = checkpoint_of(ValidationMetricsModel(**SMALL), tmp_path / "best.ckpt")

    assert load_checkpoint(str(path)).stored_decode == name
    # a run folder is a model too: upstream takes its best.ckpt
    assert load_checkpoint(str(tmp_path)).stored_decode == name


@needs(*UPSTREAM)
def test_load_checkpoint_loads_a_checkpoint_that_keeps_no_decoder(tmp_path):
    """Every checkpoint written before the decoder was kept is one of these."""
    import torch
    from benchmark.predict_dgs_2026 import load_checkpoint

    from sign_language_segmentation.model.model import PoseTaggingModel

    model = PoseTaggingModel(**SMALL)                # upstream's class keeps no decoder
    path = checkpoint_of(model, tmp_path / "best.ckpt")
    assert "decode" not in torch.load(path, map_location="cpu",
                                      weights_only=False)["hyper_parameters"]

    loaded = load_checkpoint(str(path))
    assert loaded.stored_decode is None
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key


@needs(*UPSTREAM, "safetensors")
@pytest.mark.parametrize("stored", ["b-ignoring", None])
def test_load_checkpoint_reads_the_config_of_a_safetensors_model(stored, tmp_path):
    """The format of the shipped model: `config.json` beside `model.safetensors`."""
    from benchmark.predict_dgs_2026 import load_checkpoint
    from safetensors.torch import save_file

    from sign_language_segmentation.model.model import PoseTaggingModel

    config = {**SMALL, "pose_dims": list(SMALL["pose_dims"])}
    if stored:
        config["decode"] = stored
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file(PoseTaggingModel(**SMALL).state_dict(), str(tmp_path / "model.safetensors"))

    assert load_checkpoint(str(tmp_path)).stored_decode == stored
    assert load_checkpoint(str(tmp_path / "model.safetensors")).stored_decode == stored


@pytest.mark.parametrize("given, model_given, stored, expected", [
    # the order: --decode of the run, the decoder stored with the model, b-honouring
    (None, True, None, "b-honouring"),
    (None, True, "i-runs-kept", "i-runs-kept"),
    ("b-ignoring", True, "i-runs-kept", "b-ignoring"),
    ("2023", True, "b-ignoring", "b-honouring"),
    # the shipped model, which is what runs without --model, keeps its decoder
    (None, False, None, "b-ignoring"),
    ("b-honouring", False, None, "b-honouring"),
    # MODEL_PATH can put another model in its place: what that one stores counts
    (None, False, "i-runs-kept", "i-runs-kept"),
])
def test_decoder_of_a_prediction_run(given, model_given, stored, expected):
    from benchmark.predict_dgs_2026 import decoder_of_run

    assert decoder_of_run(given, model_given=model_given, stored=stored).name == expected


#: what a model "predicts" in the test below, in upstream ids (UNK=0, O=1, B=2, I=3):
#:   frame        0 1 2 3 4 5 6 7 8
#:   both levels  O I I O B I B I O
PREDICTED_IDS = [1, 3, 3, 1, 2, 3, 2, 3, 1]
PREDICTED_SEGMENTS = {
    "b-honouring": spans((4, 5), (6, 7)),
    "b-ignoring": spans((1, 2), (4, 7)),
    "i-runs-kept": spans((1, 2), (4, 5), (6, 7)),
}


@needs(*UPSTREAM)
@pytest.mark.parametrize("stored, model_given, option, expected", [
    ("i-runs-kept", True, None, "i-runs-kept"),      # the decoder kept with the model
    ("i-runs-kept", True, "upstream", "b-ignoring"),   # --decode first, by its earlier name
    (None, True, None, "b-honouring"),               # a checkpoint that keeps none
    (None, False, None, "b-ignoring"),               # the shipped model keeps its decoder
    (None, False, "b-honouring", "b-honouring"),
])
def test_prediction_file_names_its_decoder_and_holds_its_segments(
        stored, model_given, option, expected, tmp_path, monkeypatch, capsys):
    """`predict_dgs_2026.py` itself, on one clip and with a model written by hand."""
    import argparse

    import numpy as np
    from benchmark import predict_dgs_2026 as predict

    frames = len(PREDICTED_IDS)

    class ModelByHand:
        stored_decode = stored
        hparams = argparse.Namespace(num_frames=1024, pose_dims=(50, 3))

        def __call__(self, pose, timestamps=None):
            return {"sign": one_hot_log_probs(PREDICTED_IDS),
                    "sentence": one_hot_log_probs(PREDICTED_IDS)}

    gloss = {"start_time": 0.02, "end_time": 0.08}
    clip = {"id": "by_hand", "fps": 50.0, "pose": None,
            "sentences": [{**gloss, "glosses": [gloss]}]}
    monkeypatch.setattr(predict, "load_checkpoint", lambda path, device="cpu": ModelByHand())
    monkeypatch.setattr(predict.dgs_data, "iter_clips_native", lambda split: iter([clip]))
    monkeypatch.setattr(predict, "prepare", lambda pose, fps, velocity=True: (
        np.zeros((frames, 50, 3), dtype=np.float32), np.arange(frames, dtype=np.float32) / fps))

    out = tmp_path / "prediction.json"
    monkeypatch.setattr(sys, "argv", [
        "predict_dgs_2026.py", "--split", "validation", "--out", str(out),
        *(["--model", str(tmp_path / "best.ckpt")] if model_given else []),
        *(["--decode", option] if option else [])])
    predict.main()

    payload = json.loads(out.read_text())
    assert payload["decode"] == expected
    assert f"segments via {expected}" in capsys.readouterr().out
    stored_clip = payload["clips"][0]
    for level in ("sign", "phrase"):
        assert stored_clip["pred"][level] == PREDICTED_SEGMENTS[expected]
        # the labels are stored in our ids whatever the decoder: O I I O B I B I O
        assert stored_clip["pred_bio"][level] == rle([O, I, I, O, B, I, B, I, O])


@needs(*UPSTREAM)
def test_the_shipped_model_stores_no_decoder():
    # so the b-ignoring of the shipped model is said in predict_dgs_2026.py, not read
    from benchmark.predict_dgs_2026 import stored_decode

    from sign_language_segmentation.bin import resolve_model_path

    assert stored_decode(resolve_model_path()) is None


# --- eval_youtube.py and the plots: the same order -----------------------------------

class PredictsByHand:
    """Stands for a loaded model: it predicts `PREDICTED_IDS` at the phrase level."""

    def __init__(self, stored_decode=None):
        import argparse

        self.stored_decode = stored_decode
        self.hparams = argparse.Namespace(num_frames=1024, pose_dims=(50, 3))

    def __call__(self, pose, timestamps=None):
        return {"sign": one_hot_log_probs(PREDICTED_IDS),
                "sentence": one_hot_log_probs(PREDICTED_IDS)}


@pytest.fixture
def youtube_clip(tmp_path, monkeypatch):
    """One YouTube dev clip of nine frames at 50 fps whose gold phrase is frames 4 to 5."""
    import types

    import numpy as np
    import pose_format
    from datasets.youtube_sl25 import load as youtube

    from sign_language_segmentation.utils import pose as upstream_pose

    pose_path = tmp_path / "by_hand.pose"
    pose_path.write_bytes(b"")
    spec = {"id": "by_hand", "fps": 50.0, "pose_path": str(pose_path),
            "sentences": [{"start_time": 0.07, "end_time": 0.11}]}
    monkeypatch.setattr(youtube, "clip_specs", lambda split, **options: iter([spec]))
    # the pose is not read: nine empty frames of 50 joints
    empty = types.SimpleNamespace(body=types.SimpleNamespace(
        data=np.ma.zeros((len(PREDICTED_IDS), 1, 50, 3))))
    monkeypatch.setattr(pose_format.Pose, "read", staticmethod(lambda data: empty))
    monkeypatch.setattr(upstream_pose, "preprocess_pose", lambda pose: pose)
    return spec


#: against the gold phrase 4-5: predicted segments, then IoU and `%`
#:   b-honouring  4-5, 6-7        2/4   2
#:   b-ignoring   1-2, 4-7        2/6   2
#:   i-runs-kept  1-2, 4-5, 6-7   2/6   3
YOUTUBE_SCORES = {"b-honouring": (2 / 4, 2.0), "b-ignoring": (2 / 6, 2.0),
                  "i-runs-kept": (2 / 6, 3.0)}


@needs(*UPSTREAM)
@pytest.mark.parametrize("stored, given, expected", [
    (None, None, "b-honouring"),
    ("i-runs-kept", None, "i-runs-kept"),
    ("i-runs-kept", "upstream", "b-ignoring"),
    (None, "i-runs-kept", "i-runs-kept"),
])
def test_eval_youtube_scores_with_the_decoder_of_the_order(stored, given, expected,
                                                           youtube_clip, tmp_path, monkeypatch):
    from benchmark import predict_dgs_2026
    from experiments import eval_youtube

    monkeypatch.setattr(predict_dgs_2026, "load_checkpoint",
                        lambda path, device="cpu": PredictsByHand(stored))
    result = eval_youtube.score(tmp_path / "best.ckpt", dev_filter=False, device="cpu",
                                decode=given)
    assert result["decode"] == expected
    assert (result["iou"], result["percentage"]) == (
        pytest.approx(YOUTUBE_SCORES[expected][0]), YOUTUBE_SCORES[expected][1])
    assert result["clips"] == 1


def in_seconds(segments, fps=50.0) -> list[float]:
    """Start and end of each segment in seconds, one after the other."""
    return [bound for segment in segments
            for bound in (segment["start"] / fps, (segment["end"] + 1) / fps)]


def bounds(drawn) -> list[float]:
    """The spans of a plot record, `(start, end)` in seconds, in the same flat form."""
    return [bound for span in drawn for bound in span]


@needs(*UPSTREAM)
@pytest.mark.parametrize("stored, given, expected", [
    (None, None, "b-honouring"),
    ("i-runs-kept", None, "i-runs-kept"),
    ("b-ignoring", "2023", "b-honouring"),
])
def test_plot_draws_the_segments_of_the_decoder_of_the_order(stored, given, expected,
                                                             youtube_clip, tmp_path, monkeypatch):
    from experiments import plot_segmentation

    monkeypatch.setattr(plot_segmentation, "load_checkpoint",
                        lambda path, device="cpu": PredictsByHand(stored))
    records = plot_segmentation.collect(tmp_path / "best.ckpt", "youtube_25", clips=1,
                                        device="cpu", quiet=True, decode=given)
    assert [record["decode"] for record in records] == [expected]
    assert bounds(records[0]["pred"]["phrase"]) == pytest.approx(
        in_seconds(PREDICTED_SEGMENTS[expected]))
    # the row beside it stays the b-ignoring one, whatever the decoder of the row "ours"
    assert bounds(records[0]["pred_upstream"]["phrase"]) == pytest.approx(
        in_seconds(PREDICTED_SEGMENTS["b-ignoring"]))


@needs(*UPSTREAM)
def test_plot_during_training_takes_the_decoder_of_the_model(youtube_clip):
    """The callback gives `collect` the model that is training, not a checkpoint."""
    from experiments import plot_segmentation
    from metrics import get_decoder

    model = PredictsByHand()
    model.decoder = get_decoder("i-runs-kept")       # as ValidationMetricsModel has it
    records = plot_segmentation.collect(model, "youtube_25", clips=1, device="cpu",
                                        quiet=True)
    assert records[0]["decode"] == "i-runs-kept"
    assert bounds(records[0]["pred"]["phrase"]) == pytest.approx(
        in_seconds(PREDICTED_SEGMENTS["i-runs-kept"]))


def test_caption_of_the_row_ours_names_its_decoder():
    from experiments.plot_segmentation import caption_of

    for name in ("b-honouring", "b-ignoring", "i-runs-kept"):
        assert name in caption_of("pred", [{"decode": name}])
    # a cache of predictions written before the decoder was recorded: b-honouring then
    assert "b-honouring" in caption_of("pred", [{}])
    # the other rows do not depend on it
    assert caption_of("gold", [{"decode": "b-ignoring"}]) == caption_of("gold", [{}])


@pytest.mark.parametrize("script", ["train", "eval_youtube", "plot_segmentation"])
def test_every_script_takes_the_earlier_names_too(script, monkeypatch, capsys):
    """`--decode 2023` and `--decode upstream` are read wherever `--decode` exists."""
    import importlib

    module = importlib.import_module(f"experiments.{script}")
    # a name that is no decoder stops the script, and the message lists what it takes
    monkeypatch.setattr(sys, "argv", [f"{script}.py", "--run", "none", "--decode", "likeliest"])
    with pytest.raises(SystemExit):
        module.main()
    message = capsys.readouterr().err
    for name in ("b-honouring", "b-ignoring", "i-runs-kept", "2023", "upstream"):
        assert name in message
