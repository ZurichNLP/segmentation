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


# --- redecode.py: write the segments of a file again ---------------------------------

#: the segments each decoder gives from the labels of `CLIP`, sign then phrase
CLIP_SEGMENTS = {
    "b-honouring": {"sign": spans((5, 6)), "phrase": spans((1, 3), (4, 6))},
    "b-ignoring": {"sign": spans((1, 3), (5, 6)), "phrase": spans((1, 6))},
    "i-runs-kept": {"sign": spans((1, 3), (5, 6)), "phrase": spans((1, 3), (4, 6))},
}


FILE_BY_HAND = {"dataset": "by_hand", "split": "validation", "model": "model by hand",
                "thresholds": {}}


@pytest.fixture
def decoded_file(tmp_path):
    """A file as predict_dgs_2026.py writes it: its segments are its labels, decoded.

    It names `2023`, the earlier name of b-honouring, and holds the segments of
    that decoder. redecode.py takes such a file and no other.
    """
    path = tmp_path / "decoded.json"
    clip = {**CLIP, "pred": CLIP_SEGMENTS["b-honouring"]}
    path.write_text(json.dumps({**FILE_BY_HAND, "decode": "2023", "clips": [clip]}, indent=2))
    return path


@pytest.fixture
def thresholds_file(tmp_path):
    """A file as predict_dgs_2023.py writes it.

    Its segments come from thresholds on the probabilities and its labels from
    the argmax, it has `thresholds` and no `decode`. Its segments cannot be made
    again from its labels, and such files are not in git.
    """
    path = tmp_path / "dgs_test_2023_by_hand.json"
    path.write_text(json.dumps({
        "dataset": "by_hand", "split": "test", "model": "2023 by hand",
        "checkpoint": "model_E4s-1.pth",
        "thresholds": {"sign": [60, 50], "phrase": [90, 90]},
        "pipeline": "v2023 original (vendored)", "clips": [CLIP]}, indent=2))
    return path


def run_redecode(monkeypatch, capsys, *arguments) -> str:
    from benchmark import redecode

    monkeypatch.setattr(sys, "argv", ["redecode.py", *[str(argument) for argument in arguments]])
    redecode.main()
    return capsys.readouterr().out


def without_segments(payload: dict) -> dict:
    """The file without the two things that redecode.py may change."""
    rest = {key: value for key, value in payload.items() if key not in ("decode", "clips")}
    rest["clips"] = [{key: value for key, value in clip.items() if key != "pred"}
                     for clip in payload["clips"]]
    return rest


@pytest.mark.parametrize("option, name", [("b-ignoring", "b-ignoring"),
                                          ("b-honouring", "b-honouring"),
                                          ("i-runs-kept", "i-runs-kept"),
                                          ("upstream", "b-ignoring")])
def test_redecode_writes_the_segments_and_the_name_and_nothing_else(
        decoded_file, option, name, monkeypatch, capsys):
    before = json.loads(decoded_file.read_text())
    run_redecode(monkeypatch, capsys, decoded_file, "--decode", option)

    text = decoded_file.read_text()
    after = json.loads(text)
    assert after["decode"] == name
    assert after["clips"][0]["pred"] == CLIP_SEGMENTS[name]
    # the labels, the gold and every other field are as they were, in the same order
    assert without_segments(after) == without_segments(before)
    assert list(after) == list(before)
    assert list(after["clips"][0]) == list(before["clips"][0])
    # and the file is laid out as predict_dgs_2026.py writes it
    assert text == json.dumps(after, indent=2)


def test_redecoded_file_scores_like_score_with_decode(decoded_file, monkeypatch, capsys):
    from benchmark.score import score_file
    from metrics import get_decoder

    expected = score_file(decoded_file, decoder=get_decoder("b-ignoring"))["levels"]
    run_redecode(monkeypatch, capsys, decoded_file, "--decode", "b-ignoring")
    # with no option now: the file alone says how it is scored
    assert score_file(decoded_file)["levels"] == expected


def test_redecode_says_what_changed_and_a_second_time_changes_nothing(decoded_file,
                                                                      monkeypatch, capsys):
    first = run_redecode(monkeypatch, capsys, decoded_file, "--decode", "b-ignoring")
    assert "2023 -> b-ignoring" in first and "2 of 2" in first
    written = decoded_file.read_bytes()

    second = run_redecode(monkeypatch, capsys, decoded_file, "--decode", "b-ignoring")
    assert "b-ignoring -> b-ignoring" in second and "0 of 2" in second
    assert decoded_file.read_bytes() == written


def test_redecode_dry_run_writes_nothing(decoded_file, monkeypatch, capsys):
    before = decoded_file.read_bytes()
    out = run_redecode(monkeypatch, capsys, decoded_file, "--decode", "b-ignoring",
                       "--dry-run")
    assert "2023 -> b-ignoring" in out and "2 of 2" in out and "not written" in out
    assert decoded_file.read_bytes() == before


def test_redecode_adds_the_name_to_a_file_that_has_none(decoded_file, monkeypatch, capsys):
    # the files of September 2026: no `decode`, segments from upstream's function
    payload = json.loads(decoded_file.read_text())
    del payload["decode"]
    payload["clips"][0]["pred"] = CLIP_SEGMENTS["b-ignoring"]
    decoded_file.write_text(json.dumps(payload, indent=2))

    out = run_redecode(monkeypatch, capsys, decoded_file, "--decode", "i-runs-kept")
    after = json.loads(decoded_file.read_text())
    assert "none -> i-runs-kept" in out
    assert after["decode"] == "i-runs-kept"
    # among the fields of the file, before the clips, where the predict script puts it
    assert list(after)[-2:] == ["decode", "clips"]


def test_redecode_refuses_a_file_it_would_lay_out_anew(decoded_file, monkeypatch, capsys):
    # written without indent: writing it back would change every line of it
    compact = json.dumps(json.loads(decoded_file.read_text()))
    decoded_file.write_text(compact)

    with pytest.raises(SystemExit) as stop:
        run_redecode(monkeypatch, capsys, decoded_file, "--decode", "b-ignoring")
    assert "laid out" in str(stop.value)
    assert "score.py" in str(stop.value) and "--decode" in str(stop.value)
    assert decoded_file.read_text() == compact


def test_redecode_refuses_a_file_without_frame_labels(decoded_file, monkeypatch, capsys):
    payload = json.loads(decoded_file.read_text())
    del payload["clips"][0]["pred_bio"]
    decoded_file.write_text(json.dumps(payload, indent=2))
    before = decoded_file.read_bytes()

    with pytest.raises(SystemExit) as stop:
        run_redecode(monkeypatch, capsys, decoded_file, "--decode", "b-ignoring")
    assert "no frame labels" in str(stop.value)
    assert "score.py" not in str(stop.value)       # it has nothing to decode either
    assert decoded_file.read_bytes() == before


@pytest.mark.parametrize("options", [[], ["--dry-run"]])
def test_redecode_refuses_a_file_whose_segments_are_not_its_labels_decoded(
        prediction_file, options, monkeypatch, capsys):
    """Writing such a file again would lose segments that nothing can make again."""
    # this file names 2023, and holds segments that b-honouring does not give
    before = prediction_file.read_bytes()
    with pytest.raises(SystemExit) as stop:
        run_redecode(monkeypatch, capsys, prediction_file, "--decode", "b-ignoring", *options)
    message = str(stop.value)
    assert str(prediction_file) in message
    # the decoder the file names, and the first clip and level where it does not hold
    assert "b-honouring" in message and "'2023'" in message
    assert "clip by_hand, sign" in message
    assert "Nothing was written" in message
    # and the way to see its scores under another decoder without changing it
    assert f"python benchmark/score.py {prediction_file} --decode" in message
    assert prediction_file.read_bytes() == before


def test_redecode_reads_a_file_without_a_name_as_upstream(decoded_file, monkeypatch, capsys):
    # no `decode`, but the segments of b-honouring: not what upstream's function gives
    payload = json.loads(decoded_file.read_text())
    del payload["decode"]
    decoded_file.write_text(json.dumps(payload, indent=2))
    before = decoded_file.read_bytes()

    with pytest.raises(SystemExit) as stop:
        run_redecode(monkeypatch, capsys, decoded_file, "--decode", "i-runs-kept")
    assert "b-ignoring" in str(stop.value) and "names no decoder" in str(stop.value)
    assert decoded_file.read_bytes() == before


@pytest.mark.parametrize("options", [[], ["--dry-run"]])
def test_redecode_refuses_a_file_of_a_threshold_model(thresholds_file, options,
                                                      monkeypatch, capsys):
    before = thresholds_file.read_bytes()
    with pytest.raises(SystemExit) as stop:
        run_redecode(monkeypatch, capsys, thresholds_file, "--decode", "b-ignoring", *options)
    message = str(stop.value)
    assert str(thresholds_file) in message and "thresholds" in message
    assert "Nothing was written" in message
    # score.py --decode refuses such a file too, so it is not offered
    assert "score.py" not in message
    assert thresholds_file.read_bytes() == before


def test_redecode_refuses_thresholds_also_when_the_segments_fit_a_decoder(thresholds_file,
                                                                          monkeypatch, capsys):
    # by chance the segments equal the labels decoded with upstream's rule: the
    # file still says that they came from thresholds
    payload = json.loads(thresholds_file.read_text())
    payload["clips"][0]["pred"] = CLIP_SEGMENTS["b-ignoring"]
    thresholds_file.write_text(json.dumps(payload, indent=2))
    before = thresholds_file.read_bytes()

    with pytest.raises(SystemExit) as stop:
        run_redecode(monkeypatch, capsys, thresholds_file, "--decode", "b-honouring")
    assert "thresholds" in str(stop.value)
    assert thresholds_file.read_bytes() == before


@pytest.mark.parametrize("order", ["good file first", "refused file first"])
def test_redecode_writes_no_file_when_one_of_them_is_refused(decoded_file, thresholds_file,
                                                             order, monkeypatch, capsys):
    files = [decoded_file, thresholds_file]
    if order == "refused file first":
        files.reverse()
    before = [path.read_bytes() for path in files]

    with pytest.raises(SystemExit) as stop:
        run_redecode(monkeypatch, capsys, *files, "--decode", "b-ignoring")
    assert str(thresholds_file) in str(stop.value)
    assert [path.read_bytes() for path in files] == before


def test_redecode_help_is_its_docstring(monkeypatch, capsys):
    """The docstring is written for a person who has never seen the script."""
    from benchmark import redecode

    monkeypatch.setattr(sys, "argv", ["redecode.py", "--help"])
    with pytest.raises(SystemExit) as stop:
        redecode.main()
    assert stop.value.code == 0
    shown = capsys.readouterr().out
    for heading in ("What a prediction file holds", "What this script does", "Why it exists",
                    "Use it when", "Do not use it when", "How it protects a file"):
        assert heading in redecode.__doc__ and heading in shown
    # every line of the docstring is in the help, as it is written
    for line in redecode.__doc__.strip().splitlines():
        assert line.rstrip() in shown


# --- score.py --decode and a file of a threshold model ------------------------------

def test_score_with_decode_refuses_a_file_with_thresholds(thresholds_file, prediction_file,
                                                          monkeypatch, capsys):
    """Its labels are the argmax, and its segments came from thresholds on probabilities.

    Decoding the labels would score another model than the one the file is the row of.
    """
    from benchmark.score import score_file
    from metrics import get_decoder

    with pytest.raises(SystemExit) as stop:
        score_file(thresholds_file, decoder=get_decoder("b-ignoring"))
    assert str(thresholds_file) in str(stop.value) and "thresholds" in str(stop.value)

    # from the command line too, also behind a file that can be decoded again
    with pytest.raises(SystemExit) as stop:
        run_score(monkeypatch, capsys, prediction_file, thresholds_file, "--decode", "upstream")
    assert str(thresholds_file) in str(stop.value)
    assert "model by hand" not in capsys.readouterr().out      # no table half printed


def test_score_without_decode_scores_a_file_with_thresholds_as_before(thresholds_file,
                                                                      monkeypatch, capsys):
    out = run_score(monkeypatch, capsys, thresholds_file)
    row = next(line for line in out.splitlines() if line.startswith("2023 by hand"))
    assert row.startswith("2023 by hand (60/50, 90/90)")
    assert numbers(row)[2:4] == ["0.625", "0.500"]          # the stored segments


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


@needs(*UPSTREAM)
@pytest.mark.parametrize("option, selection", [
    ("b-honouring", 0.0),      # no B, so no segment: nothing matches the gold one
    ("b-ignoring", 1.0),       # the run of I is the gold segment
    ("i-runs-kept", 1.0),
    ("2023", 0.0),             # the earlier names
    ("upstream", 1.0),
])
def test_selection_metric_follows_the_decoder_of_the_run(option, selection, monkeypatch):
    """From `--decode` to `validation_mean_mf1s`, the metric that keeps a checkpoint.

    Gold `O B I I O`, predicted `O I I I O`, at both levels. `train.py` gives
    the run its decoder with `set_decoder`; then one validation pass of
    Lightning, in which the model predicts those labels whatever its weights.
    """
    import argparse

    import pytorch_lightning as pl
    from experiments import validation_metrics
    from experiments.train import set_decoder
    from experiments.validation_metrics import ValidationMetricsModel
    from torch.utils.data import DataLoader

    # set_decoder sets the decoder of the class: it is put back after the test
    monkeypatch.setattr(ValidationMetricsModel, "decoder", ValidationMetricsModel.decoder)
    set_decoder(argparse.Namespace(), option)

    batch, log_probs = missing_b_batch()
    monkeypatch.setattr(validation_metrics, "masked_forward",
                        lambda model, pose, timestamps=None, lengths=None: log_probs)
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator="cpu",
                         enable_progress_bar=False, enable_model_summary=False)
    logged = trainer.validate(ValidationMetricsModel(**SMALL),
                              dataloaders=DataLoader([batch], batch_size=None),
                              verbose=False)[0]

    # what ModelCheckpoint monitors, and the same under the name of the set
    assert logged["validation_mean_mf1s"] == selection
    assert logged["validation_dev_mean_mf1s"] == selection
    for level in ("sign", "phrase"):
        assert logged[f"validation_{level}_mf1s"] == selection
        # frame F1 is the same for every decoder: four of five frames are right
        assert logged[f"validation_{level}_frame_f1_micro"] == pytest.approx(4 / 5)


@needs("numpy", "sklearn")       # the options of train.py import metrics
def test_train_refuses_a_decode_that_is_not_a_decoder(monkeypatch, capsys):
    from experiments import train

    monkeypatch.setattr(sys, "argv", ["train.py", "--decode", "likeliest"])
    with pytest.raises(SystemExit) as stop:
        train.main()
    assert stop.value.code == 2
    assert "b-honouring" in capsys.readouterr().err


@needs(*UPSTREAM)
@pytest.mark.parametrize("given, expected", [
    (None, "b-honouring"),            # --decode not given: the default
    ("b-honouring", "b-honouring"),
    ("b-ignoring", "b-ignoring"),
    ("i-runs-kept", "i-runs-kept"),
    ("2023", "b-honouring"),          # the earlier names
    ("upstream", "b-ignoring"),
])
def test_set_decoder_gives_the_run_its_decoder_in_three_places(given, expected, monkeypatch):
    """What `train.py` does with `--decode`: the model class, a model built after, and `args`."""
    import argparse

    from experiments.train import set_decoder
    from experiments.validation_metrics import ValidationMetricsModel
    from metrics import get_decoder

    # another decoder than the expected one to start from, so that each case is
    # seen to set it; monkeypatch puts the decoder of the class back afterwards
    other = "b-ignoring" if expected == "i-runs-kept" else "i-runs-kept"
    monkeypatch.setattr(ValidationMetricsModel, "decoder", get_decoder(other))
    args = argparse.Namespace()

    decoder = set_decoder(args, given)

    assert decoder.name == expected
    # validation decodes with the decoder of the class
    assert ValidationMetricsModel.decoder is decoder
    # a model built now keeps the name, and so will its checkpoints
    assert ValidationMetricsModel(**SMALL).hparams["decode"] == expected
    # and run_config.json and W&B get it from here
    assert args.decode == expected


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A folder that stands for `segment-any-sign/`, as the working directory."""
    folder = tmp_path / "project"
    folder.mkdir()
    monkeypatch.chdir(folder)
    return folder


def start_main(monkeypatch, *arguments) -> None:
    """`python experiments/train.py ARGUMENTS` in this process, with no data read.

    `main()` sets names of the upstream package and of the model class for the
    run it starts. They are put back after the test, so that no other test
    meets them. The two functions that read the corpus are replaced.
    """
    # upstream's args.py parses the command line when it is first imported, and
    # the import of its training module does that: not with the line of pytest
    monkeypatch.setattr(sys, "argv", ["train.py"])
    from experiments import train
    from experiments.validation_metrics import ValidationMetricsModel

    import sign_language_segmentation.train as upstream_train
    from sign_language_segmentation.datasets import common

    for name in ("PoseTaggingModel", "get_dataloader", "pl"):
        monkeypatch.setattr(upstream_train, name, getattr(upstream_train, name))
    monkeypatch.setattr(common, "create_bio", common.create_bio)
    monkeypatch.setattr(common, "open", open, raising=False)
    for name in ("val_dataset_names", "levels", "accumulate", "grad_log_every",
                 "metrics_every_n_steps", "class_weights", "decoder"):
        monkeypatch.setattr(ValidationMetricsModel, name, getattr(ValidationMetricsModel, name))

    # the run folder is not what these tests are about: no link `dist` is made
    monkeypatch.setattr(train, "link_dist", lambda project=None: None)
    monkeypatch.setattr(train, "dataset_size", lambda args: 4)
    monkeypatch.setattr(train, "inverse_class_weights",
                        lambda args, levels, **options: {level: [0.0, 1.0, 9.0, 1.0]
                                                         for level in levels})
    monkeypatch.setattr(sys, "argv", ["train.py", "--no_wandb", "--device", "cpu",
                                      "--datasets", "dgs_corpus", "--limit", "2",
                                      *arguments])
    # and now it must parse the line of this test, so it is imported anew
    monkeypatch.delitem(sys.modules, "sign_language_segmentation.args", raising=False)
    train.main()


class Reached(Exception):
    """Raised in place of the next step of `main()`, to stop it there."""


@needs(*UPSTREAM)
@pytest.mark.parametrize("options, weighting, decoder", [
    ([], "inverse", "b-honouring"),                           # auto with dice off
    (["--dice_loss_weight", "1.5"], "none", "b-honouring"),   # auto with dice on: no weights
    (["--class-weights", "none"], "none", "b-honouring"),
    (["--class-weights", "none", "--decode", "upstream"], "none", "b-ignoring"),
    (["--class-weights", "inverse-b", "--decode", "2023"], "inverse-b", "b-honouring"),
    (["--class-weights", "1,5,1", "--decode", "i-runs-kept"], "1,5,1", "i-runs-kept"),
])
def test_main_asks_for_the_warning_with_the_resolved_class_weighting(
        options, weighting, decoder, project, monkeypatch):
    """`--class-weights auto` is resolved before the warning is asked for."""
    from experiments import train
    from experiments.validation_metrics import ValidationMetricsModel

    def asked(*arguments):
        # what the run is in at this point, beside what the warning is asked with
        raise Reached(arguments, ValidationMetricsModel.decoder.name)

    monkeypatch.setattr(train, "decoder_warning", asked)
    with pytest.raises(Reached) as reached:
        start_main(monkeypatch, *options)
    arguments, decoder_of_the_class = reached.value.args
    assert arguments == (weighting, decoder)
    assert decoder_of_the_class == decoder


@needs(*UPSTREAM)
@pytest.mark.parametrize("options, warnings", [
    (["--class-weights", "none"], 1),
    (["--dice_loss_weight", "1.5"], 1),
    ([], 0),
    (["--class-weights", "none", "--decode", "b-ignoring"], 0),
])
def test_main_prints_the_warning_once_and_goes_on(options, warnings, project,
                                                  monkeypatch, capsys):
    from experiments import train

    def next_step(args):
        raise Reached

    # with --dry-run the step after the warning is the dry run itself
    monkeypatch.setattr(train, "dry_run", next_step)
    with pytest.raises(Reached):
        start_main(monkeypatch, "--dry-run", *options)
    printed = capsys.readouterr().out
    assert printed.count("warning: this run trains without class weights") == warnings


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


#: What a model "predicts" in the tests below, in upstream ids (UNK=0, O=1, B=2, I=3).
#: Two frames are UNK, which a model can give and which every script turns
#: into O before it decodes: so that step is run in each of them.
#:   frame        0 1 2 3 4 5 6 7 8
#:   upstream     U I I U B I B I O
#:   in our ids   O I I O B I B I O
PREDICTED_IDS = [0, 3, 3, 0, 2, 3, 2, 3, 1]
PREDICTED_SEGMENTS = {
    "b-honouring": spans((4, 5), (6, 7)),
    "b-ignoring": spans((1, 2), (4, 7)),
    "i-runs-kept": spans((1, 2), (4, 5), (6, 7)),
}


def run_predict(monkeypatch, tmp_path, stored, model_given, option) -> dict:
    """`predict_dgs_2026.py` itself, on one clip and with a model written by hand.

    `stored` is the decoder the model keeps, or None. Returns the prediction file.
    """
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
    return json.loads(out.read_text())


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
    payload = run_predict(monkeypatch, tmp_path, stored, model_given, option)
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
    # the b-ignoring segments, which are counted in the printed summary and not
    # drawn, are the same whatever the decoder of the row "ours"
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


@pytest.mark.parametrize("given, cached, said", [
    (None, {"decode": "b-ignoring"}, False),            # no --decode: nothing was asked
    ("b-ignoring", {"decode": "b-ignoring"}, False),
    ("upstream", {"decode": "b-ignoring"}, False),      # the same decoder by its earlier name
    ("i-runs-kept", {"decode": "b-ignoring"}, True),
    ("2023", {}, False),                                # a cache from before: b-honouring
    ("b-ignoring", {}, True),
])
def test_reuse_note_only_when_decode_asks_for_another_decoder_than_the_cache(given, cached, said):
    from experiments.plot_segmentation import reuse_note
    from metrics import get_decoder

    note = reuse_note(given, [cached])
    if not said:
        assert note is None
        return
    # one line that names what was asked, what the cached data has, and what to do
    assert "\n" not in note
    assert f"--decode {get_decoder(given).name}" in note
    assert cached.get("decode", "b-honouring") in note
    assert "without --reuse" in note


@needs("matplotlib")
@pytest.mark.parametrize("options, notes", [
    (["--decode", "i-runs-kept"], 1),
    (["--decode", "2023"], 0),        # what the cached data was decoded with
    ([], 0),
])
def test_plot_with_reuse_says_when_decode_is_not_what_the_cache_holds(options, notes,
                                                                      monkeypatch, capsys):
    """`plot_segmentation.py --reuse` itself, on figure data written by hand."""
    import paths
    from experiments import plot_segmentation

    run = "by_hand-2026.11.25"
    run_dir = paths.output("dist") / run
    run_dir.mkdir(parents=True)
    cache = run_dir / "segments_dgs_corpus_phrase_120s.json"
    cache.write_text(json.dumps([{
        "id": "by_hand", "fps": 50.0, "frames": 9, "units": "seconds", "pose_path": "none",
        "decode": "b-honouring",
        "gold": {"sign": [], "phrase": [[0.08, 0.12]]},
        "pred": {"sign": [], "phrase": [[0.08, 0.12], [0.12, 0.16]]},
        "pred_upstream": {"sign": [], "phrase": [[0.02, 0.06], [0.08, 0.16]]}}]))
    before = cache.read_bytes()

    monkeypatch.setattr(sys, "argv", ["plot_segmentation.py", "--run", run, "--datasets",
                                      "dgs_corpus", "--reuse", "--zoom-clips", "0", *options])
    plot_segmentation.main()

    printed = capsys.readouterr().out
    assert "reusing 1 cached predictions" in printed
    assert printed.count("is not applied") == notes
    # the figure is drawn from the cached data either way, and the data is left alone
    assert (run_dir / "segments_dgs_corpus_phrase_120s.png").exists()
    assert cache.read_bytes() == before


# --- the earlier names, 2023 and upstream, in each script that takes --decode -----------

def run_folder(run: str):
    """An empty run folder with a `best.ckpt`, for a script that takes `--run`."""
    import paths

    folder = paths.output("dist") / run
    folder.mkdir(parents=True)
    (folder / "best.ckpt").write_bytes(b"")
    return folder


def used_by_score(option, request, monkeypatch, capsys) -> str:
    out = run_score(monkeypatch, capsys, request.getfixturevalue("prediction_file"),
                    "--decode", option)
    row = next(line for line in out.splitlines() if line.startswith("model by hand"))
    name = row.split("[")[1].split("]")[0]
    # the row is the one of that decoder, not only named after it
    assert numbers(row)[2:4] == [f"{value:.3f}" for value in DECODED[name]["sign"]]
    return name


def used_by_redecode(option, request, monkeypatch, capsys) -> str:
    path = request.getfixturevalue("decoded_file")
    run_redecode(monkeypatch, capsys, path, "--decode", option)
    payload = json.loads(path.read_text())
    assert payload["clips"][0]["pred"] == CLIP_SEGMENTS[payload["decode"]]
    return payload["decode"]


def used_by_predict(option, request, monkeypatch, capsys) -> str:
    payload = run_predict(monkeypatch, request.getfixturevalue("tmp_path"), stored=None,
                          model_given=True, option=option)
    assert payload["clips"][0]["pred"]["sign"] == PREDICTED_SEGMENTS[payload["decode"]]
    return payload["decode"]


def used_by_eval_youtube(option, request, monkeypatch, capsys) -> str:
    from benchmark import predict_dgs_2026
    from experiments import eval_youtube

    request.getfixturevalue("youtube_clip")
    run_folder("by_hand-2026.11.25")
    monkeypatch.setattr(predict_dgs_2026, "load_checkpoint",
                        lambda path, device="cpu": PredictsByHand())
    scores = request.getfixturevalue("tmp_path") / "youtube_dev.json"
    monkeypatch.setattr(sys, "argv", ["eval_youtube.py", "--run", "by_hand-2026.11.25", "--set",
                                      "raw", "--decode", option, "--json", str(scores)])
    eval_youtube.main()
    result = json.loads(scores.read_text())["raw"]
    assert (result["iou"], result["percentage"]) == (
        pytest.approx(YOUTUBE_SCORES[result["decode"]][0]), YOUTUBE_SCORES[result["decode"]][1])
    assert f"decoder {result['decode']}" in capsys.readouterr().out
    return result["decode"]


def used_by_plot_segmentation(option, request, monkeypatch, capsys) -> str:
    from experiments import plot_segmentation

    request.getfixturevalue("youtube_clip")
    folder = run_folder("by_hand-2026.11.25")
    monkeypatch.setattr(plot_segmentation, "load_checkpoint",
                        lambda path, device="cpu": PredictsByHand())
    monkeypatch.setattr(sys, "argv", ["plot_segmentation.py", "--run", "by_hand-2026.11.25",
                                      "--datasets", "youtube_25", "--clips", "1", "--zoom-clips",
                                      "0", "--device", "cpu", "--decode", option])
    plot_segmentation.main()
    # the figure is written, and beside it the data it was drawn from
    assert (folder / "segments_youtube_25_phrase_120s.png").exists()
    record = json.loads((folder / "segments_youtube_25_phrase_120s.json").read_text())[0]
    assert bounds(record["pred"]["phrase"]) == pytest.approx(
        in_seconds(PREDICTED_SEGMENTS[record["decode"]]))
    return record["decode"]


def used_by_train(option, request, monkeypatch, capsys) -> str:
    from experiments import train

    request.getfixturevalue("project")

    def asked(weighting, decoder):
        raise Reached(decoder)

    monkeypatch.setattr(train, "decoder_warning", asked)
    with pytest.raises(Reached) as reached:
        start_main(monkeypatch, "--decode", option)
    return reached.value.args[0]


USED_BY = {"score": used_by_score, "redecode": used_by_redecode,
           "predict_dgs_2026": used_by_predict, "eval_youtube": used_by_eval_youtube,
           "plot_segmentation": used_by_plot_segmentation, "train": used_by_train}


@needs(*UPSTREAM, "matplotlib")
@pytest.mark.parametrize("script", list(USED_BY))
@pytest.mark.parametrize("option, expected", [("2023", "b-honouring"),
                                              ("upstream", "b-ignoring")])
def test_every_script_runs_with_the_earlier_names(script, option, expected, request,
                                                  monkeypatch, capsys):
    """`--decode 2023` and `--decode upstream` in each script that takes `--decode`.

    Each script is run, on inputs written by hand, and what it wrote or printed
    says which decoder it used: the one of today that the earlier name stands for.
    """
    assert USED_BY[script](option, request, monkeypatch, capsys) == expected


# --- only metrics/ names the two decoding functions ---------------------------------------

#: The two functions that turn argmax labels into segments. Only `metrics/` may
#: name them in code: every other file takes a decoder from `metrics.decoders`.
#: The 2023 function with thresholds, `probs_to_segments`, is another matter: it
#: reads probabilities, and the scripts of the 2023 model call it.
DECODING_FUNCTIONS = {"bio_to_segments", "likeliest_probs_to_segments"}


def decoding_functions_named(source: str) -> list[str]:
    """`line: name` for each place where the code of `source` names such a function.

    An import, a call or any other use counts. A comment or a docstring that
    mentions the name does not: only the code is read.
    """
    import ast

    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.alias):
            name = node.name.rsplit(".", 1)[-1]
        else:
            continue
        if name in DECODING_FUNCTIONS:
            found.append(f"{node.lineno}: {name}")
    return sorted(found)


@pytest.mark.parametrize("source", [
    "from metrics import bio_to_segments\n",
    "segments = bio_to_segments(labels, b=2, i=3)\n",
    ("import sign_language_segmentation.metrics as upstream\n"
     "segments = upstream.likeliest_probs_to_segments(probs)\n"),
    "from sign_language_segmentation.metrics import likeliest_probs_to_segments as decode\n",
    "decode = metrics.bio_to_segments\n",
])
def test_the_scan_finds_a_decoding_function_in_code(source):
    assert decoding_functions_named(source)


@pytest.mark.parametrize("source", [
    '"""Segments come from bio_to_segments."""\n',
    "# not likeliest_probs_to_segments: that one never looks at B\nx = 1\n",
    "from metrics import get_decoder\nsegments = get_decoder(name).decode(labels)\n",
    "segments = probs_to_segments(probs, 60, 50)\n",
])
def test_the_scan_reads_code_only_and_these_two_functions_only(source):
    assert decoding_functions_named(source) == []


def test_no_file_outside_metrics_names_a_decoding_function():
    """Outside `metrics/` and the tests, no file of the project names a decoding function.

    That is what is checked: the two names, `bio_to_segments` and
    `likeliest_probs_to_segments`, in the code of each file. It keeps a script
    from calling one of them in place of a decoder of `metrics.decoders`. A
    script that wrote a decoding rule of its own, under another name, would not
    be found.
    """
    from tests.project_files import ROOT, project_files

    scanned = [name for name in project_files(".py")
               if not name.startswith(("metrics/", "tests/"))]
    assert scanned, "no file was read"
    offenders = {name: found for name in scanned
                 if (found := decoding_functions_named((ROOT / name).read_text()))}
    assert offenders == {}


# --- add_result_row.py: the row says which decoder, when it is not the usual one ----

SCORES = {"frame_f1": 0.5, "iou": 0.8, "percentage": 1.1, "mf1s": 0.3, "clips": 3, "skipped": 0}


@pytest.mark.parametrize("youtube, dgs, note", [
    # the usual case says nothing, also for scores written before the decoder was named
    ({"raw": {"decode": "b-honouring"}, "filtered": {"decode": "b-honouring"}}, "2023", ""),
    ({"raw": {}, "filtered": {}}, "b-honouring", ""),
    ({"raw": {}}, None, ""),
    # another decoder is named, by its name of today
    ({"raw": {"decode": "i-runs-kept"}, "filtered": {"decode": "i-runs-kept"}}, "i-runs-kept",
     "decoder `i-runs-kept`"),
    ({"raw": {"decode": "b-ignoring"}}, "upstream", "decoder `b-ignoring`"),
    ({"raw": {"decode": "b-ignoring"}}, None, "decoder `b-ignoring`"),
    # and when the two parts of the row were not decoded alike, each is named
    ({"raw": {"decode": "b-honouring"}, "filtered": {"decode": "b-honouring"}}, "b-ignoring",
     "decoder `b-honouring` on YouTube, `b-ignoring` on DGS"),
    ({"raw": {"decode": "i-runs-kept"}}, "2023",
     "decoder `i-runs-kept` on YouTube, `b-honouring` on DGS"),
])
def test_decoder_note_of_a_row(youtube, dgs, note):
    from experiments.add_result_row import decoder_note

    assert decoder_note(youtube, dgs) == note


README_BY_HAND = """# Experiments

| | | | | Filtered YouTube dev | | | | Raw YouTube dev | | | | Out of domain (DGS) | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **#** | **run** | **change** | **step** | **F1-ma** | **IoU** | **%** | **mF1S** | **F1-ma** | **IoU** | **%** | **mF1S** | **F1-ma** | **IoU** | **%** | **mF1S** |
| 1 | `earlier_run-2026.09.09` | weights 2,20,1 | 100 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

Text after the table.
"""


@needs("torch")
@pytest.mark.parametrize("name, change", [
    ("b-honouring", "weights 2,80,1"),
    ("i-runs-kept", "weights 2,80,1; decoder `i-runs-kept`"),
    ("b-ignoring", "weights 2,80,1; decoder `b-ignoring`"),
])
def test_added_row_names_the_decoder_when_it_is_not_b_honouring(name, change, tmp_path,
                                                                monkeypatch, capsys):
    """`add_result_row.py` itself, on a run folder, scores and a README written by hand."""
    import paths
    import torch
    from experiments import add_result_row

    run = "by_hand-2026.11.25"
    run_dir = paths.output("dist") / run
    run_dir.mkdir(parents=True)
    torch.save({"global_step": 1234}, run_dir / "best.ckpt")
    youtube = tmp_path / "youtube_dev.json"
    youtube.write_text(json.dumps({"raw": {**SCORES, "decode": name},
                                   "filtered": {**SCORES, "decode": name}}))
    dgs = tmp_path / "by_hand.json"
    dgs.write_text(json.dumps({**FILE_BY_HAND, "decode": name,
                               "clips": [{**CLIP, "pred": CLIP_SEGMENTS[name]}]}, indent=2))
    readme = tmp_path / "README.md"
    readme.write_text(README_BY_HAND)

    monkeypatch.setattr(sys, "argv", ["add_result_row.py", "--run", run, "--note",
                                      "weights 2,80,1", "--youtube", str(youtube),
                                      "--dgs", str(dgs), "--readme", str(readme)])
    add_result_row.main()

    row = next(line for line in readme.read_text().splitlines() if "`by_hand-2026.11.25`" in line)
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[:4] == ["2", "`by_hand-2026.11.25`", change, "1,234"]
    assert len(cells) == 16
