"""How the scripts take a decoder.

Run from `segment-any-sign/`:
    pytest tests/test_decoder_use.py

The decoders themselves are tested in test_decoders.py. Each test here imports
what it tests, so one missing function cannot stop the others.
"""

import json
import sys

import pytest

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
