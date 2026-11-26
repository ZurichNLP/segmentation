"""The record of a run, read back from a checkpoint (`experiments/run_record.py`).

Run from `segment-any-sign/`:
    pytest tests/test_run_record.py

The checkpoints are written in pytest's tmp_path by a small model of ours, as
Lightning writes them during training, or by hand for one written before
checkpoints kept a record. The tests that need the model are marked with
`needs` and skipped, with the missing packages named, where they are not
installed.
"""

import json
import subprocess
import sys

import pytest

from tests.conftest import ROOT
from tests.test_locations import UPSTREAM, needs
from tests.test_train_outputs import RECORD, checkpoint_with

#: what `train.py` writes there, quoted as `shlex.join` quotes it
TYPED = "python experiments/train.py --note 'two words' --sampling video"


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    """A checkpoint of a run whose record has a command as typed."""
    return checkpoint_with({**RECORD, "command_as_typed": TYPED}, tmp_path, monkeypatch)


@pytest.fixture
def older_checkpoint(tmp_path):
    """A checkpoint as written before checkpoints kept the record of their run."""
    import torch

    path = tmp_path / "older.ckpt"
    torch.save({"state_dict": {}, "hyper_parameters": {"hidden_dim": 64}, "global_step": 6},
               path)
    return path


@needs(*UPSTREAM)
def test_record_of_gives_the_record_the_checkpoint_keeps(checkpoint):
    from experiments.run_record import record_of

    assert record_of(checkpoint) == {**RECORD, "command_as_typed": TYPED}


def test_record_of_gives_none_for_a_checkpoint_written_before(older_checkpoint):
    from experiments.run_record import record_of

    assert record_of(older_checkpoint) is None
    assert record_of(str(older_checkpoint)) is None          # a path as a string too


@needs(*UPSTREAM)
def test_the_command_prints_the_record_as_json(checkpoint, capsys):
    from experiments.run_record import main

    assert main([str(checkpoint)]) == 0
    assert json.loads(capsys.readouterr().out) == {**RECORD, "command_as_typed": TYPED}


@needs(*UPSTREAM)
def test_the_command_prints_one_key_as_it_is(checkpoint, capsys):
    from experiments.run_record import main

    # a text as it is, so that the command can be pasted into a shell
    assert main([str(checkpoint), "--key", "command_as_typed"]) == 0
    assert capsys.readouterr().out == TYPED + "\n"
    # anything else as JSON
    assert main([str(checkpoint), "--key", "options"]) == 0
    assert json.loads(capsys.readouterr().out) == RECORD["options"]


@needs(*UPSTREAM)
def test_the_command_names_the_keys_when_the_key_is_not_there(checkpoint, capsys):
    from experiments.run_record import main

    assert main([str(checkpoint), "--key", "comand"]) == 2
    message = capsys.readouterr().err
    assert "comand" in message and "command_as_typed" in message and "options" in message


def test_the_command_says_when_a_checkpoint_keeps_no_record(older_checkpoint, capsys):
    from experiments.run_record import main

    assert main([str(older_checkpoint)]) == 1
    printed = capsys.readouterr()
    assert printed.out == ""
    assert str(older_checkpoint) in printed.err and "run_config.json" in printed.err


@needs(*UPSTREAM)
def test_the_command_runs_as_a_module_from_segment_any_sign(checkpoint):
    done = subprocess.run([sys.executable, "-m", "experiments.run_record", str(checkpoint),
                           "--key", "command_as_typed"],
                          cwd=ROOT, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout == TYPED + "\n"
