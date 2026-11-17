"""Unit tests for the record of the working tree that every training run keeps.

Run from `segment-any-sign/`:
    pytest tests/test_provenance.py

The repositories are real ones, built in pytest's tmp_path; git is not faked.
Each test imports what it tests, so one missing module cannot stop the others.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]   # segment-any-sign/

# identity and signing given on the command line, so no global config is needed
GIT = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
       "-c", "commit.gpgsign=false"]


def git(repo: Path, *cmd: str) -> str:
    return subprocess.run([*GIT, "-C", str(repo), *cmd], check=True,
                          capture_output=True, text=True).stdout.strip()


def git_state(directory: Path) -> dict:
    from experiments.provenance import git_state
    return git_state(directory)


@pytest.fixture
def repo(tmp_path):
    """A real repository on branch `main` with one committed file, `file.txt`."""
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    (path / "file.txt").write_text("one\n")
    git(path, "add", "file.txt")
    git(path, "commit", "-q", "-m", "init")
    return path


# --- write_run_config --------------------------------------------------------

def test_run_config_records_commit_branch_and_dirty_tree(tmp_path):
    from experiments.train import write_run_config

    args = argparse.Namespace(learning_rate=1e-3)
    mine = argparse.Namespace(phrase="glosses", velocity="off",
                              select_on="mean_mf1s", limit=None)
    write_run_config(tmp_path / "run", args, mine, "validation_mean_mf1s")

    config = json.loads((tmp_path / "run" / "run_config.json").read_text())
    assert set(config["code"]) == {"commit", "branch", "dirty", "uncommitted"}


# --- git_state ---------------------------------------------------------------

def test_clean_tree_is_not_dirty(repo):
    state = git_state(repo)
    assert state["dirty"] is False
    assert state["uncommitted"] == []
    assert state["commit"] == git(repo, "rev-parse", "HEAD")
    assert state["branch"] == "main"


def test_modified_tracked_file_is_dirty(repo):
    (repo / "file.txt").write_text("two\n")
    state = git_state(repo)
    assert state["dirty"] is True
    assert state["uncommitted"] == ["file.txt"]


def test_staged_change_is_dirty(repo):
    (repo / "file.txt").write_text("two\n")
    git(repo, "add", "file.txt")
    state = git_state(repo)
    assert state["dirty"] is True
    assert state["uncommitted"] == ["file.txt"]


def test_untracked_file_is_dirty_and_listed(repo):
    (repo / "new.txt").write_text("new\n")
    state = git_state(repo)
    assert state["dirty"] is True
    assert state["uncommitted"] == ["new.txt"]


def test_untracked_file_is_reported_whatever_the_git_config(repo):
    git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "new.txt").write_text("new\n")
    state = git_state(repo)
    assert state["dirty"] is True
    assert state["uncommitted"] == ["new.txt"]


def test_not_a_repository_is_unknown_not_clean(tmp_path, monkeypatch):
    # keep git from finding a repository in a directory above tmp_path
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    assert git_state(plain) == {"commit": "unknown", "branch": "unknown",
                                "dirty": "unknown", "uncommitted": None}


# --- describe ----------------------------------------------------------------

def test_describe_marks_a_tree_that_could_not_be_checked():
    from experiments.provenance import describe

    state = {"commit": "0123456789abcdef0123456789abcdef01234567", "branch": "main",
             "dirty": "unknown", "uncommitted": None}
    clean = describe({**state, "dirty": False})
    assert state["commit"].startswith(clean)
    assert describe(state) == clean + " (dirty unknown)"


# --- python -m experiments.provenance ----------------------------------------

def test_module_prints_short_commit_then_dirty_marker(repo):
    def run() -> str:
        result = subprocess.run(
            [sys.executable, "-m", "experiments.provenance", str(repo)],
            cwd=ROOT, capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        assert len(lines) == 1
        return lines[0]

    clean = run()
    assert len(clean) >= 7
    assert git(repo, "rev-parse", "HEAD").startswith(clean)

    (repo / "file.txt").write_text("two\n")
    assert run() == clean + " (dirty)"
