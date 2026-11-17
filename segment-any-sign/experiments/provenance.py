"""The state of the code a run trains from: commit, branch and uncommitted work.

The one definition of a dirty tree. `train.py` records it in `run_config.json`
and `pretrain_youtube.slurm` prints it in the job log. Both use this
definition, but at different times: the log line is printed when the job
starts and run_config.json is written later, so an edit to the checkout in
between gives two answers.

    python -m experiments.provenance [directory]

prints the short commit, followed by " (dirty)" when the tree is dirty and
" (dirty unknown)" when it could not be checked. The directory defaults to the
checkout this module lives in.

Dirty means `git status --porcelain` reports anything: a change to a tracked
file, staged or not, or an untracked file that .gitignore does not cover,
whatever the user's `status.showUntrackedFiles` says. When git cannot answer
(no repository, no git, no answer within 30 seconds), dirty is "unknown",
never False: a tree nobody could check is not a clean tree.

A finished run leaves its predictions file and its README row uncommitted, and
edits in the latex submodule count too, so a run started before those are
committed is recorded as dirty. The list of uncommitted paths shows whether
code changed or only results.

Standard library only, and never raises: a missing record must not stop a run
that is ready to train.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _git(repo: Path, *cmd: str) -> str | None:
    """The output of `git -C repo cmd`, or None when the call fails or times out."""
    try:
        return subprocess.run(["git", "-C", str(repo), *cmd], check=True,
                              capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None


def git_state(repo: Path) -> dict:
    """Commit, branch, dirty and uncommitted paths of the checkout at `repo`.

    `dirty` is True, False, or "unknown" when `git status` fails, in which case
    `uncommitted` is None rather than an empty list. `uncommitted` holds the
    paths git reports, relative to the repository root. Commit and branch are
    "unknown" when git cannot read them.
    """
    commit = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    # untracked files spelled out: status.showUntrackedFiles=no would hide them
    status = _git(repo, "status", "--porcelain", "--untracked-files=normal")
    # each line is "XY path": two status columns and a space, then the path.
    # Not stripped first: X is a space for a change that is not staged.
    uncommitted = None if status is None else [line[3:] for line in status.splitlines()]
    return {"commit": commit.strip() if commit else "unknown",
            "branch": branch.strip() if branch else "unknown",
            "dirty": "unknown" if uncommitted is None else bool(uncommitted),
            "uncommitted": uncommitted}


def describe(state: dict) -> str:
    """The short commit, then " (dirty)", or " (dirty unknown)" if unchecked."""
    marker = ""
    if state["dirty"] is True:
        marker = " (dirty)"
    elif state["dirty"] == "unknown":
        marker = " (dirty unknown)"
    return state["commit"][:8] + marker


if __name__ == "__main__":
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    print(describe(git_state(repo)))
