"""The project's own files, for the tests that read the code instead of running it."""

import subprocess
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # segment-any-sign/

# percent-format notebooks. A notebook is run and never imported, so it may ask
# for its locations in its configuration cell: see test_no_absolute_paths.py
NOTEBOOKS = "datasets/*/explore.py"


def project_files(*suffixes: str) -> list[str]:
    """Files of `segment-any-sign/` with one of the suffixes, as relative paths.

    The files git tracks, and the untracked ones it does not ignore, so that a
    new file is read before its first commit. The notebooks are among them, and
    the submodule `latex` is not entered.
    """
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    return sorted(name for name in set(listed)
                  if name.endswith(suffixes) and (ROOT / name).is_file())


def notebooks() -> list[str]:
    """The notebooks among the project's files, as relative paths."""
    return [name for name in project_files(".py") if fnmatch(name, NOTEBOOKS)]
