"""Every test reads a paths file of its own.

`paths.py` reads `paths.local.yaml`, or `paths.uzh.yaml` when that does not exist.
A test must not depend on the machine it runs on and must not create a folder
outside pytest's tmp_path. So each test starts with a paths file in its
tmp_path that holds the keys and values of `paths.example.yaml`, with
`output_dir` set to a folder of that tmp_path. The example is read as YAML, so
its comments and its layout do not matter. The data keys are empty, as in the
example: a test that needs one sets it with the fixture `set_locations`, or
writes a file of its own and calls `paths.use_file`.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]   # segment-any-sign/


@pytest.fixture(autouse=True)
def paths_file(tmp_path):
    """The paths file of this test: the example, with every output inside tmp_path."""
    import paths
    import yaml

    content = yaml.safe_load((ROOT / "paths.example.yaml").read_text())
    content["output_dir"] = str(tmp_path / "outputs")
    file = tmp_path / "paths.test.yaml"
    file.write_text(yaml.safe_dump(content, sort_keys=False))
    paths.use_file(file)
    yield file
    paths.use_file(None)


@pytest.fixture
def set_locations(paths_file):
    """Set keys in the paths file of this test: `set_locations({"data.corpus.root": folder})`.

    A key is written as `paths.describe()` shows it; None empties it.
    """
    def write(locations: dict) -> None:
        import paths
        import yaml

        content = yaml.safe_load(paths_file.read_text())
        for key, folder in locations.items():
            *sections, name = key.split(".")
            target = content
            for section in sections:
                target = target.setdefault(section, {})
            target[name] = None if folder is None else str(folder)
        paths_file.write_text(yaml.safe_dump(content, sort_keys=False))
        paths.use_file(paths_file)

    return write


@pytest.fixture
def no_user_name(paths_file, monkeypatch):
    """An account whose user name cannot be found, as with an arbitrary uid in a container.

    Nothing in the environment names the user and the password database has no
    entry for the account. `getpass.getuser()` then raises KeyError on Python
    3.11 and 3.12, and OSError on 3.14.
    """
    import pwd

    for name in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.delenv(name, raising=False)

    def no_entry(uid):
        raise KeyError(f"getpwuid(): uid not found: {uid}")

    monkeypatch.setattr(pwd, "getpwuid", no_entry)
