"""Every test reads a paths file of its own.

`paths.py` reads `paths.local.yaml`, or `paths.uzh.yaml` when that does not exist.
A test must not depend on the machine it runs on and must not create a folder
outside pytest's tmp_path. So each test starts with a copy of
`paths.example.yaml` in its tmp_path, with `output_dir` set to a folder of that
tmp_path. The data keys are empty, as in the example: a test that needs one sets
it with the fixture `set_locations`, or writes a file of its own and calls
`paths.use_file`.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]   # segment-any-sign/


@pytest.fixture(autouse=True)
def paths_file(tmp_path):
    """The paths file of this test: the example, with every output inside tmp_path."""
    import paths

    example = (ROOT / "paths.example.yaml").read_text()
    assert example.count("\noutput_dir:\n") == 1, "paths.example.yaml: output_dir is not empty"
    file = tmp_path / "paths.test.yaml"
    file.write_text(example.replace("\noutput_dir:\n",
                                    f"\noutput_dir: {tmp_path / 'outputs'}\n"))
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
