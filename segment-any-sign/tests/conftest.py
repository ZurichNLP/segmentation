"""Every test reads a paths file of its own.

`paths.py` reads `paths.local.yaml`, or `paths.uzh.yaml` when that does not exist.
A test must not depend on the machine it runs on and must not create a folder
outside pytest's tmp_path. So each test starts with a copy of
`paths.example.yaml` in its tmp_path, with `output_dir` set to a folder of that
tmp_path. The data keys are empty, as in the example: a test that needs one
writes a file of its own and calls `paths.use_file`.
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
