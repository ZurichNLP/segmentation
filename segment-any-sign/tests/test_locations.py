"""The loaders read their folders from the paths file.

Run from `segment-any-sign/`:
    pytest tests/test_locations.py

Each corpus is a few hand-made files in pytest's tmp_path, and the paths file of
the test points at them (`set_locations`, see conftest.py). No test reads a real
corpus. Each test imports what it tests, so one missing module cannot stop the
others.
"""

import inspect
import json
from pathlib import Path

import pytest

DGS_DOWNLOADS = "data.public_dgs_corpus.downloads"


def message_of(call, *args, **kwargs) -> str:
    """The message of the PathsError that the call must raise."""
    import paths

    with pytest.raises(paths.PathsError) as error:
        call(*args, **kwargs)
    return str(error.value)


# --- Public DGS Corpus -------------------------------------------------------

def dgs_archive(folder: Path) -> dict:
    """A download archive of one document, and the index the loader must build from it.

    As in the real archive, a file has a meaningless name and sits beside an
    `.INFO` file that holds its original name.
    """
    originals = ["1234567_a.pose", "1234567_b.pose", "1234567.eaf", "1234567.cmdi",
                 "1234567.mp4"]
    folder.mkdir()
    for number, original in enumerate(originals):
        (folder / f"download_{number}").write_text("")
        (folder / f"download_{number}.INFO").write_text(json.dumps({"original_fname": original}))
    # a sidecar whose file is gone, and one that is not JSON: both are skipped
    (folder / "gone.INFO").write_text(json.dumps({"original_fname": "7654321_a.pose"}))
    (folder / "broken.INFO").write_text("not json")
    return {"pose": {"1234567_a": f"{folder}/download_0", "1234567_b": f"{folder}/download_1"},
            "eaf": {"1234567": f"{folder}/download_2"},
            "cmdi": {"1234567": f"{folder}/download_3"}}


def test_dgs_archive_index_reads_the_configured_folder(tmp_path, set_locations):
    from datasets.public_dgs_corpus import load as dgs_data

    expected = dgs_archive(tmp_path / "downloads")
    set_locations({DGS_DOWNLOADS: tmp_path / "downloads"})

    assert dgs_data._archive_index() == expected
    # and it is what the call with the folder gives, file path for file path
    assert dgs_data._archive_index(str(tmp_path / "downloads")) == expected


def test_dgs_archive_index_with_a_folder_given_needs_no_key(tmp_path):
    from datasets.public_dgs_corpus import load as dgs_data

    expected = dgs_archive(tmp_path / "elsewhere")
    assert dgs_data._archive_index(str(tmp_path / "elsewhere")) == expected


def test_dgs_archive_index_without_the_key_names_it(tmp_path, set_locations):
    from datasets.public_dgs_corpus import load as dgs_data

    assert DGS_DOWNLOADS in message_of(dgs_data._archive_index)

    set_locations({DGS_DOWNLOADS: tmp_path / "not_there"})
    message = message_of(dgs_data._archive_index)
    assert DGS_DOWNLOADS in message
    assert str(tmp_path / "not_there") in message


def test_dgs_clip_specs_without_the_key_names_it():
    from datasets.public_dgs_corpus import load as dgs_data

    # a generator: the folder is asked for when the first clip is, not before
    specs = dgs_data.clip_specs_native("dev")
    assert DGS_DOWNLOADS in message_of(next, specs)


@pytest.mark.parametrize("function, parameter", [
    ("vendored", "backup"),
    ("iter_clips", "tfds_root"),
    ("iter_clips", "backup"),
    ("_archive_index", "backup"),
    ("clip_specs_native", "backup"),
    ("iter_clips_native", "backup"),
])
def test_dgs_loader_holds_no_folder_as_a_default(function, parameter):
    from datasets.public_dgs_corpus import load as dgs_data

    # a default is evaluated at import, where no location may be asked for
    parameters = inspect.signature(getattr(dgs_data, function)).parameters
    assert parameters[parameter].default is None


def test_dgs_loader_holds_no_folder_as_a_constant():
    from datasets.public_dgs_corpus import load as dgs_data

    assert not hasattr(dgs_data, "BACKUP")
    assert not hasattr(dgs_data, "TFDS_ROOT")
    # both callers of `local` pass the folder, so it has no default at all
    assert inspect.signature(dgs_data.local).parameters["backup"].default is inspect.Parameter.empty


def test_dgs_dataset_holds_no_folder_as_a_default():
    from experiments.dgs_dataset import DGSCorpusDataset

    assert inspect.signature(DGSCorpusDataset.__init__).parameters["backup"].default is None
