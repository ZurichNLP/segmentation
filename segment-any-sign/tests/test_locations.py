"""The loaders read their folders from the paths file.

Run from `segment-any-sign/`:
    pytest tests/test_locations.py

Each corpus is a few hand-made files in pytest's tmp_path, and the paths file of
the test points at them (`set_locations`, see conftest.py). No test reads a real
corpus. Each test imports what it tests, so one missing module cannot stop the
others.

Most tests need only what the metric tests need. A test that reads a pose file
or imports a module built on the upstream package is marked with `needs`, and
is skipped, with the missing packages named, where they are not installed. In
the sas environment no test is skipped.
"""

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

DGS_DOWNLOADS = "data.public_dgs_corpus.downloads"

# the upstream package, installed with `pip install -e .`, and the packages
# that its module `datasets.common` imports
UPSTREAM = ("sign_language_segmentation", "torch", "pose_format", "pose_anonymization")


def needs(*packages: str):
    """Mark a test: skipped when one of these packages is not installed, which the reason names.

    Only an absent package skips a test. A package that is installed and fails
    to import, or a missing package that is not listed, still fails it.
    """
    missing = [package for package in packages if importlib.util.find_spec(package) is None]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"not installed: {', '.join(missing)}. The sas environment has every package")


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


@needs(*UPSTREAM, "pympi")              # the ELAN reader of the upstream package
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


@needs(*UPSTREAM)
def test_dgs_dataset_holds_no_folder_as_a_default():
    from experiments.dgs_dataset import DGSCorpusDataset

    assert inspect.signature(DGSCorpusDataset.__init__).parameters["backup"].default is None


# --- YouTube-SL-25 -----------------------------------------------------------

YOUTUBE_ROOT = "data.youtube_sl25.root"
YOUTUBE_VIDEOS = "data.youtube_sl25.videos"

# upstream's example pose: a real file with a header, small enough for a test
EXAMPLE_POSE = Path(__file__).resolve().parents[2] / "sign_language_segmentation" / "tests" / "example.pose"

VTT = """WEBVTT

00:00:01.000 --> 00:00:02.500
hello

00:00:03.000 --> 00:00:04.000
[Music]

00:01:05.250 --> 00:01:06.000
world
"""                                                     # [Music] is a sound marker: dropped
VTT_CUES = [[1000, 2500], [65250, 66000]]               # milliseconds, as cached

# video id -> (subtitle file, sign language, cues in milliseconds)
YOUTUBE_VIDEOS_ON_DISK = {
    "vid_a": ("vid_a.en.vtt", "ase", VTT_CUES),
    "vid_b": ("vid_b.de.vtt", "gsg", VTT_CUES),
    "vid_c": ("vid_c.en-GB.vtt", "ase", VTT_CUES),
}


@pytest.fixture
def youtube(tmp_path, set_locations):
    """A corpus of three videos, and the paths file pointing at it. Returns its folder.

    Besides the three: a pose without subtitles, subtitles without a pose, and a
    video whose pose file is cut short. The metadata has CRLF line endings, as
    the real file has.
    """
    root = tmp_path / "youtube"
    (root / "mediapipe").mkdir(parents=True)
    (root / "subtitles").mkdir()
    for video, (subtitle, _, _) in YOUTUBE_VIDEOS_ON_DISK.items():
        (root / "mediapipe" / f"{video}.pose").write_bytes(EXAMPLE_POSE.read_bytes())
        (root / "subtitles" / subtitle).write_text(VTT)
    (root / "mediapipe" / "vid_pose_only.pose").write_bytes(EXAMPLE_POSE.read_bytes())
    (root / "subtitles" / "vid_subtitle_only.en.vtt").write_text(VTT)
    (root / "mediapipe" / "vid_cut.pose").write_bytes(EXAMPLE_POSE.read_bytes()[:10])
    (root / "subtitles" / "vid_cut.en.vtt").write_text(VTT)
    (root / "youtube-sl-25_youtube-sl-25-metadata.csv").write_text(
        "".join(f"{video},{language}\r\n"
                for video, (_, language, _) in YOUTUBE_VIDEOS_ON_DISK.items()), newline="")
    set_locations({YOUTUBE_ROOT: root})
    return root


def example_pose_header() -> tuple[float, int]:
    """fps and frame count of the example pose, read here in full, not by the loader."""
    from pose_format import Pose

    with open(EXAMPLE_POSE, "rb") as handle:
        pose = Pose.read(handle)
    return float(pose.body.fps), len(pose.body.data)


def test_youtube_index_languages_and_cues_read_the_configured_folder(youtube, tmp_path):
    from datasets.youtube_sl25 import load as yt

    index = yt.video_index()
    assert index == {
        video: {"pose": f"{youtube}/mediapipe/{video}.pose",
                "subtitle": f"{youtube}/subtitles/{subtitle}"}
        for video, subtitle in [("vid_a", "vid_a.en.vtt"), ("vid_b", "vid_b.de.vtt"),
                                ("vid_c", "vid_c.en-GB.vtt"), ("vid_cut", "vid_cut.en.vtt")]}
    assert list(index) == ["vid_a", "vid_b", "vid_c", "vid_cut"]
    assert yt.languages() == {"vid_a": "ase", "vid_b": "gsg", "vid_c": "ase"}

    expected = {video: VTT_CUES for video in index}
    assert yt._cues(index) == expected

    # the cache is in the configured cache folder...
    cache_file = tmp_path / "outputs" / "cache" / "youtube_sl25_cues.json"
    assert json.loads(cache_file.read_text()) == expected
    # ...and that file is what the next call reads
    cache_file.write_text(json.dumps({**expected, "vid_a": [[1, 2]]}))
    assert yt._cues(index)["vid_a"] == [[1, 2]]


@needs("pose_format")
def test_youtube_clip_specs_are_what_the_files_say(youtube, tmp_path):
    from datasets.youtube_sl25 import load as yt

    fps, frames = example_pose_header()
    sentences = [{"start_time": 1.0, "end_time": 2.5, "glosses": []},
                 {"start_time": 65.25, "end_time": 66.0, "glosses": []}]
    assert list(yt.clip_specs("all")) == [
        {"id": video, "pose_path": f"{youtube}/mediapipe/{video}.pose", "fps": fps,
         "total_frames": frames, "sentences": sentences}
        for video in ("vid_a", "vid_b", "vid_c")]            # vid_cut cannot be read

    # the header cache is in the configured cache folder, the unreadable file as null
    cache = json.loads((tmp_path / "outputs" / "cache" / "youtube_sl25_pose_meta.json").read_text())
    assert cache == {**{video: {"fps": fps, "total_frames": frames}
                        for video in ("vid_a", "vid_b", "vid_c")}, "vid_cut": None}


@needs("pose_format")
def test_youtube_train_and_dev_split_the_videos(youtube):
    from datasets.youtube_sl25 import load as yt

    def ids(split):
        return [spec["id"] for spec in yt.clip_specs(split, per_language=1)]

    # one dev video per sign language: two languages, three videos
    assert len(ids("dev")) == 2 and len(ids("train")) == 1
    assert sorted(ids("dev") + ids("train")) == ids("all") == ["vid_a", "vid_b", "vid_c"]


def test_youtube_loader_without_the_key_names_it():
    from datasets.youtube_sl25 import load as yt

    assert YOUTUBE_ROOT in message_of(yt.video_index)
    assert YOUTUBE_ROOT in message_of(yt.languages)
    assert YOUTUBE_ROOT in message_of(next, yt.clip_specs("train"))


def test_youtube_loader_holds_no_folder_as_a_constant():
    from datasets.youtube_sl25 import load as yt

    for name in ("ROOT", "POSES", "SUBTITLES", "METADATA", "CACHE_DIR", "META_CACHE",
                 "CUE_CACHE", "QUALITY_CACHE"):
        assert not hasattr(yt, name), name


@needs(*UPSTREAM)
def test_youtube_manifest_records_the_configured_folder(youtube):
    from experiments.youtube_dataset import YouTube25Dataset

    # only what the manifest reads: building the dataset would load every clip
    dataset = YouTube25Dataset.__new__(YouTube25Dataset)
    dataset.dev_filter = False
    dataset._all_split_ids = {}
    assert dataset.get_split_manifest()["source"] == str(youtube)


def test_video_path_is_none_without_the_key():
    from experiments.plot_segmentation import video_path

    assert video_path("vid_a") is None


def test_video_path_looks_under_ase_then_non_ase(tmp_path, set_locations):
    from experiments.plot_segmentation import video_path

    videos = tmp_path / "videos"
    for folder, video in [("ase", "vid_a"), ("non-ase", "vid_b"), ("ase", "vid_c"),
                          ("non-ase", "vid_c")]:
        (videos / folder / "downloads" / video).mkdir(parents=True)
        (videos / folder / "downloads" / video / f"{video}.mp4").write_bytes(b"")
    set_locations({YOUTUBE_VIDEOS: videos})

    assert video_path("vid_a") == f"{videos}/ase/downloads/vid_a/vid_a.mp4"
    assert video_path("vid_b") == f"{videos}/non-ase/downloads/vid_b/vid_b.mp4"
    assert video_path("vid_c") == f"{videos}/ase/downloads/vid_c/vid_c.mp4"
    assert video_path("vid_without_video") is None


def test_video_path_is_none_with_a_folder_that_is_not_there(tmp_path, set_locations):
    from experiments.plot_segmentation import video_path

    # as when the key is not set: the figure then draws the pose
    set_locations({YOUTUBE_VIDEOS: tmp_path / "not_there"})
    assert video_path("vid_a") is None


# --- the class-weights cache -------------------------------------------------

@needs(*UPSTREAM)
def test_class_weights_are_cached_in_the_configured_folder(tmp_path, monkeypatch):
    import argparse

    from experiments.train import inverse_class_weights

    from sign_language_segmentation.datasets import common

    class OneClip:
        """100 frames at 25 fps: one sign of 0.4 s (10 frames), one phrase of 2 s (50)."""

        def __init__(self):
            self.items = [{"id": "clip", "total_frames": 100, "fps": 25.0,
                           "glosses": [{"start": 0, "end": 400}],
                           "sentences": [{"start": 0, "end": 2000}]}]

        @classmethod
        def from_args(cls, split, args, **kwargs):
            return cls()

    monkeypatch.setitem(common.DATASET_REGISTRY, "one_clip", OneClip)
    args = argparse.Namespace(datasets="one_clip", num_frames=1024, velocity=False,
                              phrase="glosses")

    weights = inverse_class_weights(args, ["sign", "sentence"])
    # total / count per class, in upstream's order UNK, O, B, I
    assert weights == {"sign": [0.0, pytest.approx(100 / 90), 100.0, pytest.approx(100 / 9)],
                       "sentence": [0.0, 2.0, 100.0, pytest.approx(100 / 49)]}

    files = list((tmp_path / "outputs" / "cache").iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("class_weights_one_clip_inverse_")
    assert json.loads(files[0].read_text()) == weights
    # and the next call reads that file
    files[0].write_text(json.dumps({"sign": [0.0, 1.0, 2.0, 3.0]}))
    assert inverse_class_weights(args, ["sign", "sentence"]) == {"sign": [0.0, 1.0, 2.0, 3.0]}
