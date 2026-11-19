"""Shared Public DGS Corpus loading.

**One clip list, one set of gold annotations, every model.** Without this the
models drift apart on things that have nothing to do with modelling — a different
joke filter, a different split file, a different person-level flattening — and the
table stops comparing what it claims to compare.

The source is the TFDS build in the folder `data.public_dgs_corpus.tfds` of the
paths file (`../../paths.py`), read through the *original v2023*
`tfds_dataset.py`, because that is what produced the reproduction we verified. It
is vendored out of git into `.cache/v2023_src/` and patched only for imports; see
`../../benchmark/predict_dgs_2023.py` for why the original code is driven
rather than rewritten.

Two entry points, split by how much processing the caller wants:

  * `vendored()` returns the patched v2023 `data` module, for the 2023 model,
    which needs that pipeline end to end.
  * `iter_clips()` returns raw `Pose` objects plus gold spans **in seconds**,
    stopping before any model-specific preprocessing. That is the entry point for
    the 2026 model and anything after it.

Gold spans stay in seconds here on purpose. Each model converts them to frames
using its own protocol — v2023 floors both bounds, the 2026 code walks frame
timestamps — and those conversions differ by up to a frame. Forcing one of them on
the other model would break the reproduction it is meant to be checked against, so
the conversion is left to the caller and documented in `../../benchmark/README.md`.

Both folders come from the paths file: `data.public_dgs_corpus.tfds`, and
`data.public_dgs_corpus.downloads` for the download archive. The TFDS records
point at a share that can no longer be read, so ELAN and CMDI files are
re-resolved by file name into the archive. A function that takes `backup` or
`tfds_root` reads the paths file when the argument is None, and only then.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import paths

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE = REPO_ROOT / ".cache"
SRC = CACHE / "v2023_src"

# The TFDS build on this cluster is `holistic-25`, so that route serves 25fps
# only. `iter_clips_native` reads the archived .pose downloads instead, which are
# the untouched 50fps originals TFDS was built from — no rebuild required.
AVAILABLE_FPS = 25
NATIVE_FPS = 50

COMPONENTS = ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]

# Documents excluded by v2023 `data.py`, unchanged by the 2026 code.
EXCLUDED_IDS = ["1289910", "1245887", "1289868", "1246064", "1584617"]

# Clips whose pose contradicts its annotation. `1177918_b` holds exactly half the
# frames of `1177918_a` from the same recording (14,127 vs 28,254): it is
# truncated at 283s while the annotation runs to 565s, so every span past the cut
# would be clamped onto the final frame. Train split only — dev and test are
# untouched, so no benchmark number moves. Dropped in memory; the file on the
# share is left exactly as it is.
DEFECTIVE_CLIPS = {"1177918_b"}

# file -> git ref it comes from
VENDORED = {
    "data.py": "v2023",
    "pose_utils.py": "e3a020b",
    "tfds_dataset.py": "e3a020b",
    "utils/probs_to_segments.py": "v2023",
    "utils/__init__.py": "v2023",
}


def git_show(ref: str, path: str) -> bytes:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "show", f"{ref}:{path}"])
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"cannot read {path} from {ref}. Fetch the tag with\n"
            f"  git fetch https://github.com/sign-language-processing/segmentation "
            f"'+refs/tags/v2023:refs/tags/v2023'") from error


def vendor_v2023() -> None:
    """Materialise the original source under .cache/, patching only imports."""
    if (SRC / "data.py").exists():
        return
    (SRC / "utils").mkdir(parents=True, exist_ok=True)
    for name, ref in VENDORED.items():
        text = git_show(ref, f"sign_language_segmentation/src/{name}").decode()
        text = text.replace("from _shared.tfds_dataset import", "from tfds_dataset import")
        text = text.replace("from .pose_utils import", "from pose_utils import")
        (SRC / name).write_text(text)
    (SRC / "__init__.py").touch()
    print(f"vendored the v2023 pipeline into {SRC}")


def stub_mediapipe() -> None:
    """tfds_dataset imports mediapipe only for reduce_face, which we never use."""
    holistic = types.ModuleType("mediapipe.solutions.holistic")
    holistic.FACEMESH_CONTOURS = []
    solutions = types.ModuleType("mediapipe.solutions")
    solutions.holistic = holistic
    mediapipe = types.ModuleType("mediapipe")
    mediapipe.solutions = solutions
    sys.modules.setdefault("mediapipe", mediapipe)
    sys.modules.setdefault("mediapipe.solutions", solutions)
    sys.modules.setdefault("mediapipe.solutions.holistic", holistic)


def _downloads(backup: str | None) -> str:
    """The download archive: the folder given, or the one of the paths file."""
    return str(paths.data("public_dgs_corpus.downloads")) if backup is None else backup


def local(path: str, backup: str) -> str:
    """Re-resolve a path stored in a TFDS record into the download archive, by file name."""
    return os.path.join(backup, os.path.basename(path))


def vendored(backup: str | None = None):
    """Import the v2023 `data` module, with annotation lookups redirected.

    Returns the module itself so a caller can use the whole original pipeline.
    `get_elan_sentences` and `filter_dataset` are wrapped to read the download
    archive; nothing else is touched.
    """
    backup = _downloads(backup)
    vendor_v2023()
    stub_mediapipe()
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    import data as v2023  # the original pipeline

    if getattr(v2023, "_redirected", False):
        return v2023

    original_get_elan = v2023.get_elan_sentences
    v2023.get_elan_sentences = lambda path: original_get_elan(local(path, backup))

    def filter_dataset(tf_datum) -> bool:
        """v2023 filter_dataset, with the CMDI path re-resolved.

        Drops the five excluded documents and anything tagged as a joke — the
        latter is what takes the test split from 10 documents to 9.
        """
        if "paths" not in tf_datum:
            return True
        if tf_datum["id"].numpy().decode("utf-8") in EXCLUDED_IDS:
            return False
        with open(local(tf_datum["paths"]["cmdi"].numpy().decode("utf-8"), backup)) as f:
            return "<cmdp:Task>Joke</cmdp:Task>" not in f.read()

    v2023.filter_dataset = filter_dataset
    v2023._redirected = True
    return v2023


def iter_clips(split: str = "test", fps: int = AVAILABLE_FPS,
               tfds_root: str | None = None, backup: str | None = None):
    """Yield one dict per signer-video, with the pose unprocessed.

    Keys: `id` (`<document>_<person>`), `pose` (a `pose_format.Pose` holding only
    the body and hand components), `fps`, and `sentences`.

    Each sentence carries its **own** `start_time` / `end_time` — the German
    translation tier's timeslots — alongside its `glosses`, all in seconds. Both
    are kept because the two models disagree about what a phrase is, and the
    disagreement is not cosmetic: see `sign_phrase_spans`.

    Sentences with no glosses are dropped, matching both models' loaders.
    """
    v2023 = vendored(backup)
    if tfds_root is None:
        tfds_root = str(paths.data("public_dgs_corpus.tfds"))

    import tfds_dataset

    data = tfds_dataset.get_tfds_dataset(
        name="dgs_corpus", poses="holistic", fps=fps, split=split,
        components=COMPONENTS, data_dir=tfds_root, filter_func=v2023.filter_dataset)

    for datum in data:
        poses = datum["pose"]
        elan_path = datum["tf_datum"]["paths"]["eaf"].numpy().decode("utf-8")
        sentences = list(v2023.get_elan_sentences(elan_path))

        for person in ("a", "b"):
            if len(poses[person].body.data) == 0:
                continue
            person_sentences = [
                {"start_time": s["start"] / 1000, "end_time": s["end"] / 1000,
                 "glosses": [{"start_time": g["start"] / 1000, "end_time": g["end"] / 1000}
                             for g in s["glosses"]]}
                for s in sentences
                if s["participant"].lower() == person and len(s["glosses"]) > 0
            ]
            yield {"id": f"{datum['id']}_{person}", "pose": poses[person],
                   "fps": fps, "sentences": person_sentences}


def sign_phrase_spans(sentences, phrase: str = "sentence") -> dict:
    """Derive sign and phrase gold spans from the shared sentence records.

    Sign spans are the glosses either way — the models agree there. Phrase spans
    do not agree, and the choice matters a great deal:

      * `phrase="sentence"` uses the sentence's own annotated bounds. This is what
        the 2026 loader does, and what its model was trained to predict.
      * `phrase="glosses"` runs from a sentence's first gloss to its last, which
        is how v2023 `build_classes_vectors` derives it.

    The gloss extent sits *inside* the annotated sentence, so the two differ by
    the lead-in and trail-out around the signing. Scoring the 2026 model against
    the gloss extent makes its near-contiguous sentence predictions look like they
    merge phrases, and costs it most of its phrase score.
    """
    if phrase not in ("sentence", "glosses"):
        raise ValueError(f"phrase must be 'sentence' or 'glosses', got {phrase!r}")

    signs = [gloss for sentence in sentences for gloss in sentence["glosses"]]
    if phrase == "sentence":
        phrases = [{"start_time": s["start_time"], "end_time": s["end_time"]}
                   for s in sentences]
    else:
        phrases = [{"start_time": s["glosses"][0]["start_time"],
                    "end_time": s["glosses"][-1]["end_time"]}
                   for s in sentences if s["glosses"]]
    return {"sign": signs, "phrase": phrases}


# -- native source: the archived .pose downloads ------------------------------
#
# The TFDS build downsampled poses to 25fps when it was made, and that is baked
# into the records. The 2026 model publishes 50fps numbers and reads raw .pose
# files directly, so serving it from TFDS would silently halve its input rate.
#
# The originals are still in the download archive, keyed by `original_fname` in
# each `.INFO` sidecar (`<document>_<person>.pose`, `<document>-*.eaf`, `.cmdi`).
# That is enough to rebuild the whole clip list without TFDS at all.


def document_id(name: str) -> str:
    """Normalise any corpus filename or split entry to its document id.

    The archive is inconsistent: some files are named `1179224_a.pose`, others
    `1429124-13403249-13545507_b.pose`, and `splits.json` mixes both forms.
    Everything before the first hyphen is the document id, and matching on it is
    what makes the three naming styles line up. Getting this wrong silently drops
    documents rather than failing — 44 of them, before this existed.
    """
    return name.split("-")[0]


def _archive_index(backup: str | None = None) -> dict:
    """Map file kind -> session key -> local path, read from the .INFO sidecars.

    Keys are the **session** stem (`1209309-13425110-13472919`), not the document
    id, because six documents were recorded twice and each session has its own
    eaf and its own poses. Keying on the document id alone silently overwrote 12
    pose files. Poses add `_<person>`.
    """
    import glob
    import json

    backup = _downloads(backup)
    index = {"pose": {}, "eaf": {}, "cmdi": {}}
    for info_path in glob.glob(os.path.join(backup, "*.INFO")):
        try:
            with open(info_path) as f:
                name = json.load(f).get("original_fname", "")
        except (OSError, ValueError):
            continue
        local_path = info_path[: -len(".INFO")]
        if not os.path.exists(local_path):
            continue
        stem, _, kind = name.rpartition(".")
        if kind == "pose":
            index["pose"][stem] = local_path
        elif kind in ("eaf", "cmdi"):
            index[kind][stem] = local_path
    return index


def _clip_id(session: str, person: str, sessions_per_doc: dict) -> str:
    """`<document>_<person>`, keeping the session suffix only when it disambiguates.

    Clip ids stay short and stable for the 394 single-session documents — which
    is every dev and test clip, so benchmark ids are unaffected.
    """
    doc = document_id(session)
    stem = doc if sessions_per_doc[doc] == 1 else session
    return f"{stem}_{person}"


def _splits(splits_path: str | None = None) -> dict:
    """Read the 2026 split file, which extends split.3.0.0-uzh-document.

    Same dev/test document ids the 2023 TFDS split uses, so both routes select
    the same clips.
    """
    import json

    if splits_path is None:
        import sign_language_segmentation
        splits_path = (Path(sign_language_segmentation.__file__).parent /
                       "datasets" / "dgs" / "splits.json")
    return json.loads(Path(splits_path).read_text())


def _pose_meta(pose_path: str, cache: dict) -> tuple[float, int]:
    """fps and frame count from a .pose header, without decoding the body."""
    if pose_path in cache:
        return cache[pose_path]["fps"], cache[pose_path]["total_frames"]

    from pose_format import Pose
    from pose_format.pose_body import EmptyPoseBody

    with open(pose_path, "rb") as f:
        pose = Pose.read(f, pose_body=EmptyPoseBody)
    fps, total = float(pose.body.fps), len(pose.body.data)
    cache[pose_path] = {"fps": fps, "total_frames": total}
    return fps, total


def clip_specs_native(split: str = "test", backup: str | None = None,
                      splits_path: str | None = None):
    """Yield clip *metadata* from the archive, without decoding any pose body.

    This is the training-side entry point: a loader that reads windows out of
    `pose_path` itself only needs `fps` and `total_frames` up front. Reading
    every 50fps body just to count frames would cost minutes per epoch start.

    Keys: `id`, `pose_path`, `fps`, `total_frames`, `sentences` (as in
    `iter_clips`, seconds). Same filters and split file as `iter_clips_native`,
    so train/dev/test here and the benchmark's clips are one and the same.
    """
    import json

    from sign_language_segmentation.datasets.dgs.utils import get_elan_sentences

    index = _archive_index(backup)
    splits = _splits(splits_path)
    dev_ids = {document_id(i) for i in splits["dev"]}
    test_ids = {document_id(i) for i in splits["test"]}
    if split in ("dev", "validation"):
        wanted, excluded = dev_ids, set()
    elif split == "test":
        wanted, excluded = test_ids, set()
    elif split == "train":
        # everything the split file does not name — matching how the 2026
        # loader derives its train set
        wanted, excluded = None, dev_ids | test_ids
    else:
        raise ValueError(f"unknown split {split!r}")

    cache_path = CACHE / "dgs_pose_meta.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    cache_size = len(cache)

    sessions_per_doc: dict = {}
    for session in index["eaf"]:
        doc = document_id(session)
        sessions_per_doc[doc] = sessions_per_doc.get(doc, 0) + 1

    for session in sorted(index["eaf"]):
        doc_id = document_id(session)
        if doc_id in EXCLUDED_IDS or doc_id in excluded:
            continue
        if wanted is not None and doc_id not in wanted:
            continue
        cmdi_path = index["cmdi"].get(session)
        if cmdi_path is None:
            continue
        with open(cmdi_path) as f:
            if "<cmdp:Task>Joke</cmdp:Task>" in f.read():
                continue

        sentences = list(get_elan_sentences(index["eaf"][session]))

        for person in ("a", "b"):
            pose_path = index["pose"].get(f"{session}_{person}")
            if pose_path is None:
                continue
            person_sentences = [
                {"start_time": s["start"] / 1000, "end_time": s["end"] / 1000,
                 "glosses": [{"start_time": g["start"] / 1000, "end_time": g["end"] / 1000}
                             for g in s["glosses"]]}
                for s in sentences
                if s["participant"].lower() == person and len(s["glosses"]) > 0
            ]
            clip_id = _clip_id(session, person, sessions_per_doc)
            if clip_id in DEFECTIVE_CLIPS:
                continue
            fps, total_frames = _pose_meta(pose_path, cache)
            yield {"id": clip_id,
                   "pose_path": pose_path,
                   "fps": fps, "total_frames": total_frames,
                   "sentences": person_sentences}

    if len(cache) != cache_size:
        CACHE.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))


def iter_clips_native(split: str = "test", backup: str | None = None,
                      splits_path: str | None = None):
    """Yield clips from the archived .pose downloads, at their native 50fps.

    Same shape as `iter_clips` — the pose is decoded here — and the same filters,
    so the clip list matches. The difference is the pose itself: untouched
    originals rather than the TFDS build's 25fps downsample.
    """
    from pose_format import Pose

    for spec in clip_specs_native(split, backup, splits_path):
        with open(spec["pose_path"], "rb") as f:
            pose = Pose.read(f)
        yield {"id": spec["id"], "pose": pose, "fps": spec["fps"],
               "sentences": spec["sentences"]}
