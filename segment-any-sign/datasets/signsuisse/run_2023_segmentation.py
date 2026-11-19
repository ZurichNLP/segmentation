"""Run the 2023 segmentation model over a directory of .pose files, writing ELAN.

Targets the SignSuisse DSGS test set by default, producing one .eaf per example
so the predictions can be compared against gloss annotation for sign-level
evaluation.

Why this is a script and not just `pose_to_segments`
----------------------------------------------------
The v2023 CLI hardcodes its decoding thresholds in `bin.py`:

    sign_segments     = probs_to_segments(probs["sign"], 60, 50)
    sentence_segments = probs_to_segments(probs["sentence"], 90, 90)

They are not exposed as flags, so sweeping them means reimplementing the small
amount of glue around the model — which is all this script is. The inference
path is a faithful copy of `bin.py` at tag v2023.

Thresholds (see --help for defaults)
------------------------------------
Moryossef & Jiang (2023) §5.2 grid-searched both parameters from 10 to 90:

  * default in `probs_to_segments`: 50 / 50 for both tiers
  * sign  — b=60, o=40/50/60 "slightly better than the default"
  * phrase — E1s: b=90, o=90;  **E4s: b=80, o=80/90**

Note the v2023 CLI hardcodes phrase thresholds 90/90, which is the *E1s* tuning,
while its `--model` default is also E1s. Running E4s with 90/90 therefore uses
E1s' phrase setting; the paper's E4s optimum is 80/80–90. This script defaults to
the paper's E4s values and prints what it used.

The output directory is named after the **sign** thresholds, since sign-level
evaluation is the purpose here.

Model + decoder are extracted from the v2023 git tag on first run and cached, so
neither the 6.8 MB checkpoint nor the old source lands in this repo.

Annotation scheme
-----------------
Each .eaf carries five tiers, in the order annotators read them top-down:

  ENTRY          reference, read-only. The SignSuisse lexicon headword the
                 example illustrates (e.g. FATAH). Spans the whole clip.
  GERMAN         reference, read-only. The German example sentence. Spans the
                 whole clip; it has no internal alignment to inherit.
  SENTENCE_AUTO  model prediction, read-only. Phrase-level segments.
  SIGN_AUTO      model prediction, read-only. Sign-level segments — the output
                 being evaluated.
  SIGN_MANUAL    **empty; this is where annotation goes.** One span per sign,
                 marking the gold sign boundaries.

The `_AUTO` suffix marks model output, so gold and predicted can never be
confused — by an annotator or by a later scoring script, which selects tiers by
name and needs no external bookkeeping.

SIGN_MANUAL is deliberately created empty rather than pre-filled with SIGN_AUTO.
Correcting predictions is faster, but it anchors annotators to the model's
boundaries, and the resulting gold standard would inherit the very biases the
evaluation is meant to measure. Annotating from scratch keeps the evaluation
independent; SIGN_AUTO is there for reference and comparison, not as a starting
point.

The video is linked by a relative path (`./<id>.mp4`), so a .eaf and its .mp4
opened from the same folder resolve with no prompt.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pympi
import torch
from pose_format import Pose
from pose_format.numpy.representation.distance import DistanceRepresentation
from pose_format.utils.generic import normalize_hands_3d, pose_hide_legs, pose_normalization_info
from pose_format.utils.optical_flow import OpticalFlowCalculator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import paths

# inside the Signsuisse folder, the key `data.signsuisse.root` of the paths file
POSE_FOLDER = "example_mediapipe"
VIDEO_FOLDER = "example_videos"

# The DSGS test set: two translation-direction files whose id sets are disjoint,
# 250 + 250 = 500 example sentences. example_mediapipe holds 18k poses for the
# whole corpus, so this restriction is what keeps us on the evaluation set.
TEST_FILES = ["metadata_test_de_dsgs.csv", "metadata_test_dsgs_de.csv"]
EXPECTED_N = 500

# Tier names. The _AUTO suffix marks model predictions; SIGN_MANUAL is created
# empty for annotators to work in, so gold and predicted never get confused.
SIGN_AUTO_TIER = "SIGN_AUTO"
SENTENCE_AUTO_TIER = "SENTENCE_AUTO"
SIGN_MANUAL_TIER = "SIGN_MANUAL"

V2023_TAG = "v2023"
MODEL_BLOB = "sign_language_segmentation/dist/{model}"
DECODER_BLOB = "sign_language_segmentation/src/utils/probs_to_segments.py"


def find_repo_root() -> Path:
    for base in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (base / "sign_language_segmentation").is_dir():
            return base
    raise RuntimeError("cannot locate the segmentation repo root")


REPO_ROOT = find_repo_root()
CACHE_DIR = REPO_ROOT / ".cache" / "v2023"


def extract_from_tag(blob: str, destination: Path) -> Path:
    """Pull a file out of the v2023 tag, caching it outside version control."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = subprocess.check_output(["git", "-C", str(REPO_ROOT), "show", f"{V2023_TAG}:{blob}"])
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"could not read {blob} from tag {V2023_TAG}. Fetch it first:\n"
            f"  git fetch https://github.com/sign-language-processing/segmentation "
            f"'+refs/tags/{V2023_TAG}:refs/tags/{V2023_TAG}'"
        ) from error
    destination.write_bytes(data)
    return destination


def load_test_examples(data_dir: Path) -> dict[str, dict[str, str]]:
    """The 500 DSGS test examples: id -> {entry, german}.

    `name` is the lexicon headword (the sign the example illustrates) and
    `example` the German sentence; both go into the ELAN file as reference tiers
    for annotators. Uses the stdlib csv module rather than pandas to keep the
    2023 env minimal.
    """
    rows: dict[str, dict[str, str]] = {}
    total = 0
    for name in TEST_FILES:
        path = data_dir / name
        if not path.exists():
            raise SystemExit(f"missing metadata file: {path}")
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                total += 1
                rows[row["id"]] = {"entry": row.get("name", ""), "german": row.get("example", "")}

    if len(rows) != total:
        raise SystemExit(f"expected disjoint id sets, found {total - len(rows)} duplicates")
    if len(rows) != EXPECTED_N:
        raise SystemExit(f"expected {EXPECTED_N} DSGS test ids, found {len(rows)}")
    return rows


def load_decoder():
    """Import the v2023 probs_to_segments without installing the old package."""
    path = extract_from_tag(DECODER_BLOB, CACHE_DIR / "probs_to_segments.py")
    spec = importlib.util.spec_from_file_location("probs_to_segments_v2023", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.probs_to_segments


def process_pose(pose: Pose, optical_flow: bool, hand_normalization: bool) -> Pose:
    """Faithful copy of bin.py::process_pose at tag v2023."""
    pose = pose.get_components(["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])
    pose = pose.normalize(pose_normalization_info(pose.header))
    pose_hide_legs(pose)

    if hand_normalization:
        normalize_hands_3d(pose)

    if optical_flow:
        calculator = OpticalFlowCalculator(fps=pose.body.fps, distance=DistanceRepresentation())
        flow = calculator(pose.body.data)  # frames - 1, people, points
        flow = np.expand_dims(flow, axis=-1)
        flow = np.concatenate([np.zeros((1, *flow.shape[1:]), dtype=flow.dtype), flow], axis=0)
        pose.body.data = np.concatenate([pose.body.data, flow], axis=-1).astype(np.float32)

    return pose


def predict(model, pose: Pose):
    with torch.no_grad():
        data = pose.body.torch().data.tensor[:, 0, :, :].unsqueeze(0)
        return model(data)


def build_eaf(
    tiers: dict[str, list],
    fps: float,
    video: Path | None,
    duration_ms: int,
    reference: dict[str, str] | None = None,
) -> pympi.Elan.Eaf:
    eaf = pympi.Elan.Eaf(author="segment-any-sign / 2023 model")
    eaf.remove_tier("default")  # pympi creates an empty tier we do not want

    # Link the video by a *relative* path so the .eaf travels: copy the .eaf and
    # the .mp4 into the same folder anywhere and ELAN resolves it without being
    # re-pointed. An absolute path would only work on this cluster. The .pose
    # file is deliberately not linked — ELAN cannot render it, and it only
    # produced a second "locate the media" prompt for annotators.
    if video is not None:
        relative = f"./{video.name}"
        eaf.add_linked_file(relative, relpath=relative, mimetype="video/mp4")

    # Tier order is deliberate and matches how annotators read the file top-down:
    # context first (ENTRY, GERMAN), then coarse-to-fine predictions
    # (SENTENCE_AUTO, SIGN_AUTO), then the empty tier they work in last.

    # Reference tiers: not predictions, just context. Each spans the whole clip,
    # since the example sentence has no internal alignment.
    for tier_id, key in (("ENTRY", "entry"), ("GERMAN", "german")):
        eaf.add_tier(tier_id)
        value = ((reference or {}).get(key) or "").strip()
        if value and duration_ms > 0:
            eaf.add_annotation(tier_id, 0, duration_ms, value)

    for tier_id in (SENTENCE_AUTO_TIER, SIGN_AUTO_TIER):
        eaf.add_tier(tier_id)
        for segment in tiers.get(tier_id, []):
            start_ms = int(segment["start"] / fps * 1000)
            end_ms = int(segment["end"] / fps * 1000)
            if end_ms > start_ms:
                eaf.add_annotation(tier_id, start_ms, end_ms)

    eaf.add_tier(SIGN_MANUAL_TIER)  # empty, for annotators

    return eaf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--poses", type=Path, default=None,
                        help="directory of .pose files (default: example_mediapipe in "
                             "data.signsuisse.root of the paths file)")
    parser.add_argument("--videos", type=Path, default=None,
                        help="directory of .mp4 files to link (default: example_videos in "
                             "the same folder)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default: <data>/elan_2023_segmentation_<tag>_<b>_<o>)")
    parser.add_argument("--model", default="model_E4s-1.pth",
                        choices=["model_E4s-1.pth", "model_E1s-1.pth", "model.pth"],
                        help="v2023 checkpoint")
    parser.add_argument("--b", type=float, default=60, help="SIGN b threshold (paper: 60)")
    parser.add_argument("--o", type=float, default=50, help="SIGN o threshold (paper: 40/50/60)")
    parser.add_argument("--sentence-b", type=float, default=80,
                        help="PHRASE b threshold (paper E4s: 80; v2023 CLI hardcodes 90 from E1s)")
    parser.add_argument("--sentence-o", type=float, default=80,
                        help="PHRASE o threshold (paper E4s: 80/90; v2023 CLI hardcodes 90)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N poses")
    parser.add_argument("--skip-existing", action="store_true",
                        help="keep existing .eaf files (default is to overwrite)")
    parser.add_argument("--all-poses", action="store_true",
                        help="process every .pose in --poses instead of only the 500 DSGS test examples")
    args = parser.parse_args()

    if args.poses is None:
        args.poses = paths.data("signsuisse.root") / POSE_FOLDER
    if args.videos is None:
        args.videos = paths.data("signsuisse.root") / VIDEO_FOLDER

    # e.g. model_E4s-1.pth -> e4s
    tag = args.model.replace("model_", "").replace(".pth", "").split("-")[0].lower() or "default"
    out_dir = args.out or args.poses.parent / f"example_segmentation_v2023_{tag}_{int(args.b)}_{int(args.o)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_poses:
        examples = {}
        pose_paths = sorted(args.poses.glob("*.pose"))
    else:
        examples = load_test_examples(paths.data("signsuisse.root"))
        pose_paths = [args.poses / f"{example_id}.pose" for example_id in sorted(examples)]
        missing = [p.name for p in pose_paths if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} pose files missing, e.g. {missing[:5]}")

    if args.limit:
        pose_paths = pose_paths[: args.limit]

    print(f"model          {args.model}  ({tag})")
    print(f"sign   b/o     {args.b:g} / {args.o:g}")
    print(f"phrase b/o     {args.sentence_b:g} / {args.sentence_o:g}")
    print(f"poses          {args.poses}  ({len(pose_paths)} files)")
    print(f"output         {out_dir}\n")

    probs_to_segments = load_decoder()
    model_path = extract_from_tag(MODEL_BLOB.format(model=args.model), CACHE_DIR / args.model)
    model = torch.jit.load(str(model_path))
    model.eval()

    # E4 variants were trained with optical flow + 3D hand normalization
    is_e4 = "E4" in args.model

    written = skipped = failed = 0
    sign_counts, sentence_counts = [], []
    errors: list[tuple[str, str]] = []
    started = time.time()

    for index, pose_path in enumerate(pose_paths, start=1):
        example_id = pose_path.stem
        eaf_path = out_dir / f"{example_id}.eaf"
        if eaf_path.exists() and args.skip_existing:
            skipped += 1
            continue

        try:
            with open(pose_path, "rb") as handle:
                pose = Pose.read(handle.read())
            fps = pose.body.fps
            duration_ms = int(len(pose.body.data) / fps * 1000)
            pose = process_pose(pose, optical_flow=is_e4, hand_normalization=is_e4)
            probs = predict(model, pose)

            tiers = {
                SIGN_AUTO_TIER: probs_to_segments(probs["sign"], args.b, args.o),
                SENTENCE_AUTO_TIER: probs_to_segments(probs["sentence"], args.sentence_b, args.sentence_o),
            }
            sign_counts.append(len(tiers[SIGN_AUTO_TIER]))
            sentence_counts.append(len(tiers[SENTENCE_AUTO_TIER]))

            video = args.videos / f"{example_id}.mp4" if args.videos else None
            build_eaf(tiers, fps, video, duration_ms,
                      reference=examples.get(example_id)).to_file(str(eaf_path))
            written += 1
        except Exception as error:  # keep going; report at the end
            failed += 1
            errors.append((example_id, f"{type(error).__name__}: {error}"))

        if index % 50 == 0 or index == len(pose_paths):
            print(f"  ...{index}/{len(pose_paths)}", flush=True)

    elapsed = time.time() - started
    print(f"\nwrote {written}, skipped {skipped} (already present), failed {failed}"
          f"  in {elapsed:.0f}s")

    if sign_counts:
        print(f"\n  sign segments per clip      mean {np.mean(sign_counts):.1f}  "
              f"median {np.median(sign_counts):.0f}  total {sum(sign_counts):,}")
        print(f"  phrase segments per clip    mean {np.mean(sentence_counts):.1f}  "
              f"median {np.median(sentence_counts):.0f}  total {sum(sentence_counts):,}")

    if errors:
        print(f"\n  first failures:")
        for example_id, message in errors[:5]:
            print(f"    {example_id}: {message}")

    print(f"\noutput: {out_dir}")


if __name__ == "__main__":
    main()
