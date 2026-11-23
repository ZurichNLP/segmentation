# %% [markdown]
# # How2Sign (ASL) — assessment
#
# **Conclusion: not usable for sign-level segmentation.** Recorded here so we do
# not re-investigate it later. It remains interesting for *phrase*-level work.
#
# Two independent reasons:
#
# 1. **The gloss annotation is not timed per sign.** The CVPR 2021 paper is
#    careful about this — ELAN tiers are "time-aligned to the video files, giving
#    us the start and end boundaries of **each sentence** and producing what we
#    call the **sentence boundaries**". Glosses are transcribed as a sequence;
#    per-gloss start/end times are never claimed. `Table 2` reports statistics
#    exhaustively and contains **no gloss row at all** — no gloss count, no gloss
#    vocabulary, no glossed hours.
# 2. **The glosses were never released.** The download section of
#    <https://how2sign.github.io/> covers RGB video, keypoints and English
#    translations, and states "This section is under construction. We will be
#    releasing the other modalities soon!". GitHub issue
#    [#5 "How can I get the gloss annotations for each sentences?"](https://github.com/how2sign/how2sign.github.io/issues/5)
#    was filed in December 2021 and is still open and unassigned.
#
# So How2Sign offers sentence boundaries plus a gloss *sequence* — the same shape
# as MEDIAPI-SKEL, and the same limitation. See `../mediapi_skel/`.
#
# Paper: [How2Sign, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.pdf)

# %% [markdown]
# ## Statistics reported in the paper
#
# Hardcoded from Table 2 — we are not recomputing these, they are for reference.

# %%
import pandas as pd

PAPER = pd.DataFrame(
    {
        "train": [2213, 69.62, 31128, 31128, 15686, 8],
        "val": [132, 3.91, 1741, 1741, 3218, 5],
        "test": [184, 5.59, 2322, 2322, 3670, 6],
        "total (green screen)": [2529, 79.12, 35191, 35191, "16k+", 9],
        "panoptic": [124, 2.96, 1582, 1582, 3260, 6],
    },
    index=["SL videos", "duration (h)", "clips", "sentences", "vocabulary", "signers"],
)

print("How2Sign, Table 2 (CVPR 2021)\n")
print("  clip average          162 frames / 5.4 s / 17 words")
print("  fingerspelled         ~20% of the English vocabulary")
print("  gloss annotation cost ~1 hour of annotator time per 90 s of video")
print("  gloss timing          SENTENCE level only — no per-sign boundaries")
PAPER

# %% [markdown]
# ## Where it lives on our server
#
# **Main copy — essentially complete:**
# `/shares/iict-sp2.ebling.cl.uzh/common/How2Sign/`
#
# ```
# download_how2sign.sh
# downloads/                     original zips
# sentence_level/{train,val,test}/rgb_front/{clips, eaf_files, pose_estimation}
# sentence_level/{train,val,test}/text/en/raw_text
# video_level/{raw_videos, mediapipe, subtitles_manual, subtitles_auto,
#              subtitles_audio, segmentation, *_embedding}
# ```
#
# **A second, partial copy** belonging to another project (`parsign`) holds only
# pose `.npz` for the train split:
# `/shares/sign-language.ebling.cl.uzh/cache-dataset-parsign/How2Sign/how2sign/`
#
# > ### The `eaf_files` are NOT gold annotation
# >
# > `sentence_level/*/rgb_front/eaf_files/` contains ~35k `.eaf` files with `SIGN`
# > and `SENTENCE` tiers, which looks like exactly what we want. It is not.
# > Their header reads `AUTHOR="sign-langauge-processing/transcription"`, dated
# > 2024-06-01, and every `ANNOTATION_VALUE` is **empty** — boundaries with no
# > gloss labels. These are **outputs of the 2023 segmentation model**, the same
# > pipeline we run in `../signsuisse/run_2023_segmentation.py`, not human
# > annotation. `video_level/segmentation/` similarly holds an 81-directory
# > `E4s-1_<b>_<o>` threshold sweep (9x9 grid, 10-90), i.e. earlier decoding
# > experiments.
# >
# > Treating these as ground truth would mean evaluating the model against itself.

# %%
import glob
import sys
from pathlib import Path

# paths.py reads every location from the paths file. It is
# in segment-any-sign/, found from the working directory.
sys.path.insert(0, next(
    str(base / "segment-any-sign")
    for base in [Path.cwd(), *Path.cwd().parents]
    if (base / "segment-any-sign" / "paths.py").is_file()))
import paths

MAIN = paths.data("how2sign.root")
CACHE = paths.data("how2sign.parsign_cache")


def count(path, pattern="*"):
    return len(glob.glob(str(path / pattern))) if path.is_dir() else 0


print("main copy —", MAIN)
print(f"  video_level/raw_videos        {count(MAIN / 'video_level/raw_videos'):>6}")
print(f"  video_level/mediapipe         {count(MAIN / 'video_level/mediapipe'):>6}")
for kind in ("subtitles_manual", "subtitles_auto", "subtitles_audio"):
    print(f"  video_level/{kind:<17} {count(MAIN / 'video_level' / kind):>6}")
print(f"  video_level/segmentation      {count(MAIN / 'video_level/segmentation'):>6}  "
      f"(E4s-1_<b>_<o> sweep dirs)")
for split in ("train", "val", "test"):
    n_eaf = count(MAIN / f"sentence_level/{split}/rgb_front/eaf_files", "*.eaf")
    print(f"  sentence_level/{split:<5} eaf_files {n_eaf:>6}  (model output, not gold)")

print(f"\nparsign cache — {CACHE}")
print("  train .npz", count(CACHE / "train", "*.npz"), "| val", count(CACHE / "val", "*.npz"),
      "| test", count(CACHE / "test", "*.npz"))

# %% [markdown]
# ## Verdict and notes
#
# - **Do not use for sign-level segmentation.** No per-sign timings exist, and the
#   glosses are not obtainable even if they did.
# - **Potentially useful for phrase-level segmentation**, where sentence
#   boundaries are exactly the right label — but only if the sentence-boundary
#   CSVs are downloaded; our local copy has none.
# - **Notable for fingerspelling.** ~20% of the vocabulary is fingerspelled, which
#   is high, and relevant to that phenomenon in the proposal — though again,
#   without per-sign timings it cannot support fingerspelling *segmentation*.
# - **Sentence-level data is ready to use** if we want phrase segmentation:
#   `sentence_level/*/rgb_front/` has the clips and pose estimation, and
#   `text/en/raw_text` the English sentences, alongside `video_level/
#   subtitles_manual` (2,528 hand-checked `.vtt`). Note manual, auto and
#   audio-derived subtitles are all present — pick deliberately, they are not
#   equivalent.
# - **Mixed frame rates** (23.976 / 24 / 60 fps in a sample of the parsign cache),
#   so nothing here should assume a constant fps.
# - **An 81-point threshold sweep already exists** in `video_level/segmentation/`
#   from earlier work. Useful as a reference for decoding behaviour on ASL, and a
#   reminder that these are predictions, not labels.
# - The `parsign` cache belongs to another project; the main copy is shared group
#   space. Do not modify either in place — copy out if needed.
#
# ### Running
#
# ```bash
# jupytext --to ipynb --execute datasets/how2sign/explore.py -o - \
#   | jupyter nbconvert --stdin --to markdown --output explore --output-dir datasets/how2sign
# ```
