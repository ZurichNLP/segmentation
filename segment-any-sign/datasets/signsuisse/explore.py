# %% [markdown]
# # SignSuisse (DSGS test set) — data exploration
#
# **Scope, stated up front:** this analysis covers the **DSGS portion of the
# WMT-SLT SignSuisse *test* set only**, read from our local copy on the server at
# `/shares/sign-language.ebling.cl.uzh/Signsuisse`. Not the training split, and not the
# LSF or LIS portions.
#
# The plan for this data is to take the **example sentences**, annotate them at
# gloss level, and use them as a sign-level evaluation set for the segmentation
# model.
#
# **How the 500 DSGS examples are defined.** The WMT-SLT test set is 1000 items
# split four ways by *translation direction*, 250 each: `de_dsgs`, `dsgs_de`,
# `fr_lsf`, `it_lis`. The two DSGS files look like they might be the same items
# with source and target swapped — both carry `spokenLanguage=de,
# signedLanguage=dsgs` and their file sizes differ by under 50 bytes — but their
# ID sets are disjoint. The direction label is the task, not the data, so the
# union is 250 + 250 = **500 distinct DSGS examples**. This notebook asserts that
# rather than assuming it.
#
# Two duration columns are easy to confuse:
#
# - `videoDuration` — the **isolated lexicon sign** (the dictionary entry).
# - `exampleVideoDuration` — the **example sentence** clip. This is the one we
#   annotate and evaluate on.
#
# Source: <https://www.wmt-slt.com/data>

# %% [markdown]
# ## Configuration

# %%
import sys
from pathlib import Path

# paths.py reads every location from the paths file. It is
# in segment-any-sign/, found from the working directory.
sys.path.insert(0, next(
    str(base / "segment-any-sign")
    for base in [Path.cwd(), *Path.cwd().parents]
    if (base / "segment-any-sign" / "paths.py").is_file()))
import paths

# true path, not a symlink into it. The dataset was migrated here from
# /shares/easier.ebling.cl.uzh/WMT_23/signsuisse; metadata and media are identical.
DATA_DIR = paths.data("signsuisse.root")

# we use the test split, DSGS only — both translation directions
TEST_FILES = ["metadata_test_de_dsgs.csv", "metadata_test_dsgs_de.csv"]
SIGNED_LANGUAGE = "dsgs"
EXPECTED_N = 500

# media directories, keyed by bare example id (e.g. 117615.pose / 117615.mp4)
MEDIA_DIRS = {
    "mediapipe": DATA_DIR / "example_mediapipe",
    "openpose": DATA_DIR / "example_openpose",
    "videos": DATA_DIR / "example_videos",
}

for path in [DATA_DIR, *(DATA_DIR / f for f in TEST_FILES), *MEDIA_DIRS.values()]:
    print(f"{'OK  ' if path.exists() else 'MISSING'} {path}")

# %% [markdown]
# ## Provenance

# %%
import datetime
import platform
import socket
import subprocess
import sys


def find_repo_root():
    """Walk up from the working directory until the package folder appears."""
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "sign_language_segmentation").is_dir():
            return base
    raise RuntimeError(f"cannot locate the segmentation repo root from {Path.cwd()}")


REPO_ROOT = find_repo_root()


def git(*args):
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()
    except Exception:
        return "unavailable"


print(f"timestamp   {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"host        {socket.gethostname()}")
print(f"python      {platform.python_version()} ({sys.prefix})")
print(f"git branch  {git('rev-parse', '--abbrev-ref', 'HEAD')}")
print(f"git commit  {git('rev-parse', '--short', 'HEAD')}")
print(f"git dirty   {'yes' if git('status', '--porcelain') else 'no'}")
print(f"data        {DATA_DIR}")

# %% [markdown]
# ## Loading
#
# Load both direction files and verify the disjointness claim rather than trusting it.

# %%
import pandas as pd

parts = {name: pd.read_csv(DATA_DIR / name) for name in TEST_FILES}
for name, part in parts.items():
    print(f"{name:<32} {len(part):>4} rows, {part.id.nunique():>4} unique ids")

ids_a, ids_b = (set(part.id) for part in parts.values())
overlap = ids_a & ids_b
print(f"\noverlap between the two direction files: {len(overlap)}")

df = pd.concat(parts.values(), ignore_index=True)

assert df.id.nunique() == len(df), "duplicate ids within the concatenated test set"
assert len(df) == EXPECTED_N, f"expected {EXPECTED_N} DSGS test examples, found {len(df)}"
assert set(df.signedLanguage.unique()) == {SIGNED_LANGUAGE}, df.signedLanguage.unique()

print(f"\n{len(df)} distinct DSGS test examples")
print(f"languages: {df.groupby(['spokenLanguage', 'signedLanguage']).size().to_dict()}")
df.head(3)

# %% [markdown]
# ## Media coverage
#
# Gloss annotation needs the example clips, so check every id resolves in each
# media directory. `no_example_ids.txt` lists entries the corpus has no example
# sentence for — none of them should land in our 500.

# %%
coverage = {}
for label, directory in MEDIA_DIRS.items():
    present = {int(p.stem) for p in directory.iterdir() if p.stem.isdigit()}
    missing = set(df.id) - present
    coverage[label] = {"present": len(set(df.id) & present), "missing": len(missing)}
    if missing:
        print(f"  {label}: missing ids {sorted(missing)[:10]}")

no_example = {int(line) for line in (DATA_DIR / "no_example_ids.txt").read_text().split()}
print(f"ids in no_example_ids.txt that are in our set: {len(set(df.id) & no_example)}")

pd.DataFrame(coverage).T

# %% [markdown]
# ## Summary

# %%
print("=== SignSuisse DSGS test set — basic statistics ===\n")
print(f"  examples                    {len(df):,}")
print(f"  unique lexical entries      {df.name.nunique():,}")
print(f"  categories                  {df.category.nunique():,}")
print(f"  total example video         {df.exampleVideoDuration.sum() / 60:,.1f} min")
print(f"  total isolated sign video   {df.videoDuration.sum() / 60:,.1f} min")

durations = pd.DataFrame({
    "example sentence (s)": df["exampleVideoDuration"],
    "isolated sign (s)": df["videoDuration"],
    "words in sentence": df["example"].str.split().str.len(),
    "chars in sentence": df["example"].str.len(),
}).describe(percentiles=[0.1, 0.5, 0.9])
durations.round(2)

# %% [markdown]
# ### Expected annotation effort
#
# A rough sizing for the gloss annotation, using the DGS corpus sign rate as a
# reference point. It is only an estimate — DSGS example sentences are read,
# studio-recorded and isolated, so the real rate will differ.

# %%
# Use a sign *rate*, not the median sign duration: signs do not tile the
# timeline back-to-back (there are transitions and pauses), and DGS merges both
# hands so its gloss spans overlap. Dividing by median duration overcounts badly.
# Moryossef & Jiang (2023) §4.1: 613 signs per training video, 7.55 min average.
DGS_SIGNS_PER_SECOND = 613 / (7.55 * 60)

total_seconds = df["exampleVideoDuration"].sum()
est_signs = total_seconds * DGS_SIGNS_PER_SECOND
print(f"  total example video        {total_seconds / 60:,.1f} min")
print(f"  DGS reference rate         {DGS_SIGNS_PER_SECOND:.2f} signs/s")
print(f"  est. signs total           {est_signs:,.0f}")
print(f"  est. signs per clip        {est_signs / len(df):,.1f}")
print(f"  German words per sentence  {df['example'].str.split().str.len().mean():.1f} (mean)")
print(f"\n  (for scale: our DGS pass counted ~350k glosses)")

# %% [markdown]
# ## Distributions

# %%
import matplotlib.pyplot as plt

# categorical slots 1-3 of the validated reference palette
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": MUTED,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 9,
    "grid.color": "#e6e5e1",
    "grid.linewidth": 0.8,
})


def style(ax, title, xlabel, ylabel="count", axis="y"):
    """Recessive grid on the value axis only; titles carry the identity."""
    ax.set_title(title, loc="left", color=INK, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)
    return ax


# %%
# example clip duration — the clips we will annotate
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(df["exampleVideoDuration"], bins=40, color=BLUE, edgecolor="white", linewidth=0.5, zorder=2)
median = df["exampleVideoDuration"].median()
ax.axvline(median, color=INK, linewidth=1.2, linestyle="--", zorder=3)
ax.annotate(f"median {median:.1f} s", xy=(median, ax.get_ylim()[1] * 0.92),
            xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9)
style(ax, f"Example sentence duration  (n={len(df):,})", "duration (s)")
plt.tight_layout()
plt.show()

# %%
# example sentence vs isolated sign — two very different regimes
fig, ax = plt.subplots(figsize=(7, 3.4))
for column, colour, label in (("exampleVideoDuration", BLUE, "example sentence"),
                              ("videoDuration", ORANGE, "isolated sign")):
    ax.hist(df[column], bins=40, histtype="step", linewidth=2, color=colour,
            label=f"{label} (median {df[column].median():.1f} s)", zorder=2)
ax.legend(frameon=False, labelcolor=INK)
style(ax, "Example sentence vs isolated sign duration", "duration (s)")
plt.tight_layout()
plt.show()

# %%
# German sentence length
words = df["example"].str.split().str.len()

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(words, bins=range(0, int(words.max()) + 2), color=BLUE, edgecolor="white",
        linewidth=0.5, zorder=2)
style(ax, f"Words per German example sentence  (median {words.median():.0f})",
      "words", "sentences")
plt.tight_layout()
plt.show()

# %%
# topic coverage
top = df["category"].value_counts().head(15).sort_values()

fig, ax = plt.subplots(figsize=(7, 4.6))
bars = ax.barh(top.index, top.values, color=BLUE, height=0.72, zorder=2)
ax.bar_label(bars, fmt="%d", padding=3, color=INK, fontsize=8)
style(ax, f"Top 15 categories of {df['category'].nunique()}", "examples", "", axis="x")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Automatic sign-level segmentation (2023 model)
#
# Statistics over the predictions produced by `run_2023_segmentation.py`, which
# must have been run separately:
#
# ```bash
# conda activate sas2023
# python datasets/signsuisse/run_2023_segmentation.py
# ```
#
# Nothing here is gold — `SIGN_MANUAL` is still empty at this stage. These are
# the model's own outputs, useful for sanity-checking the run and for sizing the
# annotation job before it starts.

# %%
import pympi

SEGMENTATION_DIR = DATA_DIR / "example_segmentation" / "v2023_e4s_60_50"

SIGN_AUTO_TIER, SENTENCE_AUTO_TIER, SIGN_MANUAL_TIER = "SIGN_AUTO", "SENTENCE_AUTO", "SIGN_MANUAL"

clip_rows, segment_rows = [], []
eaf_paths = sorted(SEGMENTATION_DIR.glob("*.eaf")) if SEGMENTATION_DIR.is_dir() else []

if not eaf_paths:
    print(f"No .eaf files under {SEGMENTATION_DIR}\n"
          f"Run run_2023_segmentation.py first; the cells below will be empty.")
else:
    for path in eaf_paths:
        eaf = pympi.Elan.Eaf(str(path))
        example_id = int(path.stem)

        def spans(tier):
            return eaf.get_annotation_data_for_tier(tier) if tier in eaf.tiers else []

        signs, sentences = spans(SIGN_AUTO_TIER), spans(SENTENCE_AUTO_TIER)
        clip_rows.append({
            "id": example_id,
            "n_sign": len(signs),
            "n_sentence": len(sentences),
            "n_manual": len(spans(SIGN_MANUAL_TIER)),
            "sign_ms": sum(end - start for start, end, *_ in signs),
        })
        for start, end, *_ in signs:
            segment_rows.append({"id": example_id, "start_ms": start, "end_ms": end,
                                 "duration_ms": end - start})

clips = pd.DataFrame(clip_rows)
segments = pd.DataFrame(segment_rows)
print(f"read {len(clips):,} ELAN files, {len(segments):,} predicted sign segments")

# %%
if len(clips):
    # join clip duration from the metadata to compute how much of each clip is
    # covered by predicted signs
    clips = clips.merge(df[["id", "exampleVideoDuration"]], on="id", how="left")
    clips["clip_ms"] = clips["exampleVideoDuration"] * 1000
    clips["coverage"] = clips["sign_ms"] / clips["clip_ms"]

    print("=== 2023 model predictions on the DSGS test set ===\n")
    print(f"  clips                       {len(clips):,}")
    print(f"  predicted sign segments     {len(segments):,}")
    print(f"  predicted phrase segments   {int(clips['n_sentence'].sum()):,}")
    print(f"  signs per clip              mean {clips['n_sign'].mean():.1f}  "
          f"median {clips['n_sign'].median():.0f}  "
          f"range {clips['n_sign'].min()}–{clips['n_sign'].max()}")
    print(f"  clips with no sign          {int((clips['n_sign'] == 0).sum())}")
    print(f"  mean coverage of clip       {clips['coverage'].mean():.1%}")
    print(f"  SIGN_MANUAL annotations     {int(clips['n_manual'].sum())} (expected 0 before annotation)")

    summary_seg = pd.DataFrame({
        "predicted sign duration (ms)": segments["duration_ms"],
        "signs per clip": clips["n_sign"],
        "coverage of clip": clips["coverage"],
    }).describe(percentiles=[0.1, 0.5, 0.9])
    display(summary_seg.round(2))

# %%
if len(clips):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    counts_sign = clips["n_sign"].value_counts().sort_index()
    ax.bar(counts_sign.index, counts_sign.values, color=BLUE, width=0.82, zorder=2)
    ax.axvline(clips["n_sign"].median(), color=INK, linewidth=1.2, linestyle="--", zorder=3)
    style(ax, f"Predicted signs per clip  (median {clips['n_sign'].median():.0f})",
          "predicted sign segments", "clips")
    plt.tight_layout()
    plt.show()

# %%
if len(segments):
    clip_ms = segments["duration_ms"].quantile(0.99)
    values_seg = segments.loc[segments["duration_ms"] <= clip_ms, "duration_ms"]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(values_seg, bins=50, color=BLUE, edgecolor="white", linewidth=0.5, zorder=2)
    median_seg = segments["duration_ms"].median()
    ax.axvline(median_seg, color=INK, linewidth=1.2, linestyle="--", zorder=3)
    ax.annotate(f"median {median_seg:,.0f} ms", xy=(median_seg, ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9)
    style(ax, f"Predicted sign duration  (n={len(values_seg):,}, clipped at p99)", "duration (ms)")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Notes and caveats
#
# - **This is the test split, DSGS only.** The training split
#   (`metadata_train.csv`, 17,221 rows) and the LSF/LIS test portions are out of
#   scope here. If we ever want more DSGS annotation material, the training split
#   is where it would come from.
# - **Nothing is annotated at gloss level yet.** This notebook characterises the
#   raw material; the gloss spans that make it a sign-level eval set still have to
#   be produced.
# - **Domain shift is confounded with language transfer.** These are short, read,
#   studio-recorded single-sentence clips with one signer, whereas the DGS corpus
#   is 7.55-minute two-signer conversation. A score drop from DGS to SignSuisse
#   therefore cannot be attributed to DSGS-vs-DGS alone — it also reflects the
#   change in register and clip length. Worth designing a control before drawing
#   transfer conclusions.
# - **These clips are short by design** (median ~5.5 s), which makes the set a
#   useful probe for the "very short input clips" edge case in the proposal.
# - **Pose files are not inspected here.** Frame counts and fps live inside the
#   `.pose` files and would need `pose_format`, which is deliberately not in the
#   `sas` env. Worth checking before evaluation, since the segmentation model
#   consumes poses rather than video.
#
# ### Running and exporting
#
# In VS Code, open this file and run the `# %%` cells directly.
#
# To regenerate `explore.md` — run from `segment-any-sign/`. This executes the
# file top-to-bottom in a fresh kernel, so the report always matches the
# committed source, and no intermediate `.ipynb` is written:
#
# ```bash
# jupytext --to ipynb --execute datasets/signsuisse/explore.py -o - \
#   | jupyter nbconvert --stdin --to markdown --output explore --output-dir datasets/signsuisse
# ```
#
# Markdown rather than HTML because GitHub renders it inline. The plots land in
# `explore_files/`; `explore.py`, `explore.md` and that folder are all tracked.
