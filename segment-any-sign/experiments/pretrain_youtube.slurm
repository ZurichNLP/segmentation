#!/bin/bash
#SBATCH --job-name=02_pretrain_youtube
#SBATCH --account=sigma.ebling.cl.uzh
#SBATCH --partition=standard
#SBATCH --constraint="H100|A100"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/%u/segment-any-sign/slurm/%x-%j.out
#SBATCH --error=/scratch/%u/segment-any-sign/slurm/%x-%j.out
#
# Stage one: subtitle-supervised phrase pretraining on YouTube-SL-25.
#
# Submit from inside segment-any-sign/, with the sas environment active: the
# job runs in the checkout it was submitted from, so a git worktree trains its
# own branch, and with the python that is on PATH at submission.
#
#     python -m paths outputs.slurm                                 # once
#     sbatch experiments/pretrain_youtube.slurm                     # 2,80,1
#     sbatch --job-name=02_pretrain_youtube_b100 \
#            experiments/pretrain_youtube.slurm 2,100,1 _b100
#     sbatch --job-name=02_pretrain_youtube_filtered \
#            experiments/pretrain_youtube.slurm 2,80,1 _filtered linear
#     sbatch --job-name=02_pretrain_youtube_b40 \
#            experiments/pretrain_youtube.slurm 2,40,1 _b40 off "weights 2,40,1"
#
# Nine positional arguments, all optional, in order. Augmentation and all three
# dropouts default to on since 2026-09-15 (run 9's recipe); pass off / 0.0 to
# turn one off:
#   1. the O,B,I class weights
#   2. a suffix for the run name
#   3. the sampling schedule (off, linear, cosine) that decays the weight of
#      videos whose subtitle timings fail DEV_FILTER. Off by default: run 4 in
#      experiments/README.md found no gain
#   4. a note for the `change` column of the row the run adds to
#      experiments/README.md when it finishes; defaults to weights and schedule
#   5. frame-rate augmentation, on or off (default on): each training window
#      resampled to 20-60 fps around 30, always 1024 frames
#      (experiments/fps_augment.py)
#   6. tempo stretch, on or off (default on): in 5% of training windows the
#      timestamps RoPE reads run at 24, 30 or 60 fps; poses and labels unchanged
#   7. frame dropout, the upper bound of each window's drop rate (default 0.15).
#      The window is widened first, so it still holds 1024 frames
#   8. body-part dropout, the chance each hand is zeroed per window (default 0.1)
#   9. attention dropout (default 0.1)
#
# Selection is on raw YouTube dev; the filtered score is reported after
# training. The suffix is what keeps parallel runs apart — same day, same
# base name means the same `dist/<name>-<date>` directory, and Lightning would
# write best-v1.ckpt into it beside the other run's checkpoint. The suffix also
# separates the W&B run and the prediction JSON, which are both named from the
# run id. Pass --job-name to sbatch to keep the SLURM logs apart as well.
#
# Every flag below is spelled out even where it is the default, so an upstream
# change cannot move a hyperparameter silently. See experiments/README.md.
#
# Sizing, from measurement rather than guesswork:
#   * 40,000 optimiser steps at ~0.83 steps/s is about 13 h. The 36 h wall gives
#     room for a slow node; `--max_time` stops the run an hour before SLURM
#     would kill it, so the final checkpoint is written rather than lost.
#   * peak GPU memory was 53 GB of 80, so either an 80 GB A100 or an H100 fits.
#   * 32 CPUs: reading is 50 ms per sample and one GPU consumes ~106 samples/s,
#     so a GPU needs ~5.3 cores. 32 leaves headroom for the validation loaders.
#   * 128 GB host memory: anonymous use is ~11 GB, the rest is reclaimable page
#     cache from the pose share. The previous 125 GB run hit its cgroup limit
#     1.4M times and reclaimed each time, never an OOM, but the extra helps.
#
# The --output directory must already exist: SLURM opens the log file before the
# script runs, so no check below can report that it is missing. Create it once
# with `python -m paths outputs.slurm`. SLURM cannot read the paths file, so the
# folder is written above with %u, SLURM's pattern for the user name: it is
# `outputs.slurm` as long as output_dir is /scratch/<user>/segment-any-sign in
# the paths file. Keep the two the same: train.py prints a warning into the log
# when the log is not in `outputs.slurm`.

set -euo pipefail

WEIGHTS="${1:-2,80,1}"     # O,B,I; every run since row 3 of the README table
SUFFIX="${2:-}"            # appended to the run name, e.g. _b100
SCHEDULE="${3:-off}"       # off | linear | cosine
FPS_AUG="${5:-on}"         # on | off (experiments/fps_augment.py)
TEMPO="${6:-on}"           # on | off (tempo stretch, experiments/fps_augment.py)
FRAME_DROPOUT="${7:-0.15}" # upper bound of the per-window drop rate; windows stay full
BODY_PART_DROPOUT="${8:-0.1}"
ATTN_DROPOUT="${9:-0.1}"
NOTE="${4:-weights ${WEIGHTS}, schedule ${SCHEDULE}, fps aug ${FPS_AUG}, tempo ${TEMPO}, frame dropout ${FRAME_DROPOUT}, body-part dropout ${BODY_PART_DROPOUT}, attention dropout ${ATTN_DROPOUT}, 40k steps}"   # README `change` column

die() { echo "pretrain_youtube.slurm: $*" >&2; exit 1; }

# the checkout the job was submitted from, so a git worktree trains its own branch
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$REPO/experiments/train.py" ] \
    || die "submit from inside segment-any-sign/: no experiments/train.py in ${REPO}"
cd "$REPO"

# sbatch passes the environment of the submission to the job, so this is the
# python of the environment that was active then. `conda activate` cannot be
# run here: there is no `conda` on the nodes.
PYTHON="$(command -v python)" \
    || die "no python on PATH: activate the sas environment before sbatch"
"$PYTHON" -c "import sign_language_segmentation, yaml" \
    || die "${PYTHON} is not the python of the sas environment: activate it before sbatch"

# every location this job reads or writes, checked as the code will check it
# and printed; the output folders are created when missing, which also guards
# against a wipe of scratch since the last job
"$PYTHON" -m paths outputs.slurm outputs.dist outputs.cache outputs.wandb \
    data.youtube_sl25.root data.public_dgs_corpus.downloads \
    || die "a location this job needs cannot be used, see the message above"
command -v nvidia-smi > /dev/null || die "no nvidia-smi: this is not a GPU node"

echo "repo       ${REPO}  ($(git rev-parse --abbrev-ref HEAD))"

echo "weights    ${WEIGHTS}   schedule ${SCHEDULE}   fps aug ${FPS_AUG}   tempo ${TEMPO}   run 02_pretrain_youtube${SUFFIX}"
echo "dropout    frame ${FRAME_DROPOUT}   body part ${BODY_PART_DROPOUT}   attention ${ATTN_DROPOUT}"
echo "node       $(hostname)"
echo "python     ${PYTHON}"
echo "job        ${SLURM_JOB_ID:-none}  cpus ${SLURM_CPUS_PER_TASK:-?}"
echo "commit     $("$PYTHON" -m experiments.provenance)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

srun "$PYTHON" experiments/train.py \
    --run_name "02_pretrain_youtube${SUFFIX}" \
    --note "${NOTE}" \
    --datasets youtube_25 \
    --val-datasets youtube_25,dgs_corpus \
    --levels phrase \
    --phrase glosses \
    --class-weights "${WEIGHTS}" \
    --sample-schedule "${SCHEDULE}" \
    --dev-filter off \
    --select-on mean_mf1s \
    --sampling frame \
    --velocity off \
    --fps-aug "${FPS_AUG}" \
    --tempo-stretch "${TEMPO}" \
    --early-stop off \
    --max-steps 40000 \
    --val-every 400 \
    --accumulate 2 \
    --num-workers 32 \
    --grad-clip 5 \
    --grad-log-every 1 \
    --eval-split validation \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --optimizer adamw-onecycle \
    --num_frames 1024 \
    --dice_loss_weight 0.0 \
    --frame_dropout "${FRAME_DROPOUT}" \
    --body_part_dropout "${BODY_PART_DROPOUT}" \
    --attn_dropout "${ATTN_DROPOUT}" \
    --device gpu --gpus 1 \
    --max_time 01:11:00:00
