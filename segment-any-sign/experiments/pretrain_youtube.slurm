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
#SBATCH --output=/scratch/zifjia/segment-any-sign/slurm/%x-%j.out
#SBATCH --error=/scratch/zifjia/segment-any-sign/slurm/%x-%j.out
#
# Stage one: subtitle-supervised phrase pretraining on YouTube-SL-25.
#
#     sbatch experiments/pretrain_youtube.slurm
#
# Every flag below is spelled out even where it is the default, so an upstream
# change cannot move a hyperparameter silently. See experiments/README.md.
#
# Sizing, from measurement rather than guesswork:
#   * 50,000 optimiser steps at ~0.83 steps/s is about 17 h. The 36 h wall gives
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
# script runs, so the mkdir below only guards against a later wipe of scratch.

set -euo pipefail

REPO=/home/zifjia/segmentation/segment-any-sign
PYTHON=/home/zifjia/data/conda/envs/sas/bin/python   # no `conda` on the nodes

mkdir -p /scratch/zifjia/segment-any-sign/slurm
cd "$REPO"

echo "node       $(hostname)"
echo "job        ${SLURM_JOB_ID:-none}  cpus ${SLURM_CPUS_PER_TASK:-?}"
echo "commit     $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

srun "$PYTHON" experiments/train.py \
    --run_name 02_pretrain_youtube \
    --datasets youtube_25 \
    --val-datasets youtube_25,dgs_corpus \
    --levels phrase \
    --phrase glosses \
    --class-weights 2,20,1 \
    --select-on mean_mf1s \
    --sampling frame \
    --velocity off \
    --fps-aug off \
    --early-stop off \
    --max-steps 50000 \
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
    --frame_dropout 0.0 \
    --body_part_dropout 0.0 \
    --attn_dropout 0.0 \
    --device gpu --gpus 1 \
    --max_time 01:11:00:00
