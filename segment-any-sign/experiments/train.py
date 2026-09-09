"""Launch a 2026-style training run on the benchmark's own DGS clips.

Thin wrapper. The training loop, model, augmentation and checkpointing are
upstream's — this only swaps the dataset for `dgs_corpus`
([`dgs_dataset.py`](dgs_dataset.py)) so a run trains on exactly the clips,
filters and gold that [`../benchmark/`](../benchmark/) scores, and then calls
`sign_language_segmentation.train.train()` unchanged.

Every hyperparameter is upstream's `args.py`; anything not passed keeps its 2026
default **except** for the basic-baseline overrides below, each chosen so that a
later experiment turns exactly one trick back on:

    --batch_size 64          (upstream 8)      --dice_loss_weight 0   (1.5)
    --epochs 500             (200)             --frame_dropout 0      (0.15)
    --learning_rate 1e-3     (1e-3, pinned)    --body_part_dropout 0  (0.1)
    velocity off             (on, unswitchable) --attn_dropout 0      (0.1)
    fps_aug off              (on, unswitchable) early stopping off     (patience 10)

`fps_aug` stays on: upstream calls it essential, and disabling it also switches
label construction from `create_bio_from_times` to `create_bio`, which would
confound the ablation.

These flags are ours and are stripped before upstream parses:

    --phrase {glosses,sentence}   what counts as a phrase (default glosses, the
                                  benchmark's definition — see ../benchmark/)
    --dry-run                     build the data, model and one forward pass,
                                  then stop. No optimiser step, no wandb.
    --skip-stats                  skip the data report (it costs ~1 min)
    --eval-split {validation,test,none}
                                  what to score the best checkpoint on when
                                  training ends. Default validation — ablations
                                  must not touch test
    --limit N                     use only the first N annotated clips per split.
                                  Turns a dry run into seconds. Never for a
                                  reported number.

A dry run or any `--limit` run is named `_scratch_*`, so its artefacts are
self-evidently throwaway; a dry run also deletes its own run directory.
    --select-on {mean_mf1s,hm_iou}
                                  what the best checkpoint maximises. Default is
                                  the mean of sign and phrase mF1S; upstream used
                                  hm_iou.

Every run first writes a data report — split sizes, segment counts, label balance
and correctness checks — to `dist/<run>/data_stats.{json,log}`, and attaches its
scalar summary to the W&B run. A run that leaks documents between splits aborts
before training rather than producing a number nobody can trust.

It also writes `dist/<run>/run_config.json`: every hyperparameter, the command
line, the git commit, and the python/torch/GPU it ran on — enough to reproduce the run from the checkpoint alone.

The full metric set from `../metrics/` is logged on **both** train and validation
under matching names, so W&B can overlay the two curves — see
[`validation_metrics.py`](validation_metrics.py).

Validation runs **every epoch** (upstream's default), and the best checkpoint
maximises `validation_mean_mf1s` — see `--select-on`.

When training ends the best checkpoint is scored **on validation**, into
`experiments/predictions/`. Test is reserved for the one model finally reported:
`--eval-split test` writes to `benchmark/predictions/` instead. **Test runs once**, after training, on
that checkpoint — through `benchmark/`, so the number lands on the same protocol
as every row of the benchmark table. Results go to `dist/<run>/test_results.log`.

    conda activate sas
    python experiments/train.py --dry-run --device cpu --no_wandb

Note upstream's `--max_time` defaults to **30 minutes**, which will silently cut
a real run short; pass it explicitly.
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

#: Everything a run writes lives on scratch, never in the repo or on /home:
#: checkpoints are ~68 MB each and the pose caches are large. `dist/` and
#: `wandb/` in the repo are symlinks to the same root, so relative paths still
#: work; this is the one path that has to name it outright.
SCRATCH = Path("/scratch/zifjia/segment-any-sign")
CACHE_DIR = SCRATCH / "cache"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    # Take our flags out of argv first: upstream's args.py calls parse_args() at
    # import time and rejects anything it does not know.
    ours = argparse.ArgumentParser(add_help=False)
    ours.add_argument("--phrase", default="glosses", choices=["glosses", "sentence"])
    ours.add_argument("--dry-run", action="store_true")
    ours.add_argument("--skip-stats", action="store_true")
    ours.add_argument("--eval-split", default="validation",
                      choices=["validation", "test", "none"],
                      help="what to evaluate the best checkpoint on when training "
                           "ends. Default validation: ablations must not touch "
                           "test, or repeated looks leak it into model selection. "
                           "Use test only for the model you finally report")
    ours.add_argument("--limit", type=int, default=None)
    ours.add_argument("--velocity", choices=["on", "off"], default="off",
                      help="append fps-normalised velocity features (3->6 dims). "
                           "Upstream declares --velocity as store_true with "
                           "default True, so it cannot be disabled there at all")
    ours.add_argument("--sampling", choices=["frame", "video"], default="frame",
                      help="frame (default): a video is drawn in proportion to "
                           "its length, so every frame is equally likely. video: "
                           "upstream's behaviour, one window per video per epoch "
                           "regardless of length, which oversamples short videos "
                           "by orders of magnitude per frame")
    ours.add_argument("--val-every", type=int, default=None,
                      help="log all train and validation metrics every N steps. "
                           "Default: 1%% of the run, capped at one epoch so a "
                           "small dataset still gets an evaluation every epoch")
    ours.add_argument("--val-datasets", default=None,
                      help="comma-separated datasets to validate on. The FIRST is "
                           "in-domain and supplies the selection metric; the rest "
                           "are logged for information. Defaults to --datasets")
    ours.add_argument("--max-steps", type=int, default=None,
                      help="total optimiser steps, the unit that compares across "
                           "datasets — DGS gives 10 steps/epoch, YouTube ~600, so "
                           "'epochs' means nothing between them. --epochs is "
                           "derived from this so OneCycle spans exactly the run. "
                           "Default 10,000 when pretraining on youtube_25, else "
                           "5,000 (which is DGS's 500 epochs)")
    ours.add_argument("--early-stop", choices=["on", "off"], default="off",
                      help="stop when the selection metric plateaus. Off by "
                           "default: OneCycle anneals over --epochs, so stopping "
                           "early leaves the schedule near peak LR and never "
                           "visits its low-LR phase")
    ours.add_argument("--fps-aug", choices=["on", "off"], default="off",
                      help="random 25-50 fps resampling per training clip")
    ours.add_argument("--class-weights", default="auto",
                      choices=["auto", "inverse", "inverse-b", "none"],
                      help="loss class weighting. auto (default) = 2023's inverse "
                           "class frequency when dice is off, unweighted when it "
                           "is on, so the loss is always one model's or the "
                           "other's and never a hybrid. 'inverse-b' weights only "
                           "B, the boundary class, by inverse frequency and "
                           "leaves O and I flat — for a corpus too imbalanced "
                           "for inverse weighting to be safe on those two")
    ours.add_argument("--select-on", default="mean_mf1s",
                      choices=["mean_mf1s", "hm_iou"],
                      help="checkpoint selection metric (default mean_mf1s: the "
                           "mean of mF1S over the supervised levels; hm_iou is "
                           "upstream's, and needs both heads)")
    ours.add_argument("--num-workers", type=int, default=None,
                      help="dataloader workers (upstream hardcodes 8). Default "
                           "is min(32, CPUs allocated to this job): reading is "
                           "the bottleneck, not compute")
    ours.add_argument("--accumulate", type=int, default=1,
                      help="gradient accumulation steps. Multiplies the effective "
                           "batch without touching activation memory, which is "
                           "what caps batch_size: peak was 53 GB of 85 GB at "
                           "batch 64, so 128 would not fit. Note a step becomes "
                           "N optimiser-free forwards, so --max-steps counts "
                           "optimiser steps and the run gets N times longer")
    ours.add_argument("--weights-from", default=None, metavar="DATASET",
                      help="measure class weights on this dataset instead of the "
                           "one being trained on. YouTube's own inverse weights "
                           "give O 47x the weight of I and collapse the model "
                           "onto O; DGS is balanced enough for 2023's formula to "
                           "behave, so --weights-from dgs_corpus carries that "
                           "formula over to a corpus it cannot be applied to "
                           "directly")
    ours.add_argument("--levels", default="auto",
                      choices=["auto", "both", "sign", "phrase"],
                      help="which heads carry supervision. auto = phrase only "
                           "when training on youtube_25 (subtitles are phrases "
                           "and there is no gloss signal), both otherwise. A "
                           "level left out gets no loss, no metrics and no "
                           "weight in the selection metric")
    mine, rest = ours.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    # register every dataset before anything asks the registry for one
    from experiments import dgs_dataset  # noqa: F401
    from experiments import youtube_dataset  # noqa: F401

    from datasets.public_dgs_corpus import load as dgs_data
    from sign_language_segmentation.args import args

    # Upstream's args.py prints the parsed Namespace at import — before any of
    # this — so that first line still shows `datasets='all'` and the two
    # /mnt/nas GCS paths it defaults to. Neither is what we read. Correct them
    # here so the value logged to W&B is truthful, then restate the effective
    # configuration below, since the misleading line cannot be unprinted.
    args.phrase = mine.phrase
    args.limit = mine.limit
    if args.datasets == "all":
        # "all" would resolve to whatever happens to be registered; be explicit
        args.datasets = dgs_dataset.DGSCorpusDataset.dataset_name
    args.corpus = args.poses = dgs_data.BACKUP
    args.data_loader = "datasets/public_dgs_corpus/load.py (archive, native fps)"

    import sign_language_segmentation.train as upstream_train
    from sign_language_segmentation.train import _dated_run_name, train

    # A dry run or a --limit run is never a result. Prefix its name so any
    # artefact it leaves is obviously throwaway even months later, and so it can
    # never collide with a real experiment id.
    throwaway = mine.dry_run or mine.limit is not None
    if throwaway:
        args.run_name = f"_scratch_{args.run_name or 'run'}"

    # One identity everywhere: upstream derives both the run directory and the
    # W&B run name from `_dated_run_name(args.run_name)`, so the experiment id
    # that names dist/<id>-<date>/ is the same string shown in W&B and in the
    # tables. The project defaults to this repo rather than upstream's generic
    # "segmentation", so our runs do not land in someone else's project.
    if args.wandb_project == "segmentation":
        args.wandb_project = "segment-any-sign"

    # Upstream's train() instantiates whatever `PoseTaggingModel` names in its own
    # module namespace, so swapping the symbol there is enough to get the extra
    # metrics without touching upstream code or copying its training loop.
    from experiments.validation_metrics import ValidationMetricsModel
    upstream_train.PoseTaggingModel = ValidationMetricsModel

    # Validate on a list of datasets, not one. Upstream builds a single DEV
    # loader; wrapping `get_dataloader` turns that call into one loader per named
    # dataset, which Lightning accepts as a list and reports with a
    # `dataloader_idx` the model maps back to a name.
    val_names = tuple((mine.val_datasets or args.datasets).split(","))
    ValidationMetricsModel.val_dataset_names = val_names
    args.val_datasets = ",".join(val_names)
    _get_dataloader = upstream_train.get_dataloader

    def get_dataloader_multi(split, dataset_names, args, **kwargs):
        if str(split) == "dev":
            # The worker count applies here too: validation reads *whole* videos,
            # 167 of them per pass, so it is the most IO-bound part of the run.
            # These loaders run inside one validation pass, so they share the one
            # budget rather than each taking it.
            built = [_get_dataloader(split, name, args, **kwargs)
                     for name in val_names]
            shares = _split_workers([loader.dataset for loader in built], workers)
            return [_report_workers(_with_workers(loader, share), f"dev/{name}")
                    for loader, share, name in zip(built, shares, val_names)]
        loader = _get_dataloader(split, dataset_names, args, **kwargs)
        if str(split) == "train" and mine.sampling == "frame":
            return _report_workers(frame_uniform_loader(loader, workers),
                                   f"train/{dataset_names}")
        return _report_workers(_with_workers(loader, workers), f"{split}/{dataset_names}")

    # upstream hardcodes num_workers=8; reading is the bottleneck, so this is
    # the single most valuable knob on a many-core node. Affinity, not
    # os.cpu_count(): under SLURM the latter reports the whole machine, so a
    # 4-CPU allocation on a 128-core node would spawn 32 workers and thrash.
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else (os.cpu_count() or 8)
    workers = mine.num_workers or max(1, min(32, available))
    upstream_train.get_dataloader = get_dataloader_multi

    # Validate on a step schedule, not an epoch one. `check_val_every_n_epoch=None`
    # is what lets `val_check_interval` count steps *across* epoch boundaries —
    # without it Lightning requires the interval to fit inside one epoch, which
    # 100 steps does not on DGS (10 steps per epoch). Wrapped in a shim rather
    # than mutating the pytorch_lightning module itself.
    class _TrainerShim:
        def __init__(self, module, extra):
            self._module, self._extra = module, extra

        def __getattr__(self, name):
            return getattr(self._module, name)

        def Trainer(self, *args, **kwargs):
            merged = {**kwargs, **self._extra}
            # callbacks are *added* to upstream's, never substituted for them:
            # replacing the list would drop ModelCheckpoint and EarlyStopping
            extra_callbacks = self._extra.get("callbacks") or []
            if extra_callbacks:
                merged["callbacks"] = list(kwargs.get("callbacks") or []) + \
                    list(extra_callbacks)
            return self._module.Trainer(*args, **merged)

    # Everything the user sets is in *gradient* steps. Lightning's
    # `val_check_interval` is the one exception: it counts training batches
    # (`total_batch_idx`), so under accumulation it must be scaled or validation
    # would fire every `val_every / accumulate` gradient steps. `global_step`,
    # which drives checkpointing, early stopping and `metrics_every_n_steps`,
    # already counts optimiser steps, and OneCycle is sized in them too.
    # `val_check_interval` is filled in below, once `--val-every` is resolved
    # against the step budget. The dict is captured by reference.
    trainer_extra = {"val_check_interval": None,
                     "check_val_every_n_epoch": None,
                     "accumulate_grad_batches": mine.accumulate}
    ValidationMetricsModel.accumulate = mine.accumulate
    upstream_train.pl = _TrainerShim(upstream_train.pl, trainer_extra)

    # Subtitle cues are translation units, so YouTube supervises phrases only
    # and the sign head must be left out rather than fed empty labels.
    pretraining = "youtube_25" in args.datasets.split(",")
    levels = mine.levels
    if levels == "auto":
        levels = "phrase" if pretraining else "both"
    ValidationMetricsModel.levels = {"both": ("sign", "sentence"),
                                     "sign": ("sign",),
                                     "phrase": ("sentence",)}[levels]
    args.levels = levels
    if levels != "both" and mine.select_on == "hm_iou":
        raise SystemExit("--select-on hm_iou needs both heads; it is the "
                         "harmonic mean of sign and phrase IoU. Use "
                         "--select-on mean_mf1s with --levels " + levels)
    if mine.max_steps is None:
        # YouTube is ~40x the data, so it gets a longer budget by default
        mine.max_steps = 20000 if pretraining else 5000

    # Defaults describe the *basic baseline*: every optional trick off, so each
    # later experiment turns exactly one back on. This deliberately departs from
    # upstream's defaults — see README.md — and each departure is listed here.
    #
    # Batch 64: measured activations are 0.33 GiB per sample at 1024 frames, so
    # ~36 GiB peak. Fine on an 80GB A100, tight on a 40GB one. It also leaves
    # only ~9 steps/epoch over 586 clips, which is why epochs is 500.
    # lr 1e-3 happens to be upstream's default too, but it is pinned here because
    # the sweep chose it: 3e-4 / 5e-4 / 1e-3 sit within 0.007 mean mF1S of each
    # other, and 1e-3 leads on IoU. Pinning keeps an upstream change from moving
    # it silently. See README.md.
    basic = {"batch_size": 64, "epochs": 500, "learning_rate": 1e-3,
             "dice_loss_weight": 0.0, "frame_dropout": 0.0,
             "body_part_dropout": 0.0, "attn_dropout": 0.0}
    for key, value in basic.items():
        if f"--{key}" not in rest:
            setattr(args, key, value)

    # Budget in steps, not epochs. OneCycle is sized by epochs x steps_per_epoch,
    # so epochs is derived to make the schedule span exactly the requested steps —
    # otherwise a run stops mid-anneal, which is what early stopping used to do.
    if ("--epochs" not in rest and mine.max_steps) or mine.val_every is None:
        clips = dataset_size(args)
        # optimiser steps, not batches: --max-steps counts gradient updates, and
        # with accumulation a batch is no longer an update
        batches_per_epoch = math.ceil(clips / args.batch_size)
        steps_per_epoch = max(1, math.ceil(batches_per_epoch / mine.accumulate))

        if "--epochs" not in rest and mine.max_steps:
            args.epochs = max(1, math.ceil(mine.max_steps / steps_per_epoch))
            print(f"\n{clips:,} training clips / batch {args.batch_size} = "
                  f"{steps_per_epoch:,} steps per epoch"
                  f"  ->  {args.epochs} epochs for ~{mine.max_steps:,} steps")

        # Evaluate 100 times over the run, but never less often than once per
        # epoch: 1% of DGS's 5,000 steps is 50, which would be five epochs.
        if mine.val_every is None:
            mine.val_every = max(1, min(mine.max_steps // 100, steps_per_epoch))

    trainer_extra["val_check_interval"] = mine.val_every * mine.accumulate

    # sample train metrics ten times per evaluation, so both curves are smooth
    ValidationMetricsModel.metrics_every_n_steps = max(1, mine.val_every // 10)

    # patience tracks the budget rather than being a fixed number of epochs
    if "--patience" not in rest:
        args.patience = max(1, round(0.1 * args.epochs))

    args.velocity = mine.velocity == "on"
    args.fps_aug = mine.fps_aug == "on"

    # `fps_aug` upstream controls *two* things: the resampling, and which label
    # builder runs (`create_bio_from_times` when on, `create_bio` when off).
    # Turning augmentation off would therefore silently change the gold, and
    # disagree with the timestamp-based gold `benchmark/predict_dgs_2026.py`
    # builds at eval. Make the label rule the same either way, so the flag means
    # only what its name says.
    import numpy as np

    from sign_language_segmentation.datasets import common as _common
    from sign_language_segmentation.utils.bio import create_bio_from_times

    def _create_bio_from_times_shim(annotations, num_frames, fps):
        times_ms = np.arange(num_frames, dtype=np.float32) / fps * 1000
        return create_bio_from_times(annotations, times_ms)

    _common.create_bio = _create_bio_from_times_shim

    # Pose windows are read one frame at a time, and Python's default 8 KB
    # buffer turns that into a syscall per frame. On the network share each
    # costs a round trip, so a 1024-frame window took 0.33 s to move 6.8 MB —
    # twenty times slower than the share's 392 MB/s. A 4 MB buffer coalesces
    # them: 0.33 s -> 0.031 s, measured. `open` is resolved from module globals
    # before builtins, so binding it here reaches every read in that module and
    # nothing else.
    _common.open = functools.partial(builtins.open, buffering=4 * 1024 * 1024)

    # Upstream always installs EarlyStopping with `patience`; making patience
    # exceed the epoch budget is how it is disabled without patching the loop.
    if mine.early_stop == "off":
        args.patience = args.epochs + 1

    # 2026 dropped 2023's class weighting when it added the Dice term, so with
    # Dice off the loss would otherwise be a plain unweighted NLL — neither
    # model's. Restore the weighting in that case.
    weighting = mine.class_weights
    if weighting == "auto":
        weighting = "inverse" if args.dice_loss_weight == 0 else "none"
    if weighting in ("inverse", "inverse-b"):
        ValidationMetricsModel.class_weights = inverse_class_weights(
            args, ValidationMetricsModel.levels, scheme=weighting,
            source=mine.weights_from)
    args.class_weighting = weighting
    args.class_weights_from = mine.weights_from or args.datasets.split(",")[0]

    report_effective_config(args, mine, _dated_run_name(args.run_name))

    run_dir = Path("dist") / _dated_run_name(args.run_name)

    # Upstream names the directory `<run_name>-<YYYY.MM.DD>`, so a second run with
    # the same id on the same day lands in it — and Lightning writes best-v1.ckpt
    # beside the first run's best.ckpt rather than overwriting, silently mixing
    # two models in one directory. Refuse instead: one id, one run, one directory.
    if not mine.dry_run and list(run_dir.glob("*.ckpt")):
        raise SystemExit(
            f"{run_dir} already holds checkpoints. Give --run_name a new "
            f"experiment id (<NN>_<slug>) rather than reusing this one.")
    # the data report describes the full corpus, so it is meaningless under --limit
    # data_stats reads the DGS loader directly, so it only describes a DGS run;
    # for any other dataset it would print numbers from a corpus not being used
    if (not mine.skip_stats and mine.limit is None
            and "dgs_corpus" in args.datasets.split(",")):
        report = write_data_report(run_dir, phrase=mine.phrase)
        # flat scalars ride along into W&B via upstream's log_hyperparams
        for split, s in report["splits"].items():
            for key in ("documents", "videos", "hours", "signs", "phrases",
                        "unannotated_videos"):
                setattr(args, f"data_{split}_{key}", s[key])

    if mine.dry_run:
        existed = run_dir.exists()
        try:
            dry_run(args)
        finally:
            # leave nothing behind: a dry run proves the wiring, it is not a result
            if not existed and run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
                print(f"cleaned up {run_dir}")
        return

    # `validation_hm_iou` is logged inside upstream's `validation_step`, which
    # ValidationMetricsModel overrides and never calls — monitoring that name
    # would watch a metric nothing produces. Ours is logged in its place.
    monitor = {"mean_mf1s": "validation_mean_mf1s",
               "hm_iou": "validation_hm_iou_ours"}[mine.select_on]
    print(f"  selection    {monitor} (max)\n")

    write_run_config(run_dir, args, mine, monitor)

    # panels follow the *running best* model: redrawn only when `monitor`
    # improves, which is the same test ModelCheckpoint applies, so the figure in
    # W&B always belongs to the checkpoint on disk
    if not args.no_wandb:
        from experiments.plot_callback import SegmentationPlotCallback
        trainer_extra["callbacks"] = [SegmentationPlotCallback(
            run_dir, val_names, monitor, phrase=mine.phrase)]

    # train() takes monitor_metric, so both ModelCheckpoint and EarlyStopping
    # follow it without patching anything
    train(monitor_metric=monitor)

    if mine.eval_split != "none":
        evaluate_best_checkpoint(run_dir, phrase=mine.phrase, split=mine.eval_split)


def _with_workers(loader, workers: int | None):
    """Rebuild a DataLoader with a different worker count.

    Upstream hardcodes `num_workers=8` inside its factory, and a DataLoader's
    worker count cannot be changed after construction, so the loader is rebuilt
    around the same dataset. Everything else is copied from the original.
    """
    from torch.utils.data import DataLoader

    # never more workers than there are items to load: the DGS dev set is 12
    # clips, and 32 workers there would leave 20 processes holding memory and
    # doing nothing for the life of the run
    workers = min(workers or 0, len(loader.dataset))
    if not workers or workers == loader.num_workers:
        return loader
    # persistence and prefetching are copied from the loader upstream built, so
    # raising the worker count changes only the worker count
    persistent = bool(getattr(loader, "persistent_workers", False)) and workers > 0
    return DataLoader(loader.dataset, batch_size=loader.batch_size,
                      sampler=loader.sampler, collate_fn=loader.collate_fn,
                      num_workers=workers, persistent_workers=persistent,
                      prefetch_factor=getattr(loader, "prefetch_factor", 4),
                      pin_memory=getattr(loader, "pin_memory", True))


def _dataset_frames(dataset) -> float:
    """Total frames behind a loader, the honest measure of how much it must read."""
    from torch.utils.data import ConcatDataset

    if isinstance(dataset, ConcatDataset):
        return sum(_dataset_frames(part) for part in dataset.datasets)
    items = getattr(dataset, "items", None)
    if not items:
        return float(len(dataset))
    return float(sum(item.get("total_frames", 1) for item in items))


def _split_workers(datasets, budget: int) -> list[int]:
    """Divide one worker budget across loaders that run at the same time.

    The validation sets are iterated back to back inside one validation pass, so
    their workers coexist; handing each the full budget would oversubscribe the
    CPUs. The split is by **frames**, since reading is what workers do and a
    corpus of few long videos costs as much as one of many short ones.

    Two guards. Every set gets at least half of an equal share, so a small set is
    never left to serialise behind one worker — with the 32-CPU budget and two
    dev sets that floor is 8. And no set gets more workers than it has clips.

    On our sets: YouTube dev holds 85% of the frames and DGS 15%, so frames alone
    would give 27 and 5; the floor lifts DGS to 8 and YouTube takes 24.

    One case does not fit the budget: every loader keeps at least one worker,
    since zero means loading synchronously in the main process. So a budget
    smaller than the number of loaders returns one each and oversubscribes by the
    difference. That needs an allocation of one or two CPUs to happen.
    """
    n = len(datasets)
    if n == 0 or budget <= 0:
        return [0] * n
    if n == 1:
        return [min(budget, max(1, len(datasets[0])))]

    caps = [max(1, len(d)) for d in datasets]
    floor = max(1, budget // (2 * n))
    weights = [_dataset_frames(d) for d in datasets]
    total = sum(weights) or 1.0

    share = [min(caps[i], max(floor, round(budget * weights[i] / total)))
             for i in range(n)]
    # Rounding and the floor together can overshoot the budget. Settle it on the
    # set that currently holds the most workers, where one either way changes the
    # least in relative terms — taking it from the smallest set instead would
    # just undo the floor that was the point.
    while sum(share) > budget:
        i = max(range(n), key=lambda j: share[j])
        if share[i] <= 1:
            break
        share[i] -= 1
    while sum(share) < budget:
        candidates = [j for j in range(n) if share[j] < caps[j]]
        if not candidates:
            break
        share[max(candidates, key=lambda j: weights[j])] += 1
    return share


def _report_workers(loader, label: str):
    """Say how many workers each loader got, so the count is never a guess."""
    print(f"  loader {label:<28} {len(loader.dataset):>6,} clips  "
          f"{loader.num_workers:>2} workers")
    return loader


def frame_uniform_loader(loader, workers: int | None = None):
    """Redraw the training loader so each *frame* is equally likely to be seen.

    Upstream shuffles the clip list, so one 1024-frame window is drawn per video
    per epoch whatever its length — a 90-minute video and a 30-second one
    contribute equally, which oversamples short videos enormously per frame.
    Weighting each video by its frame count fixes that. Sampling is with
    replacement, and the number of draws per epoch is unchanged, so steps per
    epoch and the schedule derived from them stay the same.
    """
    from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

    def lengths(dataset):
        if isinstance(dataset, ConcatDataset):
            return [n for part in dataset.datasets for n in lengths(part)]
        return [float(item["total_frames"]) for item in dataset.items]

    weights = lengths(loader.dataset)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                    replacement=True)
    print(f"  frame-uniform sampling over {len(weights):,} videos "
          f"({min(weights):,.0f}-{max(weights):,.0f} frames each)")
    return DataLoader(loader.dataset, batch_size=loader.batch_size,
                      sampler=sampler, collate_fn=loader.collate_fn,
                      # capped like _with_workers: --limit can leave fewer clips
                      # than workers
                      num_workers=min(workers or loader.num_workers,
                                      len(loader.dataset)),
                      persistent_workers=loader.persistent_workers,
                      prefetch_factor=loader.prefetch_factor,
                      pin_memory=loader.pin_memory)


def dataset_size(args) -> int:
    """Number of training clips, for turning a step budget into epochs."""
    from sign_language_segmentation.datasets.common import Split, build_datasets

    return len(build_datasets(names=args.datasets, split=Split.TRAIN, args=args,
                              num_frames=args.num_frames, velocity=args.velocity,
                              fps_aug=args.fps_aug, frame_dropout=0.0,
                              body_part_dropout=0.0))


#: With `--class-weights inverse-b`, how much more an I frame is worth than an O
#: frame. At 1.0 the two are equal, so B and I keep exactly the relationship the
#: corpus gives them and the only change from inverse weighting is that O is
#: brought down from its own inverse-frequency value. That value is what breaks a
#: YouTube run: at 2% O it is 47, and the model then calls 78% of frames O.
IO_RATIO = 1.0


def inverse_class_weights(args, levels, cache_dir: Path = None,
                          scheme: str = "inverse", source: str = None) -> dict:
    """2023's per-level inverse class frequency, counted over the whole corpus.

    v2023 counted classes over the entire training set and used `total / count[i]`
    as each class's weight. This does the same, exactly: every training clip, not
    a sample of random windows. It used to draw 50 windows, which put the phrase B
    rate anywhere between 0.32% and 0.51% run to run — weights of 315 against 195
    for one corpus.

    Counted from the spans arithmetically rather than by decoding poses: a segment
    contributes one B and `length - 1` I frames, and everything else is O. That is
    the same label rule `create_bio_from_times` applies to within a frame, and it
    turns an hour of decoding into a second of counting.

    With frame-uniform sampling every frame is equally likely to be drawn, so the
    corpus distribution is exactly what training windows converge to.

    `source` measures on a different corpus than the one being trained on. That
    is not a hack for its own sake: 2023's formula needs a corpus where O is
    common, and YouTube's 2% O breaks it, while DGS's 57% O is exactly the
    balance the formula was written against. The weights carried over are then
    deterministic and use 2023's rule unchanged.

    Cached on scratch, keyed by dataset, level set, phrase definition and the
    clip list itself, so a changed split can never silently reuse old weights.

    Two schemes. `inverse` is 2023's, `total / count` for every class. `inverse-b`
    keeps the corpus relationship between B and I — `count_I / count_B`, which is
    what inverse frequency gives those two — and only pulls O down, to
    `IO_RATIO` against I's 1.

    Use `inverse-b` when O is rare. Inverse weighting equalises the three classes'
    contribution to the loss, which is harmless on DGS (57% O, so O and I both
    land near 1.9 and only B is boosted) and destroys a YouTube run: at 2% O it
    gives O a weight of 47 against I's 1. Measured on a run: the model called
    78% of dev frames O where the gold on those clips is 14%, and phrase IoU
    was 0.002 against the 0.85 the same setup reaches without it.

    Returns one weight per BIO class in upstream's order (UNK, O, B, I). UNK gets
    0: it marks padding, which the loss masks out anyway.
    """
    import hashlib

    from sign_language_segmentation.datasets.common import (DATASET_REGISTRY,
                                                            Split)
    from sign_language_segmentation.utils.bio import BIO

    name = source or args.datasets.split(",")[0]
    dataset = DATASET_REGISTRY[name].from_args(
        Split.TRAIN, args, num_frames=args.num_frames, velocity=args.velocity,
        fps_aug=False, frame_dropout=0.0, body_part_dropout=0.0)

    signature = hashlib.sha256(
        repr([(item["id"], item["total_frames"], len(item["glosses"]),
               len(item["sentences"])) for item in dataset.items]
             + [name, tuple(levels), getattr(args, "phrase", ""), scheme,
                IO_RATIO]
             ).encode()).hexdigest()[:16]
    def warn_if_o_dominates(weights: dict) -> None:
        """The trap that cost one run: inverse weighting on a corpus with little
        O pushes the model to predict O everywhere. Checked on the weights
        themselves, not while counting, so a cached read is guarded too — the
        cache persists, so a warning that only fired on a miss would never fire
        again."""
        for level, w in weights.items():
            if scheme != "inverse":
                return
            if w[BIO["O"]] > 10 * max(w[BIO["I"]], 1e-9):
                print(f"  WARNING {name}/{level}: inverse weighting gives O "
                      f"{w[BIO['O']] / w[BIO['I']]:.0f}x the weight of I. This "
                      f"collapses the model onto O. Use --class-weights "
                      f"inverse-b, or --weights-from a corpus with more O.")

    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_path = cache_dir / f"class_weights_{name}_{scheme}_{signature}.json"
    if cache_path.exists():
        weights = json.loads(cache_path.read_text())
        print(f"class weights from cache {cache_path.name}: "
              + "  ".join(f"{level} B {weights[level][BIO['B']]:.1f}"
                          for level in weights))
        warn_if_o_dominates(weights)
        return weights

    spans_for = {"sign": "glosses", "sentence": "sentences"}
    weights = {}
    for level in levels:
        b = i = total = 0
        for item in dataset.items:
            frames = int(item["total_frames"])
            total += frames
            for span in item[spans_for[level]]:
                # milliseconds in, frames out; a one-frame span is B alone
                length = max(1, round((span["end"] - span["start"])
                                      / 1000 * item["fps"]))
                b += 1
                i += length - 1
        o = max(0, total - b - i)
        if o == 0:
            # spans covering every frame would give O a weight of 0 under
            # `inverse`, silently removing the class from the loss
            print(f"  WARNING {name}/{level}: no O frames at all — spans cover "
                  f"the whole corpus, or they overlap enough to look like it")
        counts = {BIO["O"]: o, BIO["B"]: b, BIO["I"]: i}
        if scheme == "inverse-b":
            # B by inverse frequency, anchored so I would be 1; O and I by hand
            by_class = {BIO["O"]: 1.0, BIO["I"]: IO_RATIO,
                        BIO["B"]: (i / b) if b else 1.0}
        else:
            by_class = {index: (total / counts[index]) if counts[index] else 0.0
                        for index in counts}
        weights[level] = [0.0 if key == "UNK" else by_class[index]
                          for key, index in BIO.items()]
        print(f"class weights {name}/{level} over {total:,} frames: "
              + "  ".join(f"{key} {counts[index] / max(total, 1):.4%} -> "
                          f"{weights[level][index]:.1f}"
                          for key, index in BIO.items() if key != "UNK"))

    warn_if_o_dominates(weights)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(weights))
    print(f"cached to {cache_path}")
    return weights


def report_effective_config(args, mine, run_name: str) -> None:
    """Print what this run will actually use, after every override is applied.

    Upstream's `args.py` prints the parsed Namespace at *import* time — before any
    of our overrides exist — so that first line reports upstream's defaults and is
    actively misleading about what is running. It cannot be suppressed from here,
    so this restates the truth after the fact.
    """
    from datasets.public_dgs_corpus import load as dgs_data

    print("\neffective configuration  (supersedes the `Arguments:` line above,"
          "\n                          which upstream prints before any override)"
          f"\n  data        {args.datasets} via {args.data_loader}"
          f"\n              {dgs_data.BACKUP}"
          f"\n  phrase gold {mine.phrase}"
          f"\n  run/wandb   {run_name}  ->  project {args.wandb_project}"
          f"\n  levels      {args.levels}"
          + f"\n  validate on {args.val_datasets}  (first is in-domain, selects the checkpoint)"
          + f"\n  training    batch {args.batch_size}"
          + (f" x {mine.accumulate} accumulated = {args.batch_size * mine.accumulate}"
             if mine.accumulate > 1 else "")
          + f"  epochs {args.epochs}  "
          f"patience {args.patience}  lr {args.learning_rate:g}  {args.optimizer}"
          f"\n  tricks OFF  dice {args.dice_loss_weight:g}  "
          f"frame_dropout {args.frame_dropout:g}  "
          f"body_part_dropout {args.body_part_dropout:g}  "
          f"attn_dropout {args.attn_dropout:g}  velocity {args.velocity}"
          f"\n  tricks ON   fps_aug {args.fps_aug}  num_frames {args.num_frames}"
          f"\n  sampling    {mine.sampling}-uniform"
          f"\n  validate    every {mine.val_every} steps"
          f"\n  schedule    ~{mine.max_steps:,} steps = {args.epochs} epochs, early stop "
          f"{'off (OneCycle runs to completion)' if mine.early_stop == 'off' else f'patience {args.patience}'}"
          f"\n  loss        " + {
              "none": "NLL, unweighted",
              "inverse": "NLL + inverse class weights (2023)",
              "inverse-b": f"NLL + inverse B weight, O 1.0 / I {IO_RATIO}",
          }[args.class_weighting]
          + (f", measured on {args.class_weights_from}"
             if args.class_weights_from != args.datasets.split(",")[0] else "")
          + f"{'' if args.dice_loss_weight == 0 else f' + dice {args.dice_loss_weight:g}'}"
          + (f"\n  LIMIT       {args.limit} clips per split — NOT a reportable run"
             if args.limit else ""))


def write_run_config(run_dir: Path, args, mine, monitor: str) -> None:
    """Record everything needed to reproduce this run, beside its checkpoints.

    Every hyperparameter (upstream's and ours), the exact command line, the git
    commit and whether the tree was dirty, and the software/hardware it ran on.
    A checkpoint whose settings cannot be recovered is not a result.
    """
    import platform
    import subprocess

    def git(*cmd: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent),
                                            *cmd], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    torch_version = device_name = "unavailable"
    try:
        import torch
        torch_version = torch.__version__
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    config = {
        "run_id": run_dir.name,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable.split("/")[-1], *sys.argv]),
        "monitor_metric": monitor,
        "ours": {"phrase": mine.phrase, "velocity": mine.velocity,
                 "select_on": mine.select_on, "limit": mine.limit},
        "args": {k: v for k, v in sorted(vars(args).items())},
        "code": {"commit": git("rev-parse", "HEAD"),
                 "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
        "environment": {"python": platform.python_version(), "torch": torch_version,
                        "host": platform.node(), "gpu": device_name},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"wrote {run_dir}/run_config.json  (commit {config['code']['commit'][:8]})")


def write_data_report(run_dir: Path, phrase: str) -> dict:
    """Write the data report beside the checkpoints, and abort on leakage."""
    from experiments import data_stats

    report = data_stats.collect(phrase=phrase)
    text = data_stats.format_report(report)
    print(text)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data_stats.json").write_text(json.dumps(report, indent=2))
    (run_dir / "data_stats.log").write_text(text + "\n")
    print(f"\nwrote {run_dir}/data_stats.json and .log\n")

    if leakage := report["problems"]["leakage"]:
        raise SystemExit(f"ABORT — documents shared between splits: {leakage}")
    return report


def evaluate_best_checkpoint(run_dir: Path, phrase: str, split: str) -> None:
    """Evaluate the best checkpoint, once, through the benchmark's own scripts.

    Shells out to `benchmark/predict_dgs_2026.py` and `score.py` rather than
    scoring inline, so a run's number and a table's number come from the same
    code and cannot drift.

    Validation predictions go to `experiments/predictions/`, test predictions to
    `benchmark/predictions/` — separate directories so that
    `score.py benchmark/predictions/*.json` can never silently mix a dev score
    into the benchmark table.
    """
    import subprocess

    # Lightning appends -v1, -v2 rather than overwriting, so a second run with
    # the same name and date leaves the *earlier* run's best.ckpt in place.
    # Take the newest and say which, or a rerun silently evaluates the old model.
    candidates = sorted(run_dir.glob("best*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print(f"no best*.ckpt in {run_dir} — skipping test")
        return
    checkpoint = candidates[-1]
    if len(candidates) > 1:
        print(f"note: {len(candidates)} checkpoints here, testing the newest "
              f"({checkpoint.name}); others: {[c.name for c in candidates[:-1]]}")

    here = Path(__file__).resolve().parent
    # the experiment id, without the date upstream appends to the directory:
    # one id names the run, the predictions and both tables
    run_id = run_dir.name.rsplit("-", 1)[0]
    predictions = (here.parent / "benchmark" / "predictions" / f"{run_id}.json"
                   if split == "test" else
                   here / "predictions" / f"{run_id}.json")
    predictions.parent.mkdir(parents=True, exist_ok=True)
    steps = [
        [sys.executable, str(here.parent / "benchmark" / "predict_dgs_2026.py"),
         "--split", split, "--model", str(checkpoint), "--phrase", phrase,
         "--label", run_id, "--out", str(predictions)],
        [sys.executable, str(here.parent / "benchmark" / "score.py"), str(predictions)],
    ]

    output = []
    for step in steps:
        print(f"\n$ {' '.join(step)}")
        result = subprocess.run(step, capture_output=True, text=True)
        print(result.stdout or result.stderr)
        output.append(result.stdout or result.stderr)
        if result.returncode != 0:
            print(f"test step failed ({result.returncode})")
            break

    (run_dir / f"{split}_results.log").write_text("\n".join(output))
    print(f"wrote {run_dir}/{split}_results.log")


def dry_run(args) -> None:
    """Build data and model, run one forward pass, report shapes. No training.

    Checks the parts a wrapper can actually break — the item list, windowing,
    collation, and that the pose dimensions the dataset produces match what the
    model expects — without burning a GPU allocation to find out.
    """
    import torch

    import sign_language_segmentation.train as upstream_train

    from sign_language_segmentation.datasets.common import Split
    # the subclass we actually train with, so a dry run exercises our loss —
    # levels, class weights and the Dice guard — not just upstream's forward
    from experiments.validation_metrics import ValidationMetricsModel

    # the *patched* factory, so the dry run sees what training sees:
    # frame-uniform sampling, one loader per validation set, and our worker
    # counts. Importing common.get_dataloader directly tested none of it.
    loaders = {}
    for split, batch_size in ((Split.TRAIN, args.batch_size), (Split.DEV, 1)):
        loader = upstream_train.get_dataloader(split=split,
                                               dataset_names=args.datasets,
                                               args=args, batch_size=batch_size,
                                               persistent_workers=False)
        # the dev split comes back as one loader per validation set
        loaders[split] = loader[0] if isinstance(loader, list) else loader
        print(f"{split}: {len(loaders[split].dataset)} clips, "
              f"{len(loaders[split])} batches of {batch_size}")

    datum = loaders[Split.TRAIN].dataset[0]
    joints, dims = datum["pose"].shape[1:3]
    print(f"\nfirst clip     pose {tuple(datum['pose'].shape)}  "
          f"(joints={joints}, dims={dims})")
    for level in ("sign", "sentence"):
        bio = datum["bio"][level]
        counts = {int(v): int((bio == v).sum()) for v in bio.unique()}
        print(f"  {level:<9} BIO counts {counts}  (UNK=0, O=1, B=2, I=3)")

    batch = next(iter(loaders[Split.TRAIN]))
    print(f"\nbatch          pose {tuple(batch['pose'].shape)}  "
          f"lengths {batch['lengths'].tolist()}")

    model = ValidationMetricsModel(
        pose_dims=(joints, dims), hidden_dim=args.hidden_dim,
        encoder_depth=args.encoder_depth, learning_rate=args.learning_rate,
        steps_per_epoch=len(loaders[Split.TRAIN]), max_epochs=args.epochs,
        dice_loss_weight=args.dice_loss_weight, optimizer=args.optimizer,
        attn_nhead=args.attn_nhead, attn_ff_mult=args.attn_ff_mult,
        attn_dropout=args.attn_dropout, fps_aug=args.fps_aug,
        frame_dropout=args.frame_dropout, num_frames=args.num_frames)
    print(f"parameters     {sum(p.numel() for p in model.parameters()):,}")

    model.eval()
    with torch.no_grad():
        out = model(batch["pose"], timestamps=batch.get("timestamps"))
    print(f"forward        sign {tuple(out['sign'].shape)}  "
          f"sentence {tuple(out['sentence'].shape)}")

    loss = model.step(batch, name="dry_run")
    print(f"levels         {ValidationMetricsModel.levels}"
          f"  class weights {'set' if ValidationMetricsModel.class_weights else 'none'}")
    print(f"loss           {float(loss):.4f}")
    print("\ndry run OK — data, model and loss all wire up")


if __name__ == "__main__":
    main()
