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
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

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
                      choices=["auto", "inverse", "none"],
                      help="loss class weighting. auto (default) = 2023's inverse "
                           "class frequency when dice is off, unweighted when it "
                           "is on, so the loss is always one model's or the "
                           "other's and never a hybrid")
    ours.add_argument("--select-on", default="mean_mf1s",
                      choices=["mean_mf1s", "hm_iou"],
                      help="checkpoint selection metric (default mean_mf1s: the "
                           "mean of sign and phrase mF1S; hm_iou is upstream's)")
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
        if str(split) != "dev":
            return _get_dataloader(split, dataset_names, args, **kwargs)
        return [_get_dataloader(split, name, args, **kwargs) for name in val_names]

    upstream_train.get_dataloader = get_dataloader_multi

    # only the phrase head has subtitle supervision
    pretraining = "youtube_25" in args.datasets.split(",")
    if pretraining:
        ValidationMetricsModel.levels = ("sentence",)
    if mine.max_steps is None:
        # YouTube is ~40x the data, so it gets a longer budget by default
        mine.max_steps = 10000 if pretraining else 5000

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
    if "--epochs" not in rest and mine.max_steps:
        clips = dataset_size(args)
        steps_per_epoch = math.ceil(clips / args.batch_size)
        args.epochs = max(1, math.ceil(mine.max_steps / steps_per_epoch))
        print(f"\n{clips:,} training clips / batch {args.batch_size} = "
              f"{steps_per_epoch:,} steps per epoch"
              f"  ->  {args.epochs} epochs for ~{mine.max_steps:,} steps")

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
    if weighting == "inverse":
        ValidationMetricsModel.class_weights = inverse_class_weights(args)
    args.class_weighting = weighting

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

    monitor = {"mean_mf1s": "validation_mean_mf1s",
               "hm_iou": "validation_hm_iou"}[mine.select_on]
    print(f"  selection    {monitor} (max)\n")

    write_run_config(run_dir, args, mine, monitor)

    # train() takes monitor_metric, so both ModelCheckpoint and EarlyStopping
    # follow it without patching anything
    train(monitor_metric=monitor)

    if mine.eval_split != "none":
        evaluate_best_checkpoint(run_dir, phrase=mine.phrase, split=mine.eval_split)


def dataset_size(args) -> int:
    """Number of training clips, for turning a step budget into epochs."""
    from sign_language_segmentation.datasets.common import Split, build_datasets

    return len(build_datasets(names=args.datasets, split=Split.TRAIN, args=args,
                              num_frames=args.num_frames, velocity=args.velocity,
                              fps_aug=args.fps_aug, frame_dropout=0.0,
                              body_part_dropout=0.0))


def inverse_class_weights(args, samples: int = 50, seed: int = 0) -> dict:
    """2023's per-level inverse class frequency, measured on training windows.

    v2023 counted classes over the whole training set and used `total / count[i]`
    as each class's weight. We sample windows instead — the balance depends on
    windowing and augmentation, so it has to be measured as the model sees it,
    and a full pass would decode every clip at every startup.

    Returns one weight per BIO class in upstream's order (UNK, O, B, I). UNK gets
    0: it marks padding, which the loss masks out anyway.
    """
    from collections import Counter

    import numpy as np

    from sign_language_segmentation.datasets.common import Split

    from experiments.dgs_dataset import DGSCorpusDataset
    from sign_language_segmentation.utils.bio import BIO

    dataset = DGSCorpusDataset(split=Split.TRAIN, num_frames=args.num_frames,
                               velocity=args.velocity, phrase=args.phrase)
    picks = np.random.default_rng(seed).choice(
        len(dataset), min(samples, len(dataset)), replace=False)

    weights = {}
    for level in ("sign", "sentence"):
        counts: Counter = Counter()
        for i in picks:
            counts.update(dataset[int(i)]["bio"][level].numpy().tolist())
        total = sum(counts.values())
        weights[level] = [0.0 if name == "UNK" or not counts.get(index)
                          else total / counts[index]
                          for name, index in BIO.items()]
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
          f"\n  validate on {args.val_datasets}  (first is in-domain, selects the checkpoint)"
          f"\n  training    batch {args.batch_size}  epochs {args.epochs}  "
          f"patience {args.patience}  lr {args.learning_rate:g}  {args.optimizer}"
          f"\n  tricks OFF  dice {args.dice_loss_weight:g}  "
          f"frame_dropout {args.frame_dropout:g}  "
          f"body_part_dropout {args.body_part_dropout:g}  "
          f"attn_dropout {args.attn_dropout:g}  velocity {args.velocity}"
          f"\n  tricks ON   fps_aug {args.fps_aug}  num_frames {args.num_frames}"
          f"\n  schedule    ~{mine.max_steps:,} steps = {args.epochs} epochs, early stop "
          f"{'off (OneCycle runs to completion)' if mine.early_stop == 'off' else f'patience {args.patience}'}"
          f"\n  loss        {'NLL' if args.class_weighting == 'none' else 'NLL + inverse class weights (2023)'}"
          f"{'' if args.dice_loss_weight == 0 else f' + dice {args.dice_loss_weight:g}'}"
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

    from sign_language_segmentation.datasets.common import Split, get_dataloader
    from sign_language_segmentation.model.model import PoseTaggingModel

    loaders = {}
    for split, batch_size in ((Split.TRAIN, args.batch_size), (Split.DEV, 1)):
        loaders[split] = get_dataloader(split=split, dataset_names=args.datasets,
                                        args=args, batch_size=batch_size,
                                        persistent_workers=False)
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

    model = PoseTaggingModel(
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
    print(f"loss           {float(loss):.4f}")
    print("\ndry run OK — data, model and loss all wire up")


if __name__ == "__main__":
    main()
