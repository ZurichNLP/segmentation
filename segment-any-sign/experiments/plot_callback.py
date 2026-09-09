"""Log a gold-vs-prediction panel to W&B whenever the model improves.

The panel belongs to the *running best* model, not to whatever the last
validation happened to produce. So it is redrawn only when the monitored metric
improves, which is the same test `ModelCheckpoint` applies — the figure in W&B
and the checkpoint on disk are then always the same model.

Logged under one key per validation dataset, so W&B keeps them as a stepped
series: the panel on screen is the newest, and the slider walks the history.

Cost is one forward per clip on the model already in memory, no checkpoint
reload. Twenty clips per dataset, only on improvement, so a run that improves
twenty times pays twenty panels.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytorch_lightning as pl  # noqa: E402

#: reference predictions drawn beside ours when the file exists. Each carries
#: its own model's decoding, so a row shows that model as it was benchmarked.
REFERENCE_PREDICTIONS = {
    "dgs_corpus": {
        "ref_2023": "benchmark/predictions/dgs_validation_2023_E4s-1.json",
        "ref_2026": "benchmark/predictions/dgs_validation_2026.json",
    },
}


class SegmentationPlotCallback(pl.Callback):
    """Redraw and log the panels each time `monitor` reaches a new best."""

    def __init__(self, run_dir: Path, datasets, monitor: str,
                 phrase: str = "sentence", clips: int = 20, root: Path = None):
        self.run_dir = Path(run_dir)
        self.datasets = [name for name in datasets if name]
        self.monitor = monitor
        self.phrase = phrase
        self.clips = clips
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.best = None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # sanity-checking runs a fake validation before training; nothing has
        # been learned yet and the panel would be noise
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        value = float(value)
        if self.best is not None and value <= self.best:
            return
        self.best = value
        self._draw(trainer, pl_module, value)

    def _draw(self, trainer, pl_module, value: float) -> None:
        """Never let a figure interrupt training: log the failure and go on."""
        try:
            import wandb

            from experiments.plot_segmentation import (add_reference, collect,
                                                       plot_clips)

            # the trainer's own W&B run, not the global `wandb.run`: they are
            # normally the same object, but taking it from the logger means the
            # images land on the run Lightning is writing metrics to, at the same
            # step, rather than wherever a stray global happens to point
            experiment = getattr(getattr(trainer, "logger", None), "experiment", None)
            if experiment is None or not hasattr(experiment, "log"):
                print("no W&B logger on the trainer — skipping segmentation panels")
                return

            was_training = pl_module.training
            device = str(pl_module.device)
            step = trainer.global_step
            images = {}
            # restored in `finally`: leaving the module in eval mode would turn
            # dropout off for the rest of the run, and a figure must never be
            # able to change how the model trains
            pl_module.eval()
            try:
                for dataset in self.datasets:
                    records = collect(pl_module, dataset, self.clips, device,
                                      phrase=self.phrase, quiet=True)
                    if not records:
                        continue
                    for key, path in REFERENCE_PREDICTIONS.get(dataset, {}).items():
                        full = self.root / path
                        if full.exists():
                            add_reference(records, full, key)
                    out = self.run_dir / f"segments_{dataset}_phrase.png"
                    plot_clips(records, out, level="phrase",
                               title=f"{self.run_dir.name} — {dataset} dev (phrase), "
                                     f"step {step}, {self.monitor} {value:.4f}")
                    images[f"segments/{dataset}"] = wandb.Image(str(out))
            finally:
                if was_training:
                    pl_module.train()
            if images:
                experiment.log(images, step=step)
                print(f"  logged {len(images)} segmentation panel(s) at step "
                      f"{step} ({self.monitor} {value:.4f})", flush=True)
        except Exception as error:
            print(f"segmentation panel skipped: {type(error).__name__}: {error}",
                  flush=True)
