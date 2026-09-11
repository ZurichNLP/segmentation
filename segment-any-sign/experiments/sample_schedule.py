"""Tilt training sampling toward cleanly-timed videos as training proceeds.

YouTube subtitle timings are nobody's ground truth: on a 300-video sample, 31%
of training videos show no more signing motion inside a subtitle than outside,
and 38% need more than two seconds of shift to line up. Those are still worth
learning from early, when the model is underfit and a wrong boundary costs
little. They are worth less late, when the model would otherwise spend its
capacity fitting them.

So rather than a hard switch — which the run would never be selected from, since
all three pretraining runs picked a checkpoint at 16k-23k steps — each video's
sampling weight decays smoothly from its frame count toward `FLOOR` times that,
if and only if its timings fail `DEV_FILTER`. Clean videos keep full weight
throughout.

    weight(v, t) = frames(v) x (1 if clean(v) else 1 - (1 - FLOOR) x t/T)

At `FLOOR = 0.25` the clean share of drawn frames goes 41% -> 53% -> 74% across
a run, averaging 54%, against a flat 41% today. Nothing is ever excluded: a floor
rather than zero means a video the measure is wrong about still contributes, and
the sampler cannot go degenerate.

Off by default. `--sample-schedule linear` turns it on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytorch_lightning as pl  # noqa: E402

#: Weight a noisy video keeps at the end of training, as a fraction of its frame
#: count. Not zero: the quality measure is a proxy, and a proxy should not get to
#: delete data outright.
FLOOR = 0.25


class SampleScheduleCallback(pl.Callback):
    """Re-weight the training sampler once per epoch.

    `WeightedRandomSampler` draws with `torch.multinomial(self.weights, ...)` at
    the start of each epoch, so replacing `weights` between epochs is enough —
    no rebuilding of the loader, and the worker processes never see it, since
    the sampler runs in the main process and sends indices out.
    """

    def __init__(self, max_steps: int, floor: float = FLOOR,
                 shape: str = "linear"):
        self.max_steps = max(1, max_steps)
        self.floor = floor
        self.shape = shape
        self.base = None          # frame counts, the flat frame-uniform weights
        self.clean = None         # per-item mask, True where timings pass
        self.reported = False

    def _prepare(self, loader) -> bool:
        """Build the frame counts and the clean mask, aligned with the sampler."""
        import numpy as np
        from torch.utils.data import ConcatDataset

        from datasets.youtube_sl25 import load as yt

        def items(dataset):
            if isinstance(dataset, ConcatDataset):
                return [i for part in dataset.datasets for i in items(part)]
            return list(getattr(dataset, "items", []))

        rows = items(loader.dataset)
        if not rows:
            return False
        keep = yt.keep_ids({row["id"] for row in rows})
        self.base = np.array([float(row["total_frames"]) for row in rows])
        self.clean = np.array([row["id"] in keep for row in rows])
        share = self.base[self.clean].sum() / max(self.base.sum(), 1.0)
        print(f"  sample schedule: {self.clean.sum():,}/{len(rows):,} videos "
              f"cleanly timed ({share:.0%} of frames), noisy weight decays "
              f"1.00 -> {self.floor:.2f}", flush=True)
        return True

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        import numpy as np
        import torch

        loader = getattr(trainer, "train_dataloader", None)
        sampler = getattr(loader, "sampler", None)
        if sampler is None or not hasattr(sampler, "weights"):
            if not self.reported:
                print("  sample schedule: training sampler carries no weights "
                      "(needs --sampling frame) — skipping", flush=True)
                self.reported = True
            return
        if self.base is None and not self._prepare(loader):
            return
        if len(self.base) != len(sampler.weights):
            if not self.reported:
                print(f"  sample schedule: {len(self.base)} items but "
                      f"{len(sampler.weights)} weights — skipping", flush=True)
                self.reported = True
            return

        progress = min(1.0, trainer.global_step / self.max_steps)
        if self.shape == "cosine":
            decay = 0.5 * (1 + np.cos(np.pi * progress))
        else:
            decay = 1.0 - progress
        factor = self.floor + (1.0 - self.floor) * decay
        weights = np.where(self.clean, self.base, self.base * factor)
        sampler.weights = torch.as_tensor(weights, dtype=torch.double)

        share = weights[self.clean].sum() / max(weights.sum(), 1e-9)
        pl_module.log("sample/noisy_weight", float(factor))
        pl_module.log("sample/clean_frame_share", float(share))
