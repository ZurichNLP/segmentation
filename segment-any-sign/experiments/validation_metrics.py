"""Log the benchmark's full metric set at validation, not just IoU.

Upstream validates on IoU alone (`validation_sign_iou`, `validation_phrase_iou`,
`validation_hm_iou`). IoU cannot see over-merging — one prediction spanning two
gold segments scores as well as two correct ones — which is exactly the failure
mode the 2026 model shows at phrase level (see ../benchmark/README.md). Watching
only IoU during training means not noticing it until the test run.

So this subclass adds every metric from [`../metrics/`](../metrics/), computed the
same way `benchmark/score.py` computes them, and logs them beside upstream's —
**on train as well as validation**, under matching names (`train_sign_iou` /
`validation_sign_iou`, and so on) so W&B can overlay the two curves in one panel.

`validation_hm_iou` is still logged unchanged, but **selection is on
`validation_mean_mf1s`** — the mean of sign and phrase mF1S. mF1S counts matched
segments, so unlike IoU it penalises the merging the 2026 model shows; selecting
on it picks checkpoints that segment rather than merely cover. Two consequences
worth stating: our checkpoints are no longer selected the way the published 2026
model was, and the mean is dominated by the sign level, since phrase mF1S runs
far lower. `--select-on hm_iou` restores upstream's choice.

**Train and validation are not measured on the same thing**, and the overlay
should be read with that in mind: training batches are random 1024-frame windows
with augmentation applied, while validation runs whole videos with none. Expect
train to look easier. The gap between the curves is still the useful signal — it
is where overfitting shows up.

Cost: none at validation. Upstream runs two forward passes per validation batch
and ours would be a third, so instead the forward is computed once and cached for
all three — which makes validation *cheaper* than it was before these metrics
existed. Training pays one extra forward every `metrics_every_n_steps` steps
(9, about once per epoch at batch 64).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from sign_language_segmentation.metrics import (  # noqa: E402
    bio_labels_to_segments, likeliest_probs_to_segments)
from sign_language_segmentation.model.model import PoseTaggingModel  # noqa: E402
from sign_language_segmentation.utils.bio import BIO  # noqa: E402

from metrics import (frame_f1, frame_f1_micro, global_iou,  # noqa: E402
                     mf1s_from_counts, segment_counts, segment_percentage)

# upstream UNK=0, O=1, B=2, I=3  ->  ours O=0, B=1, I=2
TO_OURS = {1: 0, 2: 1, 3: 2}

LEVELS = {"sign": "sign", "sentence": "phrase"}  # upstream name -> ours


def _remap(labels: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(labels)
    for src, dst in TO_OURS.items():
        out[labels == src] = dst
    return out


def _empty() -> dict:
    return {our: {"frame_f1": [], "frame_f1_micro": [], "iou": [], "percentage": [],
                  "counts": None}
            for our in LEVELS.values()}


class ValidationMetricsModel(PoseTaggingModel):
    """PoseTaggingModel reporting the benchmark's metrics on train and validation.

    Also restores 2023's **inverse class-frequency weighting** when it is set.
    Upstream replaced that weighting with a Dice term on the sign head, so
    switching Dice off leaves a plain unweighted NLL that is neither model's
    loss. `class_weights` is set by `train.py` before training starts.
    """

    #: {"sign": [...], "sentence": [...]}, one weight per BIO class, or None for
    #: upstream's unweighted NLL. Set before the model is constructed.
    class_weights = None

    #: which heads carry supervision. Subtitle pretraining sets ("sentence",):
    #: masking the sign loss would still compute it, and scoring a head with no
    #: gold wastes a decode per clip.
    levels = ("sign", "sentence")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.class_weights:
            import torch.nn as nn
            for level, attr in (("sign", "sign_loss_fn"),
                                ("sentence", "phrase_loss_fn")):
                weight = torch.tensor(self.class_weights[level], dtype=torch.float)
                # NLLLoss is a Module, so Lightning moves the weight with the model
                setattr(self, attr, nn.NLLLoss(reduction="none", weight=weight))

    #: compute train metrics every N steps. 9 ~= once per epoch at batch 64 over
    #: 586 clips; 1 would roughly double training cost.
    metrics_every_n_steps = 9

    #: set for the duration of one validation_step so upstream's two internal
    #: forward passes reuse ours instead of recomputing it
    _cached_log_probs = None

    def forward(self, pose_data, timestamps=None, *args, **kwargs):
        if self._cached_log_probs is not None:
            return self._cached_log_probs
        return super().forward(pose_data, timestamps, *args, **kwargs)

    def _accumulate(self, batch, collected: dict, log_probs=None) -> None:
        """Score one batch into `collected`, exactly as benchmark/score.py would."""
        with torch.no_grad():
            if log_probs is None:
                log_probs = self.forward(batch["pose"],
                                         timestamps=batch.get("timestamps"))

            for upstream_name, our in LEVELS.items():
                if upstream_name not in self.levels:
                    continue
                gold_all = batch["bio"][upstream_name]
                for i in range(len(batch["pose"])):
                    gold = gold_all[i]
                    num_frames = int((gold != BIO["UNK"]).sum())
                    if num_frames == 0:
                        continue

                    gold_segments = bio_labels_to_segments(gold[:num_frames])
                    # annotated clips only, as the benchmark scores by default:
                    # an unannotated clip is a free 1.0 on every metric
                    if not gold_segments:
                        continue

                    # .float() because `precision="bf16-mixed"` can hand back
                    # bf16, which numpy cannot convert (upstream's own
                    # likeliest_probs_to_segments trips on this too)
                    probs = log_probs[upstream_name][i][:num_frames].cpu().float()
                    pred_segments = likeliest_probs_to_segments(probs)
                    gold_bio = _remap(gold[:num_frames].cpu()).numpy()
                    pred_bio = _remap(probs.argmax(dim=1)).numpy()

                    bucket = collected[our]
                    # labels=None matches score.py: average over present classes
                    bucket["frame_f1"].append(frame_f1(pred_bio, gold_bio, labels=None))
                    bucket["frame_f1_micro"].append(
                        frame_f1_micro(pred_bio, gold_bio, labels=None))
                    bucket["iou"].append(global_iou(pred_segments, gold_segments, num_frames))
                    bucket["percentage"].append(
                        segment_percentage(pred_segments, gold_segments))
                    counts = segment_counts(pred_segments, gold_segments)
                    bucket["counts"] = counts if bucket["counts"] is None \
                        else bucket["counts"] + counts

    def step(self, batch, name: str):
        """Upstream's step, restricted to the heads that carry supervision.

        Upstream iterates both heads unconditionally, so an unsupervised one
        cannot be skipped by hiding its labels — this mirrors its loss instead.
        Kept deliberately close to the original; the only change is the `levels`
        filter and the matching guard on the sign-head Dice term.
        """
        if set(self.levels) == set(LEVELS):
            return super().step(batch, name)

        pose_data = batch["pose"]
        batch_size = len(pose_data)
        log_probs = self.forward(pose_data, timestamps=batch.get("timestamps"))

        total_loss = torch.zeros(1, device=self.device).squeeze()
        for pred_type, loss_fn in (("sign", self.sign_loss_fn),
                                   ("sentence", self.phrase_loss_fn)):
            if pred_type not in self.levels:
                continue
            gold = batch["bio"][pred_type]
            loss = loss_fn(log_probs[pred_type].transpose(1, 2), gold)
            mask = (gold != BIO["UNK"]).float()
            masked_loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            total_loss = total_loss + masked_loss
            self.log(f"{name}_{pred_type}_loss", masked_loss,
                     batch_size=batch_size, prog_bar=True)

        self.log(f"{name}_loss", total_loss, batch_size=batch_size)
        # the Dice term is defined on the sign head only, so it goes with it
        if self.hparams.dice_loss_weight > 0.0 and "sign" in self.levels:
            return super().step(batch, name)
        return total_loss

    def _log_collected(self, collected: dict, prefix: str) -> None:
        """Log means, the two IoUs' harmonic mean, and the mF1S selection metric."""
        ious, mf1s = {}, {}
        for our, bucket in collected.items():
            if not bucket["frame_f1"]:
                continue
            for key in ("frame_f1", "frame_f1_micro", "iou", "percentage"):
                self.log(f"{prefix}_{our}_{key}",
                         sum(bucket[key]) / len(bucket[key]), prog_bar=False)
            ious[our] = sum(bucket["iou"]) / len(bucket["iou"])
            # mF1S is aggregated micro over the corpus, never averaged per clip
            if bucket["counts"] is not None:
                mf1s[our] = mf1s_from_counts(bucket["counts"])
                self.log(f"{prefix}_{our}_mf1s", mf1s[our], prog_bar=False)

        # mirrors upstream's validation_hm_iou so the two curves can be overlaid.
        # Upstream logs its own `validation_hm_iou`; ours is named differently to
        # avoid ever shadowing the metric that selects the checkpoint.
        sign, phrase = ious.get("sign", 0.0), ious.get("phrase", 0.0)
        if sign > 0 and phrase > 0:
            self.log(f"{prefix}_hm_iou_ours", 2 * sign * phrase / (sign + phrase))

        # The checkpoint selection metric. Always logged at validation — a metric
        # the trainer monitors must exist from the first validation epoch or
        # EarlyStopping raises. At train time it is logged only when a batch was
        # actually sampled: with few steps per epoch, `metrics_every_n_steps` can
        # skip whole epochs, and logging 0.0 for those would draw a sawtooth.
        if prefix == "validation" or collected["sign"]["frame_f1"]:
            self.log(f"{prefix}_mean_mf1s",
                     (mf1s.get("sign", 0.0) + mf1s.get("phrase", 0.0)) / 2,
                     prog_bar=(prefix == "validation"))

    def on_train_epoch_start(self) -> None:
        self._train_collected = _empty()

    def training_step(self, batch, *args):
        loss = super().training_step(batch, *args)
        if self.global_step % self.metrics_every_n_steps == 0:
            self._accumulate(batch, self._train_collected)
        return loss

    def on_train_epoch_end(self) -> None:
        self._log_collected(self._train_collected, "train")

    def on_validation_epoch_start(self) -> None:
        self._val_collected = _empty()

    def validation_step(self, batch, *args):
        # Upstream runs two forwards per validation batch — one in `step` for the
        # loss, one in `validation_step` for IoU — and ours would be a third. On
        # whole dev videos that is the single largest cost in a run, so compute it
        # once and let both reuse it through the cache in `forward` above. No
        # upstream logic is duplicated; it simply gets handed the tensor it would
        # otherwise recompute.
        with torch.no_grad():
            log_probs = super().forward(batch["pose"],
                                        timestamps=batch.get("timestamps"))
        self._cached_log_probs = log_probs
        try:
            loss = super().validation_step(batch, *args)
        finally:
            self._cached_log_probs = None

        self._accumulate(batch, self._val_collected, log_probs=log_probs)
        return loss

    def on_validation_epoch_end(self) -> None:
        self._log_collected(self._val_collected, "validation")
