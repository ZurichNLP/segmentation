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
`validation_mean_mf1s`** — the mean of sign and phrase mF1S, over the supervised
levels only, so a phrase-only run (YouTube) reports phrase mF1S. mF1S counts matched
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sign_language_segmentation.metrics import bio_labels_to_segments  # noqa: E402
from sign_language_segmentation.model.model import PoseTaggingModel  # noqa: E402
from sign_language_segmentation.utils.bio import BIO  # noqa: E402

from metrics import (bio_to_segments, frame_f1, frame_f1_micro,  # noqa: E402
                     global_iou, mf1s_from_counts, segment_counts,
                     segment_percentage)

# upstream UNK=0, O=1, B=2, I=3  ->  ours O=0, B=1, I=2
TO_OURS = {1: 0, 2: 1, 3: 2}

LEVELS = {"sign": "sign", "sentence": "phrase"}  # upstream name -> ours


def _remap(labels: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(labels)
    for src, dst in TO_OURS.items():
        out[labels == src] = dst
    return out


#: our BIO ids, in the order the confusion matrix indexes them
CLASSES = ("O", "B", "I")


def _empty() -> dict:
    return {our: {"frame_f1": [], "frame_f1_micro": [], "iou": [], "percentage": [],
                  "counts": None, "confusion": None}
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

    #: names of the validation dataloaders, in the order they are passed to
    #: `trainer.fit`. The **first is in-domain** and supplies the selection
    #: metric; the rest are logged for information only.
    val_dataset_names = ("dev",)

    #: which heads carry supervision. Subtitle pretraining sets ("sentence",):
    #: masking the sign loss would still compute it, and scoring a head with no
    #: gold wastes a decode per clip.
    levels = ("sign", "sentence")

    #: gradient accumulation, so the LR schedule can be sized in *optimiser*
    #: steps. Upstream is handed batches per epoch and OneCycleLR steps once per
    #: optimiser step, so with accumulation the schedule would be N times too
    #: long and never finish its decay.
    accumulate = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.accumulate > 1:
            import math
            self.steps_per_epoch = max(1, math.ceil(self.steps_per_epoch
                                                    / self.accumulate))
        if self.class_weights:
            import torch.nn as nn
            for level, attr in (("sign", "sign_loss_fn"),
                                ("sentence", "phrase_loss_fn")):
                # only supervised levels are measured, so a one-head run has no
                # entry for the other and must leave its loss function alone
                if level not in self.class_weights:
                    continue
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
                    # not upstream's likeliest_probs_to_segments: that one never
                    # looks at B, so back-to-back segments can only ever be
                    # counted as one. bio_to_segments is the 2023 paper's
                    # Algorithm 1 at argmax, and restores a reachable % of 1.
                    pred_segments = bio_to_segments(
                        probs.argmax(dim=1).numpy(), b=BIO["B"], i=BIO["I"])
                    gold_bio = _remap(gold[:num_frames].cpu()).numpy()
                    pred_bio = _remap(probs.argmax(dim=1)).numpy()

                    bucket = collected[our]
                    # gold x pred counts over O/B/I, summed across clips. Macro and
                    # micro F1 both hide *which* class is wrong, and every failure
                    # so far has been one class: O predicted everywhere, then B.
                    pair = np.bincount(gold_bio.astype(int) * 3 + pred_bio.astype(int),
                                       minlength=9).reshape(3, 3)
                    bucket["confusion"] = pair if bucket["confusion"] is None \
                        else bucket["confusion"] + pair
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

        # The Dice term is defined on the sign head only. It is inlined rather
        # than delegated to `super().step()`, which would recompute the forward
        # *and* add the NLL of every head including the unsupervised one — with
        # --levels sign that silently trained the phrase head as well.
        if self.hparams.dice_loss_weight > 0.0 and "sign" in self.levels:
            sign_gold = batch["bio"]["sign"]
            mask = (sign_gold != BIO["UNK"]).float()
            sign_probs = log_probs["sign"].exp()
            pred_sign = (sign_probs[:, :, BIO["B"]] + sign_probs[:, :, BIO["I"]]) * mask
            gold_sign = (sign_gold >= BIO["B"]).float() * mask
            dice_num = 2.0 * (pred_sign * gold_sign).sum()
            dice_den = pred_sign.sum() + gold_sign.sum() + 1e-6
            dice_loss = (1.0 - dice_num / dice_den) * self.hparams.dice_loss_weight
            total_loss = total_loss + dice_loss
            self.log(f"{name}_dice_loss", dice_loss, batch_size=batch_size)
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

        self._log_classes(collected, prefix)

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
        # Averaged over the *supervised* levels only. During subtitle pretraining
        # the sign head is masked, so counting an absent sign mF1S as 0.0 would
        # halve every logged value — monotonic, so selection still worked, but the
        # number itself was meaningless.
        supervised = [our for our, upstream in (("sign", "sign"), ("phrase", "sentence"))
                      if upstream in self.levels]
        # At train time, log only when a batch was actually scored. Testing the
        # sign bucket specifically would never fire in a phrase-only run, which
        # silently dropped `train_mean_mf1s` from exactly the runs that have one
        # head.
        if prefix == "validation" or any(collected[our]["frame_f1"]
                                         for our in supervised):
            self.log(f"{prefix}_mean_mf1s",
                     sum(mf1s.get(our, 0.0) for our in supervised) / len(supervised),
                     prog_bar=(prefix == "validation"))

    def _log_classes(self, collected: dict, prefix: str) -> None:
        """Per-class shares and scores, under a `classes/` prefix of their own.

        The gold share is logged beside the predicted one so the panel carries its
        own reference line: a model calling 27% of frames B against a gold 0.48%
        is visible immediately, rather than after someone goes and measures it.
        """
        for our, bucket in collected.items():
            pair = bucket.get("confusion")
            if pair is None or not pair.sum():
                continue
            total = float(pair.sum())
            for index, name in enumerate(CLASSES):
                hit = float(pair[index, index])
                gold = float(pair[index].sum())
                pred = float(pair[:, index].sum())
                precision = hit / pred if pred else 0.0
                recall = hit / gold if gold else 0.0
                f1 = (2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
                stem = f"classes/{prefix}_{our}_{name}"
                self.log(f"{stem}_pred_share", pred / total)
                self.log(f"{stem}_gold_share", gold / total)
                self.log(f"{stem}_precision", precision)
                self.log(f"{stem}_recall", recall)
                self.log(f"{stem}_f1", f1)

    #: log gradient diagnostics every N optimiser steps. Cheap — one pass over
    #: 5.7M gradients against a ~0.6 s step — but not free, so it is a knob.
    grad_log_every = 1

    def on_before_optimizer_step(self, optimizer) -> None:
        """Gradient diagnostics, once per optimiser step, *before* clipping.

        Lightning calls this hook immediately before `_clip_gradients`, so these
        are the raw norms — which is what you need to choose a clip threshold.
        With accumulation the gradient here is already the accumulated one, so
        the norms describe the effective batch, not the micro-batch.

        Per-module norms separate a genuinely large update from one head blowing
        up: the phrase head sits on a class that is 0.6% of frames, and its
        gradient is the heavy-tailed one.
        """
        if self.grad_log_every <= 0 or self.global_step % self.grad_log_every:
            return
        total = torch.zeros((), device=self.device)
        biggest = torch.zeros((), device=self.device)
        for name, module in (("cnn", self.frame_cnn),
                             ("input_norm", self.input_norm),
                             ("encoder", self.encoder_attn),
                             ("sign_head", self.sign_bio_head),
                             ("phrase_head", self.sentence_bio_head)):
            squares = [p.grad.pow(2).sum() for p in module.parameters()
                       if p.grad is not None]
            if not squares:
                continue
            group = torch.stack(squares).sum()
            total = total + group
            self.log(f"grad/norm_{name}", group.sqrt())
            peaks = [p.grad.abs().max() for p in module.parameters()
                     if p.grad is not None]
            biggest = torch.maximum(biggest, torch.stack(peaks).max())
        self.log("grad/norm", total.sqrt(), prog_bar=False)
        self.log("grad/max_abs", biggest)

    _train_collected = None

    def on_train_epoch_start(self) -> None:
        if self._train_collected is None:
            self._train_collected = _empty()

    def training_step(self, batch, *args):
        loss = super().training_step(batch, *args)
        if self._train_collected is None:
            self._train_collected = _empty()
        if self.global_step % self.metrics_every_n_steps == 0:
            self._accumulate(batch, self._train_collected)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_collected = {name: _empty() for name in self.val_dataset_names}

    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        """One forward per batch, metrics tagged by which validation set it came from.

        Upstream's own `validation_step` is deliberately not called: it logs
        `validation_sign_iou` and friends under fixed names, which collide across
        dataloaders and would be silently averaged over datasets of different
        domains. Our metrics are a superset of those, and are tagged per set.
        """
        name = self.val_dataset_names[dataloader_idx] \
            if dataloader_idx < len(self.val_dataset_names) else str(dataloader_idx)

        with torch.no_grad():
            log_probs = super().forward(batch["pose"],
                                        timestamps=batch.get("timestamps"))
        # let `step` reuse the forward rather than recompute it for the loss
        self._cached_log_probs = log_probs
        try:
            loss = self.step(batch, name=f"validation_{name}")
        finally:
            self._cached_log_probs = None

        self._accumulate(batch, self._val_collected[name], log_probs=log_probs)
        return loss

    def on_validation_epoch_end(self) -> None:
        # train metrics are flushed here so both sides share one cadence: an
        # epoch is 603 steps on YouTube and 10 on DGS, so epoch-end would put
        # the two curves on incomparable x-axes
        if self._train_collected is not None:
            self._log_collected(self._train_collected, "train")
            self._train_collected = _empty()

        for index, name in enumerate(self.val_dataset_names):
            self._log_collected(self._val_collected[name], f"validation_{name}")
            if index == 0:
                # the in-domain set also supplies the unqualified names the
                # trainer monitors for checkpointing and early stopping
                self._log_collected(self._val_collected[name], "validation")
