"""Padding the model cannot see.

Upstream's `PoseTaggingModel` assumes every sequence is a full `num_frames`
window. Its attention takes no mask, its BatchNorm averages over every position,
and a long video's last transformer chunk is zero-padded. So a padded frame is
not inert: real frames attend to it, it shifts batch statistics, and after a
convolution, BatchNorm and SiLU it is no longer zero, so the next convolution
carries it into the real frames beside it.

This makes padding invisible end to end, so a sequence padded inside a batch is
processed exactly as the same sequence alone:

  * convolutions: activations at padded positions are zeroed after every layer —
    the same zeros a convolution pads at a sequence end — with each U-Net level's
    valid length tracked through its strided and transposed convolutions
  * BatchNorm: `MaskedBatchNorm1d` takes training statistics from real positions
    only; in eval mode it uses the running statistics, as before
  * attention: padded positions are masked out as keys
  * chunking: a long video is cut into `num_frames` chunks as upstream cuts it,
    the last one padded and masked

No parameter changes, so every checkpoint — ours and the shipped 2026 one — loads
as it is. A batch with no padding takes upstream's code paths unchanged: the
fused BatchNorm kernel and attention without a mask. Only padded inputs are
computed differently.

    from experiments.masked_model import enable_masking
    model = enable_masking(model)          # any PoseTaggingModel, in place
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F
from torch import nn


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """`nn.BatchNorm1d` whose training statistics come from real positions only.

    Same parameters and buffers, so the state dict is interchangeable with the
    module it replaces. Without a mask, with nothing masked, or in eval mode it is
    `nn.BatchNorm1d` itself.

    With padding in training, the real positions are gathered into one (V, C)
    batch and handed to `F.batch_norm`, then scattered back; padded positions come
    out as zero. So this *is* BatchNorm over the real frames — the fused kernel's
    statistics, running-average update and gradient — and it keeps what the fused
    kernel keeps for the backward pass, one copy of the input.

    An earlier version computed the statistics by hand and kept about three
    copies. In the second U-Net, where 384 features are folded into the batch, one
    copy is 1.5 GiB at batch 64, and a padded batch ran an 80 GB H100 out of
    memory at step ~20k.
    """

    def forward(self, input: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None or not self.training or bool(mask.all()):
            return super().forward(input)
        self._check_input_dim(input)

        factor = 0.0
        if self.track_running_stats:
            # what nn.BatchNorm1d.forward does before calling F.batch_norm
            self.num_batches_tracked.add_(1)
            factor = (1.0 / float(self.num_batches_tracked)
                      if self.momentum is None else self.momentum)

        rows = input.transpose(1, 2)[mask]                       # (V, C), real positions
        normed = F.batch_norm(
            rows,
            self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None,
            self.weight, self.bias, True, factor, self.eps)
        out = torch.zeros(input.shape[0], input.shape[2], input.shape[1],
                          dtype=normed.dtype, device=normed.device)
        out[mask] = normed
        return out.transpose(1, 2)


def use_masked_batchnorm(module: nn.Module) -> nn.Module:
    """Replace every `nn.BatchNorm1d` under `module` with `MaskedBatchNorm1d`, in place."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d) and not isinstance(child, MaskedBatchNorm1d):
            masked = MaskedBatchNorm1d(child.num_features, eps=child.eps,
                                       momentum=child.momentum, affine=child.affine,
                                       track_running_stats=child.track_running_stats)
            masked.load_state_dict(child.state_dict())
            masked.to(child.running_mean.device if child.running_mean is not None
                      else child.weight.device)
            masked.train(child.training)
            setattr(module, name, masked)
        else:
            use_masked_batchnorm(child)
    return module


def _position_mask(lengths: torch.Tensor, size: int) -> torch.Tensor:
    """(N, size) bool, True at positions before each row's length."""
    return torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]


def _keep(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Zero every position at or past `lengths` along the last axis of (N, C, T)."""
    return values * _position_mask(lengths, values.shape[-1]).unsqueeze(1).to(values.dtype)


def _unet_block(block, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """`PoseEncoderUNetBlock.forward`, with padded positions held at zero.

    Each layer is conv -> BatchNorm -> SiLU. After it, positions past the level's
    valid length are zeroed, which is what the next convolution would see past
    the end of an unpadded sequence. A strided convolution of length L gives
    ceil(L / s); a transposed one of valid length L gives (L - 1) s + 1, the length
    upstream then pads to its skip connection.
    """
    batch, sequence, joints, channels = x.shape
    h = x.permute(0, 2, 3, 1).reshape(batch * joints, channels, sequence)
    lens = lengths.repeat_interleave(joints)
    h = _keep(h, lens)

    skips = []
    for layer in block.encoder_layers:
        conv, norm, act = layer
        h = conv(h)
        lens = -(-lens // conv.stride[0])
        h = _keep(act(norm(h, _position_mask(lens, h.shape[-1]))), lens)
        skips.append((h, lens))

    for layer in block.decoder_layers:
        skip, skip_lens = skips.pop()
        diff = skip.shape[-1] - h.shape[-1]
        if diff > 0:
            left = diff // 2
            if left:
                # never with upstream's strides (<= 1 frame of difference); a left
                # pad would shift frames against the lengths tracked here
                raise ValueError(f"U-Net skip needs {diff} frames of padding, "
                                 f"which masking does not support")
            h = F.pad(h, (0, diff))
        conv, norm, act = layer
        h = conv(h + skip)
        lens = torch.where(skip_lens > 0, (skip_lens - 1) * conv.stride[0] + 1, skip_lens)
        h = _keep(act(norm(h, _position_mask(lens, h.shape[-1]))), lens)

    _, out_channels, out_sequence = h.shape
    h = h.view(batch, joints, out_channels, out_sequence).permute(0, 3, 1, 2)
    return block.fc(h.mean(dim=-1))


def _frame_cnn(cnn: nn.Sequential, pose_data: torch.Tensor,
               lengths: torch.Tensor = None) -> torch.Tensor:
    if lengths is None:
        return cnn(pose_data)
    x = pose_data
    for module in cnn:
        x = _unet_block(module, x, lengths) if hasattr(module, "encoder_layers") else module(x)
    return x


def _attention_layer(layer, x: torch.Tensor, timestamps: torch.Tensor,
                     valid: torch.Tensor = None) -> torch.Tensor:
    """`RoPETransformerEncoderLayer.forward` with padded positions masked as keys.

    On GPU, rows computed in a batch of another shape differ from upstream by
    float noise (up to ~1e-1 in log-probability with TF32, the default on H100;
    ~1e-4 without). That is arithmetic, not masking: unmasked padding differs by
    ~1e2 on the same test.
    """
    if valid is None:
        return layer(x, timestamps)
    batch, sequence, dim = x.shape
    cos, sin = layer._compute_rope(timestamps)

    h = layer.norm1(x)
    q, k, v = layer.qkv(h).reshape(batch, sequence, 3, layer.nhead, layer.head_dim).unbind(2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    q = q * cos + layer._rotate_half(q) * sin
    k = k * cos + layer._rotate_half(k) * sin

    # a row with no real frame at all (a chunk wholly past a sequence's end) may
    # attend to everything rather than nothing, which would be NaN; its output is
    # padding and is never read
    keys = valid | ~valid.any(dim=-1, keepdim=True)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=keys[:, None, None, :],
        dropout_p=layer.attn_drop.p if layer.training else 0.0)
    x = x + layer.out_proj(out.transpose(1, 2).reshape(batch, sequence, dim))
    return x + layer.ffn(layer.norm2(x))


def masked_encode(self, pose_data: torch.Tensor, timestamps: torch.Tensor = None,
                  lengths: torch.Tensor = None) -> torch.Tensor:
    """`PoseTaggingModel.encode`, with frames at or past `lengths` masked throughout.

    `lengths` is per sequence, as `collate_fn` returns it; None, or all equal to the
    padded length, means no padding. Long sequences are cut into `num_frames`
    chunks like upstream, generalised past batch size one, and a last chunk
    shorter than `num_frames` is padded and masked.
    """
    batch, total = pose_data.shape[:2]
    if lengths is not None:
        lengths = lengths.to(pose_data.device)
        if bool((lengths >= total).all()):
            lengths = None

    x = self.input_norm(_frame_cnn(self.frame_cnn, pose_data, lengths))

    if timestamps is None:
        ts = (torch.arange(total, device=x.device, dtype=torch.float32)
              / self.REFERENCE_FPS).unsqueeze(0).expand(batch, -1)
    else:
        ts = timestamps.to(x.device)
        if ts.dim() == 1:
            ts = ts.unsqueeze(0).expand(batch, -1)

    valid = None if lengths is None else _position_mask(lengths, total)
    chunk = self.hparams.num_frames
    if total <= chunk:
        for layer in self.encoder_attn:
            x = _attention_layer(layer, x, ts, valid)
        return x

    chunks = math.ceil(total / chunk)
    pad = chunks * chunk - total
    chunk_valid = None
    if valid is not None or pad:
        full = valid if valid is not None else torch.ones(batch, total, dtype=torch.bool,
                                                          device=x.device)
        chunk_valid = F.pad(full, (0, pad), value=False).reshape(batch * chunks, chunk)
    x_chunks = F.pad(x, (0, 0, 0, pad)).reshape(batch * chunks, chunk, x.shape[-1])
    ts_chunks = F.pad(ts, (0, pad)).reshape(batch * chunks, chunk)
    for layer in self.encoder_attn:
        x_chunks = _attention_layer(layer, x_chunks, ts_chunks, chunk_valid)
    return x_chunks.reshape(batch, chunks * chunk, x.shape[-1])[:, :total]


def masked_forward(self, pose_data: torch.Tensor, timestamps: torch.Tensor = None,
                   lengths: torch.Tensor = None) -> dict:
    """`PoseTaggingModel.forward`, passing `lengths` through to `masked_encode`."""
    encoded = masked_encode(self, pose_data, timestamps, lengths)
    return {
        "sign": F.log_softmax(self.sign_bio_head(encoded), dim=-1),
        "sentence": F.log_softmax(self.sentence_bio_head(encoded), dim=-1),
    }


def enable_masking(model):
    """Make a loaded `PoseTaggingModel` padding-aware, in place, and return it."""
    use_masked_batchnorm(model.frame_cnn)
    model.encode = types.MethodType(masked_encode, model)
    model.forward = types.MethodType(masked_forward, model)
    return model
