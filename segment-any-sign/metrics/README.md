# Metrics

Scoring for the benchmark. Reimplemented rather than imported from `sign_language_segmentation`, so it stays independent of any one model: plain segment lists in, no torch, no training-loop coupling.

```python
from metrics import evaluate_level, aggregate

per_clip = [evaluate_level(pred, gold, num_frames) for pred, gold, num_frames in clips]
aggregate(per_clip)
# {'frame_f1': ..., 'frame_f1_micro': ..., 'iou': ..., 'percentage': ..., 'mf1s': ...}
```

Always go through `aggregate`: it averages the frame F1s, `iou` and `percentage` over clips, but aggregates `mf1s` **micro** by summing counts across the corpus. Averaging per-clip `mf1s` yourself gives a different, non-comparable number.

## The metrics

| | level | what it measures | source |
|---|---|---|---|
| `frame_f1` | frame | macro F1 over O/B/I — the paper's primary metric | M&J §4.4 |
| `frame_f1_micro` | frame | the same pooled across classes | diagnostic |
| `iou` | frame | overlap, pooled, no matching | M&J §4.4 |
| `percentage` | segment | `#pred / #gold`, optimal 1 | M&J §4.4 |
| `mf1s` | segment | matched segments clearing an IoU threshold | Renz et al. |

Those are the keys in the returned dict; the underlying functions are `global_iou` and `segment_percentage`.

They disagree deliberately, which is why all of them are reported:

- **`iou` is blind to grouping.** One prediction spanning two gold signs scores IoU 1.0; `percentage` (0.5) and `mf1s` expose it.
- **`frame_f1_micro` is blind to rare classes.** It equals frame accuracy, so a model that never predicts `B` still scores ~0.99 while `frame_f1` collapses to ~0.33. A sanity check on the macro figure, never a headline.

## mF1S: following Renz et al.

Verified against the paper ([arXiv 2011.12986](https://arxiv.org/abs/2011.12986)) and the code ([RenzKa/sign-segmentation](https://github.com/RenzKa/sign-segmentation), `eval.py::get_sign_metric`). The rule: copy their choices, including arbitrary ones, and deviate only for an outright bug.

**Copied** — *segment level* (frame labels become segments first, then matching and counting happen over segments); thresholds 0.40–0.75 step 0.05; *greedy* one-to-one matching, largest IoU first, dropping that row and column (not optimal — an optimal assignment would inflate our scores relative to theirs); strict `>` against the threshold; *micro* aggregation across the corpus.

**One deviation, for a real bug** — their union is the enclosing span (`max(end) - min(start)`) with the intersection never clamped, so disjoint segments get a *negative* IoU. We use `len_a + len_b - intersection`, clamped at zero. This cannot move mF1S: those pairs fail every threshold either way.

Their **exclusive-end** convention is *not* a bug and is left alone — their segments end one past the last frame, so `end - start` is correct there. Ours are inclusive, hence `end - start + 1`.

## Not carried over from `sign_language_segmentation`

- **`segment_f1`** computed `(p*r)/(p+r)` — half of an F1. `mf1s` replaces it.
- **`segment_percentage`** is in the paper but was dropped upstream; restored.
- **`hm_IoU`** is upstream but not in the paper, so comparable to nothing published. Derivable from the two IoU columns if ever wanted.

## Conventions

A segment is `{"start": int, "end": int}` in **frames**, both bounds **inclusive** (length `end - start + 1`). BIO labels are `0 = O`, `1 = B`, `2 = I`. Out-of-range segments are clipped, not rejected.

| edge case | behaviour |
|---|---|
| no gold, no prediction | iou 1.0, percentage 1.0, mf1s 1.0 |
| no gold, some prediction | iou 0.0, mf1s 0.0, percentage = **count** of predictions |
| some gold, no prediction | iou 0.0, mf1s 0.0, percentage 0.0 |

The middle row is the no-signer case from the proposal: `percentage` returns a count rather than dividing by zero, so it is a count and not a ratio there.

## The table this feeds

Sign and phrase level kept separate, as in Table 1 of the 2023 paper. Most datasets support only one level — the Public DGS Corpus is the exception. Cells for an unsupported level are **blank, not zero**: no annotation is not a score of zero.

## Tests

```bash
conda activate sas && cd segment-any-sign && pytest metrics/
```

45 tests: each metric's definition and edge cases, the disagreements above, greedy-versus-optimal matching, micro-versus-macro aggregation, and inclusive-bound arithmetic.
