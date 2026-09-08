# Benchmark

Inference-only evaluation: run existing models over the curated datasets, score with [`../metrics/`](../metrics/), produce the table. No training here — that is a future `experiments/`.

## Rules

**One evaluation protocol for every model, inherited from Moryossef & Jiang (2023). Deviate only for an explicit bug, and record why.**

Same rule [`../metrics/`](../metrics/) follows for metric definitions. It fixes the clip list, split, filters, gold, and frame conversion; a newer model is measured against those whether or not it was built for them.

A model's dedicated **preprocessing is not part of the protocol**: that is the model, not the measurement.

| deviation taken | why it is a bug, not a preference |
|---|---|
| 3 of 17 test clips dropped | no annotation at all — a free ~1.0 on every metric |
| upstream `segment_f1` dropped | computed `(p*r)/(p+r)`, half an F1; `mF1S` replaces it |

## Running it

```bash
conda activate sas2023                                    # the 2023 model only
python benchmark/predict_dgs_2023.py --split test --model model_E4s-1.pth

conda activate sas                                        # the 2026 model + scoring
python benchmark/predict_dgs_2026.py --split test
python benchmark/score.py benchmark/predictions/*.json
```

Clips, filters and gold come from [`../datasets/public_dgs_corpus/load.py`](../datasets/public_dgs_corpus/load.py) for every model: `iter_clips` reads the TFDS build at 25fps, `iter_clips_native` the archived `.pose` originals at 50fps. Each model gets the one it was developed against. Predictions land in `predictions/*.json` as segments plus RLE'd frame labels.

## Results

Public DGS Corpus test — 9 documents, **14 annotated videos** of 17.

`F1-ma`/`F1-mi` are frame-level F1 over O/B/I, macro/micro; `IoU` is pooled frame overlap; `%` is `#pred / #gold`, optimal at **1** either way; `mF1S` is matched-segment F1 ([`../metrics/`](../metrics/)). Models carry decoding thresholds as `(sign b/o, phrase b/o)`.

| | | Sign | | | | | Phrase | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Model** | **Source** | **F1-ma** | **F1-mi** | **IoU** | **%** | **mF1S** | **F1-ma** | **F1-mi** | **IoU** | **%** | **mF1S** |
| *— paper reported results below —* | | | | | | | | | | | |
| M&J 2023 E1s (50/50, 50/50) | paper, Tab. 2 | 0.63 | — | 0.69 | 1.11 | — | 0.65 | — | 0.82 | 1.63 | — |
| M&J 2023 E4s (50/50, 50/50) | paper, Tab. 2 | 0.59 | — | 0.63 | 1.13 | — | 0.62 | — | 0.79 | 1.43 | — |
| M&J 2023 E1s\* (60/50, 90/90) | paper, Tab. 2 | — | — | 0.69 | 1.03 | — | — | — | 0.85 | 1.02 | — |
| M&J 2023 E4s\* (60/50, 80/80) | paper, Tab. 2 | — | — | 0.63 | 1.06 | — | — | — | 0.79 | 1.12 | — |
| Hands-On 2025 (n/r) | paper, Tab. II | 0.86 | — | 0.76 | 0.98 | — | — | — | — | — | — |
| 2026 E169, 50fps | [dist/2026](https://github.com/sign-language-processing/segmentation/blob/main/dist/2026/README.md) | — | — | 0.652 | — | — | — | — | 0.925 | — | — |
| *— our benchmark results below —* | | | | | | | | | | | |
| 2023 E1s (60/50, 90/90) | ours | 0.560 | 0.701 | 0.621 | 1.031 | 0.441 | 0.589 | 0.854 | 0.814 | 0.964 | 0.361 |
| 2023 E4s (60/50, 80/80) | ours | 0.553 | 0.695 | 0.620 | 1.074 | 0.429 | 0.593 | 0.858 | 0.817 | 1.073 | 0.376 |
| 2026 shipped (argmax, 50fps) | ours | 0.529 | 0.800 | 0.611 | 0.943 | 0.436 | 0.513 | 0.860 | 0.791 | 0.437 | 0.071 |
| *— our experimental results below —* | | | | | | | | | | | |
| `00_2026_baseline` | ours | 0.510 | 0.783 | 0.597 | 1.071 | 0.391 | 0.549 | 0.886 | 0.825 | **0.771** | **0.200** |

`*` marks the 2023 paper's tuned decoding, which moves only IoU and `%` — hence `—` for the decoding-independent F1. `F1-mi` and `mF1S` are unpublished by anyone.

**Hands-On** is the only follow-up on this corpus. See [`../literature/2025-hands-on/`](../literature/2025-hands-on/).

## Protocol

**The three dropped clips.** `1180022_b`, `1187154_a`, `1419122_a` are the unannotated partner in a two-signer conversation — those documents carry `Deutsche_Übersetzung` and `Lexem_Gebärde` tiers for one participant only. A person is on camera but is the listener, and every model correctly predicts almost nothing. With no gold and no prediction each scores ~1.0 throughout: sign IoU 0.679 across 17 against 0.611 across 14. `score.py --all-clips` restores the 17-video set. Upstream's 2026 eval independently drops the same three.

**What counts as a phrase.** The two codebases disagree:

| | phrase = | source |
|---|---|---|
| v2023 | first gloss → last gloss of a sentence | `data.py::build_classes_vectors` |
| 2026 | the `Deutsche_Übersetzung` tier's timeslots | `datasets/dgs/dataset.py:127` |

Sentence bounds are wider; the gloss extent sits inside them, trimming lead-in and trail-out. Phrase IoU against both, same predictions:

| Model | vs gloss extent (**benchmark**) | vs sentence bounds |
|---|---|---|
| 2023 E1s | 0.814 | 0.863 |
| 2023 E4s | 0.817 | 0.849 |
| 2026 | 0.792 | 0.922 |

We use the **2023 gloss extent** for every row in our benchmark table above.

## Reproductions

### Moryossef & Jiang (2023)

`--all-clips` figures, since the paper scored all 17:

| model | level | frame F1 | (paper) | IoU | (paper) |
|---|---|---|---|---|---|
| E1s | sign | 0.6378 | 0.63 | 0.6878 | 0.69 |
| E1s | phrase | 0.6615 | 0.65 | 0.8471 | 0.85 |
| E4s | sign | 0.5924 | 0.59 | 0.6280 | 0.63 |
| E4s | phrase | 0.6258 | 0.62 | 0.7902 | 0.79 |

IoU is compared against the tuned-decoding rows (E1s\*/E4s\*), which is what we run; frame F1 is decoding-independent.

`predict_dgs_2023.py` **drives the original v2023 code**. A few noted details:

- `tfds_dataset.py` imports **its own `pose_utils`**, not pose-format's — its `pose_hide_legs` zeroes eight leg points *and* their confidences. The same-named pose-format helper costs ~0.07 frame F1.
- **`pose-format` version matters**: v2023 pinned `>=0.3.2`, 0.9.0 changes the scores. `sas2023` pins 0.3.2.
- Two different golds: `floor(t * fps)` inclusive for IoU and `%`, `build_bio`'s walk (effectively `ceil`) for frame metrics.
- **Frame F1 compares argmax of raw probabilities against `build_bio` labels**, never decoded segments.
- Its macro F1 passes **no label set**, averaging only over present classes; `score.py` passes `labels=None` to match.
- `<cmdp:Task>Joke</cmdp:Task>` documents are dropped, taking the test split from 10 documents to 9.

### The 2026 model (unconfirmed)

[dist/2026](https://github.com/sign-language-processing/segmentation/blob/main/dist/2026/README.md) — CNN-UNet + RoPE transformer, trained on DGS Corpus 3.0.0-uzh-document, shipped as `dist/2026/model.safetensors`.

Reproduces under *its* protocol: `--phrase sentence` gives phrase IoU 0.922 against a published 0.925, same 14 clips. Sign is 0.611 against 0.652 either way (the sign definition never changed); the likeliest remaining difference is pose provenance — upstream reads a `poses_dir` of MediaPipe Holistic poses, we read the archived `.pose` downloads, and sign boundaries are the more extraction-sensitive level. Unconfirmed.

Metrics were checked against their `evaluate.py`: macro frame F1 with no label set, `segment_IoU` identical to our `global_iou`, argmax decoding, gold from BIO labels, mean over clips. Three corrections were needed, each a trap for the next model:

1. **It does not use TFDS** — its loader reads raw `.pose` and `.eaf` directly.
2. **fps.** The TFDS build baked in a 25fps downsample; the model publishes at 50. The 50fps originals were already in the download archive, keyed by `original_fname` in each `.INFO` sidecar — check there before rebuilding a config at ~144 GB and hours. Worth +0.06 sign IoU.
3. **Phrase definition**, above.

**IoU alone can mislead.** This model selects on IoU and reports nothing else. At phrase level it reaches 0.922 IoU with `%` at 0.437 — nearly the right frames from **under half** the segments, merging adjacent sentences. IoU cannot see this: one prediction spanning two gold phrases scores as well as two correct ones. Worth further manual investigation.
