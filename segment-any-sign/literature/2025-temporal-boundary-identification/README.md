# Sign Language Video Segmentation Using Temporal Boundary Identification

- **Authors:** Kavu Maithri Rao, Yasser Hamidullah, Eleftherios Avramidis
- **Venue:** ACL 2025 Student Research Workshop
- **Link:** https://aclanthology.org/2025.acl-srw.93/
- **Status:** read by Zifan

## Why it is here

Follow-up to the 2023 model: revisits subtitle-unit segmentation with BIO
tagging and optical flow in a Seq2Seq formulation, on BOBSL and YouTube-ASL.

## Findings

- **Two datasets, two separately trained models** — BOBSL (manually-aligned
  subset: 60 videos, 40/10/10 split, 25 fps, ~45 min each) and YouTube-ASL
  (70/20/10). Not a joint model: parameter counts and training times differ per
  dataset. Combining them is named as future work.
- **Sequence Encoder beats Autoregressive** on both, and trains far faster
  (~14 h vs ~1 day).

| model | dataset | F1 | IoU | % |
|---|---|---|---|---|
| Sequence Encoder | BOBSL | 0.58 | 0.60 | 2.50 |
| Sequence Encoder | YouTube-ASL | 0.56 | 0.58 | 0.70 |
| Autoregressive | BOBSL | 0.55 | 0.51 | 1.74 |
| Autoregressive | YouTube-ASL | 0.47 | 0.50 | 0.55 |

- **The corpora bias segmentation in opposite directions**: BOBSL over-segments
  (`%` 2.50 / 1.74), YouTube-ASL under-segments (`%` 0.70 / 0.55). They attribute
  this to differing annotation granularity between the two.
- **Pipeline**: ImageNet-pretrained ResNet-101 as a frozen feature extractor over
  optical flow (BOBSL's pre-computed; RAFT at a 10-frame stride for YouTube-ASL),
  then a BiLSTM predicting frame-level BIO over fixed 375-frame windows.

## What we can borrow

- **Subtitle timings straight to BIO**, with no alignment step and no filtering
  for subtitle quality. That is the cheapest possible weak-label recipe for a
  YouTube pretraining stage, and it is the label format our pipeline already
  uses.
- **Optical flow for boundaries.** Both this and the 2023 E4 variant arrived at
  optical flow independently, which is some evidence it carries boundary signal
  that raw pose does not.
- **A warning for that pretraining stage.** Subtitle-derived boundaries are
  coarser than phrase annotation — YouTube-ASL gives `%` 0.70 here. Pretraining on
  them risks biasing toward the under-segmentation our phrase level already
  shows, so the bias should be measured rather than assumed away.

Their features are ResNet-101 over RGB and flow, not MediaPipe pose, so the
numbers are not comparable to ours — only the recipe transfers.
