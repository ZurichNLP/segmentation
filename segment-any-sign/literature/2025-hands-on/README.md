# Hands-On: Segmenting Individual Signs from Continuous Sequences

- **Authors:** JianHe Low, Harry Walsh, Ozge Mercanoglu Sincan, Richard Bowden
- **Venue:** arXiv, April 2025
- **Link:** https://arxiv.org/abs/2504.08593
- **Status:** partially read

## Why it is here

Follow-up to the 2023 model: stronger hand-centric features (HaMeR) and heavier Transformer backbones for sign-level segmentation.

## Notes

- Source of the observation that the DGS Corpus and BSL Corpus "are among the only large-scale datasets publicly available with the frame-level gloss annotations necessary for sign segmentation" — which matches what we found when surveying ASL options.

## Reported DGS numbers

The only follow-up we have found that evaluates on the **Public DGS Corpus**, so its Table II rows are carried into our [`../../benchmark/`](../../benchmark/) table. Sign level only; it reports no phrase-level results.

| method | F1 | IoU | % |
|---|---|---|---|
| Bi-LSTM + 3D Pose [M&J 2023, E1s] | 0.63 | 0.69 | 1.11 |
| + Hand Norm [M&J 2023, E4s] | 0.59 | 0.63 | 1.13 |
| Ours (Hands-On) | **0.86** | **0.76** | **0.98** |

Its two baseline rows are exactly the 2023 paper's E1s/E4s default-decoding numbers, so it is anchored to the same reference points we reproduce.

Open questions before the 0.86 can be compared to ours directly:

- **Split.** Described only as "adhering to the MeineDGS translation protocols"; we have not confirmed it is the same 9 documents / 17 videos we evaluate on.
- **Decoding.** No threshold values given, so we cannot tell whether the `%` of 0.98 comes from tuned decoding as ours does.
- **Metric details.** It says it uses the metrics of [M&J 2023]; whether the macro-F1 label-set quirk (see the benchmark README) carries over is unknown.

_TODO: read in full and resolve the three points above._
