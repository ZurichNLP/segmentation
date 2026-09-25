# Extracting Signs from Weakly Aligned Sign Language Corpora: A Study on LSF and LSM

- **Authors:** Lorena de la Garza, Julie Halbout, Julie Lascar, Niels Martínez-Guevara, Arturo Curiel, Michèle Gouiffès, Annelies Braffort
- **Venue:** sign-lang@LREC 2026 (12th Workshop on the Representation and Processing of Sign Languages), pp. 174-183
- **Link:** https://www.sign-lang.uni-hamburg.de/lrec/pub/26039.html
- **Status:** read by Zifan

## Why it is here

Applies our segmentation models to LSF and LSM. Adds Mexican Sign Language to the set of languages the model has been tried on, and works from weakly aligned corpora — a different failure mode from the edge cases in our proposal.

## How they use our 2023 model

- **Off the shelf:** MediaPipe poses in, sign-level segments out (§4.2). No finetuning, phrase level unused.
- **No segmentation numbers.** Only downstream annotation precision (LSF P=0.53 at IoU ≥ 0.1; LSM P≈0.08, manual), so nothing for our benchmark table.
- **Feedback on LSM** (interpreted, cropped TV): segments are off in granularity, but the paper says *over*-segmented in §6 and *under*-segmented in §7.
- **Alternatives they plan to try:** Varol et al. (2021) and HaMeR ([Hands-On](../2025-hands-on/)).

## How segmentation helps their downstream task

- **Candidates for MIL-NCE:** each subtitle word gets a bag of sign segments instead of dense sliding windows.
- **Unit of annotation:** the top-1 segment per word becomes the label, and refinement then trims its boundaries.
- **Short segments are lost:** anything under 16 frames is dropped before Swin3D (§4.3), so over-segmentation costs them candidates.

## Worth investigating: Mediapi-signary

[MEDIAPI-SKEL](../../datasets/mediapi_skel/explore.md) has no sign-level annotation, but **Mediapi-signary** (Lascar et al., 2024) does: 445 classes, 15k expert-checked, timed occurrences on the same Média'Pi! source. It is sparse (spotted signs only, not full BIO), but it may do for sign-level eval or LSF finetuning. TODO: check availability and alignment with MEDIAPI-SKEL videos.

## Further remarks

- **SEA:** they borrow its 0–5 s lag for LSM weak alignment, and plan to compare against [SEA](../2025-segment-embed-align/).
- **Interpreted vs original content** is a domain gap (LSM ≪ LSF), close to our proposal's edge cases.
