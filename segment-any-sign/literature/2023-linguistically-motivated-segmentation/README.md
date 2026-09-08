# Linguistically Motivated Sign Language Segmentation

- **Authors:** Amit Moryossef, Zifan Jiang
- **Venue:** Findings of EMNLP 2023
- **Link:** https://aclanthology.org/2023.findings-emnlp.846/
- **Code:** https://github.com/sign-language-processing/segmentation (tag `v2023`)
- **Status:** read

## Why it is here

The paper this project continues. Introduces the BIO-tagged sign and phrase segmentation model trained on the Public DGS Corpus, with optical flow and 3D hand normalisation (the E1s/E4s variants).

## Notes

- §4.1 is the reference for dataset statistics; §5.2 for decoding thresholds (sign b=60, o=40/50/60; phrase E1s 90/90, E4s 80/80-90). The v2023 CLI hardcodes 60/50 and 90/90 — the phrase pair is the E1s tuning.
- Our reproduction of the corpus statistics is in [`../../datasets/public_dgs_corpus/`](../../datasets/public_dgs_corpus/): dev/test document counts match exactly, train is +2.
