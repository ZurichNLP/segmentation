# Continuous Sign Language Recognition using Multimodal Input and Handshape-aware Boundary Detection

- **Authors:** Mingyu Zhao, Zhanfu Yang, Yang Zhou, Zhaoyang Xia, Can Jin, Xiaoxiao He, Shuhang Lin, Carol Neidle, Dimitri Metaxas
- **Venue:** sign-lang@LREC 2026 (12th Workshop on the Representation and Processing of Sign Languages: Language in Motion), pp. 501–512
- **Link:** https://lrec.elra.info/lrec2026-ws-signlang-52
- **Preprint:** https://arxiv.org/abs/2511.19907 — titled *MHB: Multimodal Handshape-aware Boundary Detection for Continuous Sign Language Recognition*, with a shorter author list
- **Status:** not yet read

## Why it is here

Follow-up work on boundary detection in continuous signing, for ASL. It evaluates on the ASLLRP corpus, which is the source of our NCSLGR data — see [`../../datasets/ncslgr/`](../../datasets/ncslgr/).

Note **Carol Neidle** is an author on the workshop version. She leads the ASLLRP, so this is the corpus's own group working on our task.

## Notes

From the abstract (not yet verified against the full paper):

- Evaluated on the ASLLRP corpus.
- The sign recognition model is trained on both citation-form isolated signs and signs pre-segmented from continuous signing using manual annotations.
- Uses a handshape classifier over 87 categories, built by integrating and normalising several existing datasets.

_TODO: read for the metrics, splits, and whether the numbers are comparable to ours. Cite the LREC version, not the preprint._
