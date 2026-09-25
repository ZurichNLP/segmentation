# Continuous Sign Language Recognition using Multimodal Input and Handshape-aware Boundary Detection

- **Authors:** Mingyu Zhao, Zhanfu Yang, Yang Zhou, Zhaoyang Xia, Can Jin, Xiaoxiao He, Shuhang Lin, Carol Neidle, Dimitri Metaxas
- **Venue:** sign-lang@LREC 2026 (12th Workshop on the Representation and Processing of Sign Languages: Language in Motion), pp. 501–512
- **Link:** https://lrec.elra.info/lrec2026-ws-signlang-52
- **PDF:** http://www.lrec-conf.org/proceedings/lrec2026/workshops/signlang/pdf/2026.signlang-1.52.pdf
- **Preprint:** https://arxiv.org/abs/2511.19907 — titled *MHB: Multimodal Handshape-aware Boundary Detection for Continuous Sign Language Recognition*, with a shorter author list. Cite the LREC version.
- **Status:** notes drafted by Claude, reviewed by Zifan

## Why it is here

Follow-up work on boundary detection in continuous signing, for ASL. It evaluates on **ASLLRP-S** (ASLLRP Continuous Signing Corpora v3, June 2025) — the SignStream 3 corpus, not the NCSLGR data we have (see [`../../datasets/ncslgr/explore.md`](../../datasets/ncslgr/explore.md)).

Note **Carol Neidle** is an author on the workshop version. She leads the ASLLRP, so this is the corpus's own group working on our task.

## What they do

- **Our task, our framing:** frame-level BIO (plus a padding class), citing Moryossef et al. (2023) for BIO over IO (§3.1).
- **Model:** 27-joint AlphaPose skeletons with velocity and acceleration → 10 ST-GCN layers → joint pooling → 2 temporal convs → per-frame labels.
- **Loss:** weighted CE (boundary frames up-weighted) + `λ · |N_pred − N_gold|`, the gap in boundary counts per utterance. How the count term is made differentiable is not stated.
- **Handshape prior:** a 3-layer GCN over hand joints, pretrained on 87 NCSLGR handshape categories (NCSLGR handshape videos + ASLLVD, DSP, ASLLRP-S frames), fused into the segmenter by gated cross-attention.
- **Downstream:** predicted segments go to an isolated-sign recogniser (Zhou et al., 2024) to show segmentation is useful.
- **Scope:** dominant-hand signs only; "hidden" signs (false starts, heavy deviations) kept for segmentation, dropped for recognition.

## Numbers

| model | data | mF1B | mF1S |
|---|---|---|---|
| I3D + MS-TCN (Renz et al., 2021) | Phoenix14 | 71.50 | 52.78 |
| MS-TCN + HaMeR ([Hands-On](../2025-hands-on/)) | unclear if re-run | 76.22 | 50.18 |
| MHB w/o handshape | ASLLRP-S | 77.29 | 56.98 |
| **MHB** | ASLLRP-S | **79.40** | **58.26** |

- **Split:** random 4:1 of ASLLRP-S (ratio copied from Renz), not released. Table 1 compares across different corpora, so the gains over baselines are not like-for-like.
- **Their stricter "Boundary Tolerance"** metric (both ends within ±2 to ±5 frames, scaled by sign length): recall 57.4%, precision 58.5%, `%` ≈ 0.98 (6,477 / 6,595).
- **Recognition:** 80.2–83.3% top-1, only on tolerance-matched segments of classes with ≥ 6–30 training examples.
- **Not comparable to ours:** different corpus and split, and their mF1S aggregation (micro or macro) is not stated. [`../../metrics/`](../../metrics/) uses micro.

## What it could mean for segment-any-sign

- **ASLLRP-S as an ASL benchmark:** 2,127 utterances, 17,522 sign tokens, with start/end handshape annotations (per our NCSLGR notes; the paper says ~2,000 classes and < 20k clips). We would curate it as `datasets/asllrp_signstream3/` with our own split, and need Neidle's permission to redistribute anything derived (see the NCSLGR licence note).
- **Boundary-count loss:** it targets our `%` metric directly, and is worth trying if `%` stays off 1.
- **Velocity and acceleration inputs:** cheap to add on pose.
- **Handshape prior:** +2.1 mF1B, +1.3 mF1S. ASLLRP-S's own start/end handshape labels could supervise it directly.
- **Boundary Tolerance:** a candidate extra metric, stricter than mF1S at IoU thresholds and easy to read.
