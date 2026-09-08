# segment-any-sign

Project workspace for *Segment Any Sign*, kept separate from the upstream `sign-language-processing/segmentation` code so it does not collide with ongoing development there.

## How to Contribute

Our conventional main branch is `segment-any-sign`, new branches will be created and merged into it by pull requests.

We can periodically sync from the upstream maintained by Amit/Rylo, and eventually merge back our final product into it.

## Motivation

A continuation of the segmentation model from [Moryossef & Jiang (2023)](https://aclanthology.org/2023.findings-emnlp.846/), which has seen several follow-ups:

- [Revisiting subtitle-unit segmentation](https://aclanthology.org/2025.acl-srw.93/) — BIO tagging and optical flow in a Seq2Seq formulation, on BOBSL and YouTube-ASL.
- [Stronger hand-centric features](https://arxiv.org/abs/2504.08593) — HaMeR features and heavier Transformer backbones.
- [SAGE](https://openaccess.thecvf.com/content/ICCV2025W/MSLR/papers/Low_SAGE_Segment-Aware_Gloss-Free_Encoding_for_Token-Efficient_Sign_Language_Translation_ICCVW_2025_paper.pdf) — segment-informed tokens for efficient translation; up to 50% shorter inputs, 2.67x lower memory.
- [Subtitle–signing alignment](https://arxiv.org/abs/2512.08094) — segmentation as an alignment signal (ACL 2026).
- [Extracting signs from weakly aligned corpora](https://www.sign-lang.uni-hamburg.de/lrec/pub/26039.html) — a study on LSF and LSM (sign-lang@LREC 2026).
- [MHB](https://lrec.elra.info/lrec2026-ws-signlang-52) — handshape-aware boundary detection for ASL, evaluated on the ASLLRP corpus (sign-lang@LREC 2026).

Our interest is how well these trained models generalize:

- The original model was trained and evaluated on DGS; we have a small study transferring to LSF in the original paper, both zero-shot and fine-tuned.
- Colleagues report it also works for ASL and NGT, and it has been useful for alignment in the SEA paper for BSL.
- Beyond more languages, we want to cover edge cases: very short input clips (ISLR datasets), and false positives when no signer is present.
- Special linguistic phenomena such as fingerspelling and indexing may need dedicated treatment.
- We want to be able to generalize and switch between different segmentation granularities more smoothly. In our 2023 model, we have trained at both the sign and phrase levels, but ideally this could be controlled by a single parameter, as in [Segment Any Text](https://aclanthology.org/2024.emnlp-main.665/).

The aim is a systematic benchmark for the task, and a way to track future progress from it; the aim is also to provide a handy, transparent segmentation tool for downstream use.

### Related work

- [Segment Anything](https://arxiv.org/abs/2304.02643) — Meta's SAM series: [SAM](https://arxiv.org/abs/2304.02643) (images, 2023), [SAM 2](https://arxiv.org/abs/2408.00714) (video, 2024), [SAM 3](https://arxiv.org/abs/2511.16719) (open-vocabulary concepts, 2025).
- [Segment Any Text](https://aclanthology.org/2024.emnlp-main.665/) — universal sentence segmentation for text, robust across languages and domains (EMNLP 2024).

Per-paper notes are collected in [`literature/`](literature/). TODO: complete them during literature review.

## Potential venues

[ACL Rolling Review](https://aclrollingreview.org/): you submit to an ARR cycle, reviews come back, then you *commit* the paper to a venue.

### NAACL 2027 — confirmed

- **Where:** San Francisco, California, USA (hybrid)
- **When:** June 1–5, 2027
- **ARR submission deadline:** **October 12, 2026**
- <https://2027.naacl.org/>

### ACL 2027 — dates not yet published

- **Where:** Kyoto, Japan
- **When:** August 17–22, 2027
- **Deadlines:** January, 2027 (estimate)
- <https://2027.aclweb.org/>

## Data

Datasets with *timed* gloss annotations, to be curated in [`datasets/`](datasets/):

- Public DGS Corpus
- BSL Corpus (TODO: download or crawl the videos)
- Signsuisse (DSGS; TODO: gloss-level annotation @agoehring)
- NCSLGR (ASL)
- More to be researched, as part of an initial literature review phase

## Method

Start by benchmarking the 2023 model ([sign-language-processing/segmentation](https://github.com/sign-language-processing/segmentation)), where @AmitMY has already explored autoresearch to improve segmentation scores. Then implement iterative, targeted improvements against the new benchmarks — until we can confidently say: **our model can segment any sign!**

## Setup

```bash
module load miniforge3
conda env create -f environment.yml
conda activate sas
```

`sas` is the environment for **the latest model** (2026) as well as for dataset curation and scoring.

Only the 2023 model needs an environment of its own, because its pose-format 0.3.2 pin cannot coexist with the >=0.8.1 the 2026 model requires:

```bash
conda env create -f environment-2023.yml
conda activate sas2023
```

See [`benchmark/`](benchmark/) for how the two are used together, and [`experiments/`](experiments/) for training runs.

## Write-up

The paper lives in [`latex/`](latex/), a submodule of the Overleaf project — so Overleaf *is* the git remote, and edits flow both ways.

```bash
git submodule update --init                  # first clone (needs Overleaf access)
git -C latex pull                            # take edits made in Overleaf
git -C latex commit -am "..." && git -C latex push   # send edits back
git add latex && git commit -m "latex: ..."  # record the new pointer here
```

The last step matters: the parent repo pins a specific submodule commit, so a push to Overleaf is invisible here until that pointer is committed. Keep it a commit of its own rather than letting it ride along with code changes.
