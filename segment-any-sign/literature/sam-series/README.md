# Segment Anything (SAM) — the Meta series

Named in the proposal's related work, and the naming inspiration for this project. Three papers so far, all from Meta:

| | paper | year | link |
|---|---|---|---|
| SAM | Segment Anything | 2023 | https://arxiv.org/abs/2304.02643 |
| SAM 2 | SAM 2: Segment Anything in Images and Videos | 2024 | https://arxiv.org/abs/2408.00714 |
| SAM 3 | SAM 3: Segment Anything with Concepts | 2025 | https://arxiv.org/abs/2511.16719 |

- **SAM** (Kirillov, Mintun, Ravi, Mao, Rolland, Gustafson, Xiao, Whitehead, Berg, Lo, Dollár, Girshick) — promptable segmentation for images, trained on SA-1B.
- **SAM 2** (Ravi, Gabeur, Hu, Hu, Ryali, Ma, Khedr, Rädle, Rolland, Gustafson, Mintun, Pan, Alwala, Carion, Wu, Girshick, Dollár, Feichtenhofer) — extends to video with a streaming memory, so segmentation becomes temporal.
- **SAM 3** — adds *promptable concept segmentation*: a short noun phrase or image exemplar segments and tracks every instance of that concept, rather than one geometrically prompted object.

- **Status:** not yet read

## Why it is here

The arc is the interesting part for us: image → video → open-vocabulary concepts, with one model generalising instead of one model per dataset. That is exactly the generalisation claim we want to make for sign segmentation.

## Notes

- SAM 2 is the closest technically, since it is the one that handles **temporal** segmentation in video.
- _Worth being careful with the analogy: SAM segments spatial regions prompted by a user, while we segment a temporal stream with no prompt. The shared idea is generality across domains, not the task formulation._
