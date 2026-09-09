# Sign Language Video Segmentation Using Temporal Boundary Identification

- **Authors:** Kavu Maithri Rao, Yasser Hamidullah, Eleftherios Avramidis
- **Venue:** ACL 2025 Student Research Workshop
- **Link:** https://aclanthology.org/2025.acl-srw.93/
- **Status:** read by Zifan

## Why it is here

Follow-up to the 2023 model: revisits subtitle-unit segmentation with BIO tagging and optical flow in a Seq2Seq formulation, on BOBSL and YouTube-ASL.

## Findings

- **Two datasets, two separately trained models** — BOBSL (manually-aligned subset: 60 videos, 40/10/10 split, 25 fps, ~45 min each) and YouTube-ASL (70/20/10). Not a joint model: parameter counts and training times differ per dataset. Combining them is named as future work.
- **Sequence Encoder beats Autoregressive** on both, and trains far faster (~14 h vs ~1 day). Their **Seq2Seq encoder-decoder with attention is the best model overall** and is where their headline numbers come from — Tables 2 and 3, not Table 1.

| model | dataset | F1 | IoU | % | table |
|---|---|---|---|---|---|
| Sequence Encoder | BOBSL | 0.58 | 0.60 | 2.50 | 1 |
| Sequence Encoder | YouTube-ASL | 0.56 | 0.58 | 0.70 | 1 |
| Autoregressive | BOBSL | 0.55 | 0.51 | 1.74 | 1 |
| Autoregressive | YouTube-ASL | 0.47 | 0.50 | 0.55 | 1 |
| Seq2Seq, no attention | BOBSL | 0.58 | 0.70 | 2.16 | 2 |
| **Seq2Seq + attention** | BOBSL | **0.60** | **0.74** | **1.03** | 2 |
| Seq2Seq, no attention | YouTube-ASL | 0.55 | 0.58 | 0.87 | 3 |
| **Seq2Seq + attention** | YouTube-ASL | **0.60** | **0.62** | **0.95** | 3 |

- **The corpora bias segmentation in opposite directions** for the Table 1 models: BOBSL over-segments (`%` 2.50 / 1.74), YouTube-ASL under-segments (`%` 0.70 / 0.55). They attribute this to differing annotation granularity between the two.
- **Attention largely removes that bias.** BOBSL goes 2.16 → 1.03 and YouTube-ASL 0.87 → 0.95, both close to the optimal 1, at the cost of ~2 days of training against 15–19 hours. So the segmentation-count problem is a property of their weaker models, not of subtitle supervision itself.
- **Pipeline**: ImageNet-pretrained ResNet-101 as a frozen feature extractor over optical flow (BOBSL's pre-computed; RAFT at a 10-frame stride for YouTube-ASL), then a BiLSTM predicting frame-level BIO over fixed 375-frame windows.

## Error types they report

From their §5.4, a qualitative analysis over predicted probability curves rather than a counted taxonomy — they name the categories and illustrate each with one figure, so there are no frequencies attached.

| error | what happens | cause they give |
|---|---|---|
| **False signing** | high `I` probability over a stretch with no signing at all | feature ambiguity: incidental motion, their example is raising and removing a hat, looks like signing to optical flow. Compounded by class imbalance biasing the model toward `I` |
| **Under-segmentation** | two distinct signing periods merged, the transition between them missed | difficulty separating signing from non-signing behaviour |
| **Boundary drift** | segments found but start and end times off; an exact match is rare | acknowledged as an open problem, §5.3 and Limitations |

What they say works: boundaries are found **without relying on pauses**, using the structure of the signing itself, including back-to-back subtitles with no gap between them (their Figures 2 and 3).

## Limitations they state

- **No comparison to phrase-based state of the art**, which they attribute to what the annotated datasets allow rather than to the method.
- **English subtitles only**, BOBSL and YouTube-ASL, so generalisation across sign languages is untested.
- **Reliance on optical flow** makes the model vulnerable to noisy or thin motion signal, notably occlusion and small movements.
- **Exact one-to-one timing** between predicted and true subtitles remains unsolved.
- **The gold itself is noisy**: manually placed subtitle boundaries are hard to delineate exactly, so some of the error is in the labels.

Two of these bear directly on our stage one. Their multilingual limitation is the gap our 56-language dev set is built to cover, and their optical-flow fragility is the argument for staying on pose.

## What we can borrow

- **Subtitle timings straight to BIO**, with no alignment step and no filtering for subtitle quality. That is the cheapest possible weak-label recipe for a YouTube pretraining stage, and it is the label format our pipeline already uses.
- **Optical flow for boundaries.** Both this and the 2023 E4 variant arrived at optical flow independently, which is some evidence it carries boundary signal that raw pose does not.
- **A bar to clear.** Their best YouTube-ASL model, Seq2Seq with attention, reaches F1 0.60, IoU 0.62 and `%` 0.95. That is the only published point on subtitle-supervised YouTube segmentation, so it is the number our pretraining stage is measured against — on a different split and different features, so as a scale rather than a score.
- **Under-segmentation is not inherent to subtitle labels.** Their weaker models give `%` 0.55–0.70 on YouTube-ASL, but attention brings it to 0.95. So a low `%` after pretraining is a sign the model is too weak or the decoding is wrong, not a fact about subtitle supervision.

Their features are ResNet-101 over RGB and flow, not MediaPipe pose, so the numbers are not comparable to ours — only the recipe transfers.
