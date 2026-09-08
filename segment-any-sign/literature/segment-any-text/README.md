# Segment Any Text: A Universal Approach for Robust, Efficient and Adaptable Sentence Segmentation

- **Authors:** Markus Frohmann, Igor Sterner, Ivan Vulić, Benjamin Minixhofer, Markus Schedl
- **Venue:** EMNLP 2024 (main conference)
- **Link:** https://aclanthology.org/2024.emnlp-main.665/
- **Status:** read by Zifan

## Why it is here

The closest analogue to what we want, in text: one sentence segmentation model that is robust across languages and domains rather than trained per corpus — including on input without punctuation or casing, which is the text equivalent of the weak cues we work with.

## Contributions

Their claim is that no prior sentence segmenter achieves **all three** of robustness to missing punctuation, adaptability to new domains, and efficiency. One mechanism each:

- **Robustness** — a pretraining scheme that randomly removes punctuation-only tokens and strips all casing and punctuation from 10% of samples per batch, so the model cannot lean on the easiest surface cue.
- **Adaptability** — LoRA fine-tuning per domain, treating "what counts as a sentence boundary" as genuinely varying between domains rather than as something to standardise away.
- **Efficiency** — architectural changes giving a 3x speedup, plus a *limited lookahead* attention mask capped at N/L tokens per layer, since stacking L layers otherwise compounds the lookahead to N x L.

Evaluated across 85 languages and 8 corpora, beating LLM baselines by the widest margin on poorly formatted text.

## Their training stages, mapped onto ours

Their staged design is the part most worth borrowing. LoRA is their choice for the adaptation stage; ours only needs to be *some* finetuning, not necessarily parameter-efficient.

| | SaT (text) | Ours (sign) |
|---|---|---|
| **Pretrain** | mC4 web text, self-supervised, sampled uniformly from **85 languages**, from XLM-R weights. Corruptions: drop punctuation-only tokens, strip casing and punctuation in 10% of samples | YouTube, **subtitle level** — subtitle timings as weak boundaries. Scale TBD; not curated yet. Corruptions: fps, frame dropout, body-part dropout, and plausibly occlusion, framing, missing hands |
| **Mid-train** | *Supervised Mixture* — Universal Dependencies training sets, falling back to OPUS100 or NLLB where UD is missing, with further corruptions (lowercase + strip punctuation, emulating ASR output) | **Sentence level** — DGS train **61,057 phrases / 586 videos / 91 h**; How2Sign 79 h (sentence-timed, unreleased glosses) |
| **Post-train** | — | **Gloss level** — DGS train **335,929 signs** over the same 586 videos |
| **Adapt** | LoRA per domain (lyrics, legal), **max 10,000 sentences** per domain | Finetuning per corner case — short clips, no signer, fingerspelling, indexing. Not curated yet |
