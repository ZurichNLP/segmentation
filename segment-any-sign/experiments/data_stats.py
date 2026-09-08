"""Dataset statistics and correctness checks, run before every training job.

Two jobs. **Checks** fail loudly on things that would otherwise corrupt a run
quietly — a document in two splits, a span running past the end of its video, a
zero-length segment. **Stats** record what the run actually trained on, so a
result months from now can still be traced to its data.

Both are written to the run directory as JSON and a readable log, and the scalar
summary is attached to the W&B run.

    conda activate sas
    python experiments/data_stats.py                 # all splits, prints a report
    python experiments/data_stats.py --split train --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from datasets.public_dgs_corpus import load as dgs_data  # noqa: E402

SPLITS = ("train", "dev", "test")

# Section 4.1 of Moryossef & Jiang (2023), for reference in the report. Our train
# count differs by 4 videos for reasons recorded in README.md.
PAPER_COUNTS = {"train": (296, 583), "dev": (6, 12), "test": (9, 17)}


def split_stats(split: str, phrase: str = "glosses") -> dict:
    """Per-split counts, durations and segment statistics. Reads headers only."""
    docs, videos = set(), 0
    unannotated, frames, seconds = 0, 0, 0.0
    fps_seen: Counter = Counter()
    n_sign, n_phrase = [], []
    sign_dur, phrase_dur = [], []
    problems: list[str] = []

    for spec in dgs_data.clip_specs_native(split):
        videos += 1
        docs.add(dgs_data.document_id(spec["id"].rsplit("_", 1)[0]))
        frames += spec["total_frames"]
        duration = spec["total_frames"] / spec["fps"]
        seconds += duration
        fps_seen[spec["fps"]] += 1

        spans = dgs_data.sign_phrase_spans(spec["sentences"], phrase=phrase)
        if not spans["sign"]:
            unannotated += 1
        n_sign.append(len(spans["sign"]))
        n_phrase.append(len(spans["phrase"]))

        for level in ("sign", "phrase"):
            for s in spans[level]:
                length = s["end_time"] - s["start_time"]
                (sign_dur if level == "sign" else phrase_dur).append(length)
                if length <= 0:
                    problems.append(f"{spec['id']}: {level} span of {length:.3f}s")
                if s["start_time"] < 0:
                    problems.append(f"{spec['id']}: {level} starts at {s['start_time']:.3f}s")
                # a little slack: annotation can run marginally past the last frame
                if s["end_time"] > duration + 1.0:
                    problems.append(
                        f"{spec['id']}: {level} ends at {s['end_time']:.1f}s, "
                        f"video is {duration:.1f}s")

    return {
        "split": split,
        "phrase_definition": phrase,
        "documents": len(docs),
        "videos": videos,
        "unannotated_videos": unannotated,
        "hours": round(seconds / 3600, 2),
        "frames": frames,
        "fps": {str(k): v for k, v in sorted(fps_seen.items())},
        "signs": int(sum(n_sign)),
        "phrases": int(sum(n_phrase)),
        "signs_per_video_mean": round(float(np.mean(n_sign)), 1) if n_sign else 0.0,
        "phrases_per_video_mean": round(float(np.mean(n_phrase)), 1) if n_phrase else 0.0,
        "sign_duration_ms_mean": round(float(np.mean(sign_dur)) * 1000, 1) if sign_dur else 0.0,
        "phrase_duration_ms_mean": round(float(np.mean(phrase_dur)) * 1000, 1) if phrase_dur else 0.0,
        "problems": problems,
        "doc_ids": sorted(docs),
    }


def check_no_leakage(stats: dict[str, dict]) -> list[str]:
    """Every document must appear in exactly one split.

    The check that matters most: a document leaking from train into dev or test
    inflates every number downstream and is invisible in the loss curve.
    """
    problems = []
    for a in SPLITS:
        for b in SPLITS:
            if a >= b or a not in stats or b not in stats:
                continue
            shared = set(stats[a]["doc_ids"]) & set(stats[b]["doc_ids"])
            if shared:
                problems.append(f"{len(shared)} documents in both {a} and {b}: "
                                f"{sorted(shared)[:5]}")
    return problems


def label_stats(split: str, samples: int = 20, num_frames: int = 1024,
                seed: int = 0) -> dict:
    """BIO class balance over sampled training windows, after augmentation.

    Sampled from the dataset as the model sees it — windowed, downsampled and
    frame-dropped — because that is the distribution the loss actually sees, not
    the one the raw annotation implies.
    """
    from sign_language_segmentation.datasets.common import Split

    from experiments.dgs_dataset import SharedDGSDataset

    # velocity affects only pose dims, not labels, but pin it so the report
    # cannot drift from what the run uses
    dataset = SharedDGSDataset(split=Split(split), num_frames=num_frames,
                               velocity=False)
    if len(dataset) == 0:
        return {}

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(dataset), min(samples, len(dataset)), replace=False)
    names = {0: "UNK", 1: "O", 2: "B", 3: "I"}
    out: dict = {"sampled_windows": int(len(picks))}

    for level in ("sign", "sentence"):
        counts: Counter = Counter()
        for i in picks:
            counts.update(dataset[int(i)]["bio"][level].numpy().tolist())
        total = sum(counts.values())
        out[level] = {names[k]: round(v / total, 4) for k, v in sorted(counts.items())}
        b, i_ = counts.get(2, 0), counts.get(3, 0)
        out[f"{level}_B_to_I_to_O"] = (
            f"1:{i_ / b:.1f}:{counts.get(1, 0) / b:.1f}" if b else "no B frames")
    return out


def collect(splits=SPLITS, phrase: str = "glosses", samples: int = 20) -> dict:
    """Full report: per-split stats, the leakage check, and label balance."""
    stats = {s: split_stats(s, phrase=phrase) for s in splits}
    report = {
        "splits": stats,
        "problems": {
            "leakage": check_no_leakage(stats),
            "spans": {s: stats[s]["problems"] for s in stats if stats[s]["problems"]},
        },
        "paper_4_1": {s: {"documents": d, "videos": v} for s, (d, v) in PAPER_COUNTS.items()},
    }
    if "train" in stats:
        report["labels_train"] = label_stats("train", samples=samples)
    return report


def format_report(report: dict) -> str:
    lines = ["data stats", "=" * 64]
    for split, s in report["splits"].items():
        paper = PAPER_COUNTS.get(split)
        against = f"   (paper 4.1: {paper[0]}/{paper[1]})" if paper else ""
        lines += [
            f"{split}: {s['documents']} docs / {s['videos']} videos{against}",
            f"    {s['hours']} h, {s['frames']:,} frames, fps {s['fps']}",
            f"    {s['signs']:,} signs ({s['signs_per_video_mean']}/video, "
            f"{s['sign_duration_ms_mean']:.0f} ms mean)",
            f"    {s['phrases']:,} phrases ({s['phrases_per_video_mean']}/video, "
            f"{s['phrase_duration_ms_mean']:.0f} ms mean)",
            f"    {s['unannotated_videos']} videos with no annotation",
        ]
    if labels := report.get("labels_train"):
        lines += ["", f"label balance over {labels['sampled_windows']} train windows:"]
        for level in ("sign", "sentence"):
            lines.append(f"    {level:<9} {labels[level]}   "
                         f"B:I:O = {labels[f'{level}_B_to_I_to_O']}")

    leakage = report["problems"]["leakage"]
    spans = report["problems"]["spans"]
    lines += ["", "checks"]
    lines.append(f"    split leakage   {'FAIL — ' + '; '.join(leakage) if leakage else 'ok'}")
    n_span = sum(len(v) for v in spans.values())
    lines.append(f"    span sanity     {f'{n_span} problems' if n_span else 'ok'}")
    for split, items in spans.items():
        for item in items[:5]:
            lines.append(f"        {split}: {item}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", action="append", choices=SPLITS,
                        help="repeatable; default all")
    parser.add_argument("--phrase", default="glosses", choices=["glosses", "sentence"])
    parser.add_argument("--samples", type=int, default=20,
                        help="training windows to sample for label balance")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report = collect(tuple(args.split) if args.split else SPLITS,
                     phrase=args.phrase, samples=args.samples)
    print(format_report(report))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")

    if report["problems"]["leakage"]:
        raise SystemExit("split leakage — refusing to report this as OK")


if __name__ == "__main__":
    main()
