"""Register our shared DGS clips as a dataset the 2026 training code can use.

The 2026 `DGSSegmentationDataset` expects `<corpus>/videos/<doc>/data.eaf` plus
poses keyed by video MD5. We have neither that tree nor the DGS videos the MD5s
come from — what we have is the download archive, keyed `<doc>_<person>.pose`.

Rather than rebuild the tree, this registers an adapter over
[`../datasets/public_dgs_corpus/load.py`](../datasets/public_dgs_corpus/load.py)
under the name **`dgs_shared`**, so `--datasets dgs_shared` trains on exactly the
clips, filters and gold the benchmark scores. That is the point: a training run
and a benchmark row must not be able to disagree about what the data is.

Everything downstream — windowing, augmentation, BIO construction, collation — is
upstream's `load_and_augment`, untouched. This class only supplies `self.items`.

    from experiments import dgs_dataset  # registers "dgs_shared"
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sign_language_segmentation.datasets.common import (  # noqa: E402
    BaseSegmentationDataset, Split, register_dataset)

from datasets.public_dgs_corpus import load as dgs_data  # noqa: E402


class SharedDGSDataset(BaseSegmentationDataset):
    """Public DGS Corpus, read through the benchmark's own loader."""

    dataset_name = "dgs_shared"

    def __init__(
        self,
        split: Split = Split.TRAIN,
        num_frames: int = 1024,
        # off by default, matching train.py. A velocity-on model takes 50x6 and a
        # velocity-off one 50x3, so a mismatch is fatal rather than silent — but
        # the defaults should still agree wherever the dataset is built directly.
        velocity: bool = False,
        fps_aug: bool = True,
        frame_dropout: float = 0.15,
        body_part_dropout: float = 0.1,
        phrase: str = "glosses",
        limit: int | None = None,
        backup: str = dgs_data.BACKUP,
        splits_path: str | None = None,
    ):
        self.split = split
        self.num_frames = num_frames
        self.velocity = velocity
        self.fps_aug = fps_aug
        self.frame_dropout = frame_dropout
        self.body_part_dropout = body_part_dropout
        self.phrase = phrase
        self.limit = limit
        self.backup = backup
        self.splits_path = splits_path

        self._init_split_tracking()
        self.items = []

        # clip_specs_native reads only .pose headers, so building the item list
        # stays cheap even though the bodies are 50fps
        for spec in dgs_data.clip_specs_native(str(split), backup, splits_path):
            # `limit` keeps the first N *annotated* clips of the split — enough to
            # exercise the whole path in seconds. Deterministic, so a limited run
            # is reproducible; never use one for a reported number.
            if limit is not None and len(self.items) >= limit:
                break
            spans = dgs_data.sign_phrase_spans(spec["sentences"], phrase=phrase)
            if limit is not None and not spans["sign"]:
                continue
            self._track_and_filter(spec["id"], split, {
                "id": spec["id"],
                "pose_path": spec["pose_path"],
                "fps": spec["fps"],
                "total_frames": spec["total_frames"],
                # upstream wants milliseconds
                "glosses": _to_ms(spans["sign"]),
                "sentences": _to_ms(spans["phrase"]),
            })

        limited = f", limited to {limit}" if limit is not None else ""
        print(f"SharedDGSDataset({split}): {len(self.items)} videos, "
              f"phrase={phrase}{limited}")

    def get_split_manifest(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "source": "datasets/public_dgs_corpus/load.py (archive, native fps)",
            "phrase": self.phrase,
            "splits": {s.value: sorted(ids) for s, ids in self._all_split_ids.items()},
        }

    @classmethod
    def from_args(cls, split: Split, args: Namespace, **augment_kwargs) -> SharedDGSDataset:
        return cls(split=split, phrase=getattr(args, "phrase", "glosses"),
                   limit=getattr(args, "limit", None),
                   backup=getattr(args, "backup", dgs_data.BACKUP),
                   splits_path=getattr(args, "splits_path", None),
                   **augment_kwargs)


def _to_ms(spans) -> list[dict[str, float]]:
    return [{"start": s["start_time"] * 1000, "end": s["end_time"] * 1000} for s in spans]


register_dataset(SharedDGSDataset.dataset_name, SharedDGSDataset)
