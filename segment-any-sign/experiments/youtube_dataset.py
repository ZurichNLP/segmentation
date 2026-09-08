"""Register YouTube-SL-25 as a training dataset, phrase level only.

Subtitle cues supervise the phrase head; the sign head has no signal here. That
distinction has to be made in the *labels*, not just skipped downstream: giving
the sign head all-`O` would teach it that nothing is ever a sign, which is worse
than no pretraining at all. Instead every sign frame is `UNK`, which upstream's
mask removes from the loss entirely.

    from experiments import youtube_dataset   # registers "youtube"
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sign_language_segmentation.datasets.common import (  # noqa: E402
    BaseSegmentationDataset, Split, load_and_augment, register_dataset)
from sign_language_segmentation.utils.bio import BIO  # noqa: E402

from datasets.youtube_sl25 import load as yt  # noqa: E402


class YouTubeSubtitleDataset(BaseSegmentationDataset):
    """~38,850 videos of weak, subtitle-timed phrase boundaries."""

    dataset_name = "youtube"

    def __init__(
        self,
        split: Split = Split.TRAIN,
        num_frames: int = 1024,
        velocity: bool = False,
        fps_aug: bool = False,
        frame_dropout: float = 0.0,
        body_part_dropout: float = 0.0,
        dev_videos: int = yt.DEV_VIDEOS,
        **_ignored,
    ):
        self.split = split
        self.num_frames = num_frames
        self.velocity = velocity
        self.fps_aug = fps_aug
        self.frame_dropout = frame_dropout
        self.body_part_dropout = body_part_dropout

        self._init_split_tracking()
        self.items = []
        name = "dev" if split in (Split.DEV, "dev", "validation") else "train"
        for spec in yt.clip_specs(name, dev_videos=dev_videos):
            self.items.append({
                "id": spec["id"],
                "pose_path": spec["pose_path"],
                "fps": spec["fps"],
                "total_frames": spec["total_frames"],
                "glosses": [],      # no sign supervision; masked in __getitem__
                "sentences": [{"start": c["start_time"] * 1000,
                               "end": c["end_time"] * 1000}
                              for c in spec["sentences"]],
            })
            self._all_split_ids[split].append(spec["id"])

        hours = sum(i["total_frames"] / i["fps"] for i in self.items) / 3600
        cues = sum(len(i["sentences"]) for i in self.items)
        print(f"YouTubeSubtitleDataset({split}): {len(self.items):,} videos, "
              f"{hours:,.0f} h, {cues:,} cues")

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        try:
            datum = load_and_augment(
                pose_path=item["pose_path"], fps=item["fps"],
                total_frames=item["total_frames"], signs=item["glosses"],
                sentences=item["sentences"], split=self.split,
                num_frames=self.num_frames, velocity=self.velocity,
                fps_aug=self.fps_aug, frame_dropout=self.frame_dropout,
                body_part_dropout=self.body_part_dropout)
        except Exception as error:
            # a truncated pose should cost one sample, not the whole run
            print(f"skipping {item['id']}: {type(error).__name__}: {error}")
            return self[(index + 1) % len(self)]

        # UNK, not O: upstream masks UNK out of the loss, so the sign head gets
        # no gradient at all here rather than learning "never a sign"
        datum["bio"]["sign"] = torch.full_like(datum["bio"]["sign"], BIO["UNK"])
        return datum

    def get_split_manifest(self) -> dict:
        return {"dataset": self.dataset_name, "source": yt.ROOT,
                "level": "phrase only (sign head masked)",
                "splits": {s.value: sorted(ids)
                           for s, ids in self._all_split_ids.items()}}

    @classmethod
    def from_args(cls, split: Split, args: Namespace, **augment_kwargs):
        return cls(split=split,
                   dev_videos=getattr(args, "dev_videos", yt.DEV_VIDEOS),
                   **augment_kwargs)


register_dataset(YouTubeSubtitleDataset.dataset_name, YouTubeSubtitleDataset)
