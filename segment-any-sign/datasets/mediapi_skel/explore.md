# MEDIAPI-SKEL (LSF) — assessment

**Conclusion: subtitle/phrase level only — it has no gloss annotation at all.**
Recorded here so we do not re-investigate it later.

MEDIAPI-SKEL is a 2D-skeleton database of French Sign Language with aligned
French subtitles, built from content produced by **deaf journalists at
Média'Pi!** — original LSF content rather than laboratory recordings or
interpreted TV, which makes it unusually naturalistic.

Paper: [Bull, Braffort & Gouiffès, LREC 2020](https://aclanthology.org/2020.lrec-1.743/)

## Why it cannot support sign-level evaluation

The alignment is **subtitle to video**. There are no glosses, so there is
nothing to derive sign boundaries from.

This matters because MEDIAPI-SKEL is already wired into the 2023 segmentation
codebase, and the way it is wired is misleading. In the pre-refactor
`src/data.py`, `process_datum_mediapi_skel` turns **each subtitle into a
"sentence" containing exactly one "sign"**:

```python
segments = [[{"start_time": ..., "end_time": ...}]
            for start_time, end_time in zip(subtitles.start_time, subtitles.end_time)]
```

So the sign tier is synthetic — one fake sign per subtitle. **Any sign-level
score computed on MEDIAPI-SKEL is really measuring subtitle units**, and is not
comparable with sign-level scores on DGS or BSL Corpus. Worth remembering when
reading the LSF transfer results from the 2023 work.

## Statistics reported in the paper


```python
print("MEDIAPI-SKEL (LREC 2020)\n")
print("  language            LSF (French Sign Language)")
print("  source              Média'Pi!, content by deaf journalists")
print("  videos              368 subtitled videos")
print("  duration            27 hours")
print("  subtitle vocabulary 17k tokens (French)")
print("  keypoints           OpenPose: 25 body + 2x21 hand + 70 face")
print("  gloss annotation    NONE — subtitle alignment only")
```

    MEDIAPI-SKEL (LREC 2020)
    
      language            LSF (French Sign Language)
      source              Média'Pi!, content by deaf journalists
      videos              368 subtitled videos
      duration            27 hours
      subtitle vocabulary 17k tokens (French)
      keypoints           OpenPose: 25 body + 2x21 hand + 70 face
      gloss annotation    NONE — subtitle alignment only


## What we hold on the server

A TFDS build (2.8 GB), the same one the 2023 experiments used:

`/shares/iict-sp2.ebling.cl.uzh/common/tensorflow_datasets/mediapi_skel/`

**No raw copy exists** — I searched `~/sp2/zifjia/backups` and the shared
directories; the only hits were unrelated `*.mediapipe.*` files matching on the
substring. Unlike the DGS build, that is not a problem here: the features are
`metadata`, `id`, `pose`, `subtitles` with **no `paths` field**, so poses and
subtitles are embedded in the tfrecords and the build is self-contained. It does
not depend on files elsewhere on disk.


```python
import json
import sys
from pathlib import Path

# paths.py reads every location from the paths file. It is
# in segment-any-sign/, found from the working directory.
sys.path.insert(0, next(
    str(base / "segment-any-sign")
    for base in [Path.cwd(), *Path.cwd().parents]
    if (base / "segment-any-sign" / "paths.py").is_file()))
import paths

TFDS_DIR = paths.data("mediapi_skel.tfds") / "mediapi_skel"

configs = sorted(p for p in TFDS_DIR.glob("*/*/dataset_info.json"))
for path in configs:
    info = json.loads(path.read_text())
    print(f"config {path.parts[-3]}  version {info.get('version')}")
    total = 0
    for split in info.get("splits", []):
        n = sum(int(x) for x in split.get("shardLengths", [])) if split.get("shardLengths") else 0
        total += n
        print(f"  {split['name']:<12} {n:>4} examples")
    print(f"  {'total':<12} {total:>4}")

features = sorted(TFDS_DIR.glob("*/*/features.json"))
if features:
    tree = json.loads(features[0].read_text())

    def walk(node, prefix=""):
        for key, value in node.get("featuresDict", {}).get("features", {}).items():
            print(f"    {prefix}{key}")
            walk(value, prefix + key + ".")

    print("\n  features:")
    walk(tree)
```

    config holistic-0  version 1.0.0
      train         277 examples
      validation     40 examples
      test           50 examples
      total         367
    
      features:
        metadata
        metadata.duration
        metadata.height
        metadata.frames
        metadata.fps
        metadata.width
        id
        pose
        subtitles


## Verdict and notes

- **No gloss annotation exists**, so this cannot be used for sign-level
  segmentation — not a matter of access, the labels do not exist.
- **Usable for phrase/subtitle-level segmentation**, which is what it was built
  for. Note the paper's own caveat that subtitles do not always correspond to
  sentences: they frequently split mid-sentence.
- **Beware the synthetic sign tier** in the old codebase (above). Any historical
  "sign F1 on MEDIAPI" number measures subtitle units.
- Local TFDS build is `holistic-0`; the split sizes below sum to 367 against the
  paper's 368 videos — a one-video difference worth checking before quoting
  corpus size.
- Useful as the **LSF** point in a multilingual benchmark, and it pairs with the
  LREC 2026 LSF/LSM paper in our related work.

### Running

```bash
jupytext --to ipynb --execute datasets/mediapi_skel/explore.py -o - \
  | jupyter nbconvert --stdin --to markdown --output explore --output-dir datasets/mediapi_skel
```
