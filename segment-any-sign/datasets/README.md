# Datasets

One folder per corpus, each with an `explore.py` and a generated `explore.md`. Raw data stays on the shared filesystem, never in this repo.

A corpus used by the benchmark also gets a `load.py`: **one clip list, one set of gold annotations, shared by every model.** [`public_dgs_corpus/load.py`](public_dgs_corpus/load.py) is the only one so far. [`youtube_sl25/load.py`](youtube_sl25/load.py) does the same for the corpus used for pretraining, which has no notebook here; see [`../experiments/`](../experiments/). Models differ in how they preprocess a pose before it enters the network — that part lives in [`../benchmark/`](../benchmark/) — but never in which clips they see or what they are scored against.

The benchmark needs **per-sign boundaries** and **poses**. Only the Public DGS Corpus has both today; every other corpus is missing exactly one thing.

| dataset | language | sign-level gold | poses | status |
|---|---|---|---|---|
| [public_dgs_corpus](public_dgs_corpus/) | DGS | ✅ 350,168 glosses | ✅ 50fps archive + 25fps TFDS | **benchmarked** |
| [ncslgr](ncslgr/) | ASL | ⚠️ 3 files only — 157 utterances, 1,350 glosses | ❌ | needs a DAI account for the remaining XML (published corpus is 1,887 utterances / 11,854 glosses), then pose extraction from the 2,636 local videos |
| [bsl_corpus](bsl_corpus/) | BSL | ✅ subset, 6,879 segments, 6.2 h | ❌ no video | awaiting 239 video files from UCL |
| [signsuisse](signsuisse/) | DSGS | ❌ not annotated | ✅ on the share | gloss annotation of the 500 prepared ELAN clips |
| [how2sign](how2sign/) | ASL | ❌ sentence-timed only, unreleased | ❌ | no per-sign timings exist; phrase level only |
| [mediapi_skel](mediapi_skel/) | LSF | ❌ none | ✅ skeletons | no glosses at all; phrase/subtitle level only |

The NCSLGR annotation we hold is the sample bundled with the SignStream XML parser's test resources, not corpus data — see `explore.py`'s `XML_DIR`. Enough to check pose quality and the BIO conversion, not to report a number.

## Where the data lives

Outside the repository. Code reads each folder from the paths file (see Locations in [`../README.md`](../README.md)): [`../paths.uzh.yaml`](../paths.uzh.yaml) holds the folders at UZH, and [`../paths.example.yaml`](../paths.example.yaml) says what each folder must contain.

| Key | What | Read by |
|---|---|---|
| `data.public_dgs_corpus.tfds` | TFDS folder with the 25 fps `dgs_corpus` build | `public_dgs_corpus/load.py` (`iter_clips`), `../benchmark/predict_dgs_2023.py` |
| `data.public_dgs_corpus.downloads` | download archive of that build: `.pose`, `.eaf`, `.cmdi` | `public_dgs_corpus/load.py` |
| `data.youtube_sl25.root` | poses, subtitles and metadata | `youtube_sl25/load.py` |
| `data.youtube_sl25.videos` | the `.mp4` files; optional | `../experiments/plot_segmentation.py` |
| `data.ncslgr.root` | download destination | `ncslgr/download.py` |
| `data.signsuisse.root` | poses, videos, the test metadata and the ELAN files of the 2023 model | `signsuisse/run_2023_segmentation.py`, `signsuisse/explore.py` |
| `data.bsl_corpus.root` | `info.pkl`, the index of Renz et al. (2021) | `bsl_corpus/explore.py` |
| `data.how2sign.root` | the main copy: `video_level/` and `sentence_level/` | `how2sign/explore.py` |
| `data.how2sign.parsign_cache` | pose `.npz` of the parsign project, which owns the folder | `how2sign/explore.py` |
| `data.mediapi_skel.tfds` | TFDS folder with the `mediapi_skel` build | `mediapi_skel/explore.py` |

The main How2Sign folder was not found on the shares on 2026-11-20: `data.how2sign.root` in `paths.uzh.yaml` still names the folder it had, which does not exist now. [`how2sign/explore.md`](how2sign/explore.md) is the report of 2026-08-11 and could not be regenerated. Its code cell shows the two folders as the notebook named them then, and the notebook stops in that cell until the key names a folder that exists.

The notebooks do not read the paths file yet: each `explore.py` names its folders in its configuration cell, near the top. The corpora that only a notebook reads (`bsl_corpus`, `how2sign`, `mediapi_skel`) have no key, and their folders are written there.

## Licensing

The free DAI account also unlocks the ASLLRP SignStream 3 corpus, which is a separate dataset and would get its own folder.

Licensing differs per corpus and is recorded in each `explore.md`. Internal research use is fine; a public release is not covered without permission.
