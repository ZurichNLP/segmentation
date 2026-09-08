# Datasets

One folder per corpus, each with an `explore.py` and a generated `explore.md`. Raw data stays on the shared filesystem, never in this repo.

A corpus used by the benchmark also gets a `load.py`: **one clip list, one set of gold annotations, shared by every model.** Only [`public_dgs_corpus/load.py`](public_dgs_corpus/load.py) exists so far. Models differ in how they preprocess a pose before it enters the network — that part lives in [`../benchmark/`](../benchmark/) — but never in which clips they see or what they are scored against.

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

```
/shares/sign-language.ebling.cl.uzh/NCSLGR/
/shares/sign-language.ebling.cl.uzh/BSL_Corpus/
/shares/sign-language.ebling.cl.uzh/Signsuisse/
/shares/iict-sp2.ebling.cl.uzh/common/How2Sign/
/shares/iict-sp2.ebling.cl.uzh/common/tensorflow_datasets/{dgs_corpus,mediapi_skel}/
/shares/iict-sp2.ebling.cl.uzh/zifjia/backups/tensorflow_datasets_2/downloads/  (DGS eaf/cmdi)
```

## Licensing

The free DAI account also unlocks the ASLLRP SignStream 3 corpus, which is a separate dataset and would get its own folder.

Licensing differs per corpus and is recorded in each `explore.md`. Internal research use is fine; a public release is not covered without permission.
