# AGENTS.md

Instructions for coding agents working in this repository on the `segment-any-sign` branch.

This file holds conventions only. It contains no project status, plan or list of known
problems on purpose: read those from the READMEs under `segment-any-sign/` and from the code,
and form your own view.

## Layout

- `segment-any-sign/` is the Segment Any Sign project workspace: benchmark, metrics,
  experiments, dataset loaders and literature notes. New project code goes here. Start with
  `segment-any-sign/README.md` and the README of the subfolder you are working in; they
  document the current layout, setup, commands and conventions.
- `sign_language_segmentation/` is the upstream package, synced from
  `sign-language-processing/segmentation`. Treat it as someone else's code: change it only to
  fix a bug, in a commit of its own, and say so in the pull request so the fix can be sent
  upstream.

## Before changing anything

- Read the code you are about to change and its call sites, and the README of that folder.
- The code, the tests and the recorded results are the truth; the READMEs are a guide. When
  they disagree, say which file says what, treat the code as correct, and fix the
  documentation in the same change.
- Make the smallest change that fully solves the task. Leave unrelated code as it is and list
  other improvements you noticed instead of making them.

## Commands

Environment setup is described in `segment-any-sign/README.md`. Project scripts are written to
be run from inside `segment-any-sign/`.

- Metric tests: `cd segment-any-sign && pytest metrics/`
- Upstream package tests: `pytest` from the repository root
- Lint: `ruff check .` from the repository root
- Benchmark and scoring: see `segment-any-sign/benchmark/README.md`
- Training and evaluation of runs: see `segment-any-sign/experiments/README.md` and the
  docstring of `segment-any-sign/experiments/train.py`

Training, pose extraction and anything that scans a whole corpus are cluster jobs. Use the
quick-check options the scripts provide for a fast sanity check, and ask before submitting a
job. If the data a command needs is not reachable from the session, say so; do not replace it
with mock data.

## Evaluation rules

These protect the comparability of every number in the repository.

- One evaluation protocol for every model, as documented in
  `segment-any-sign/benchmark/README.md`. An experiment may change the model; it may not
  change how it is scored. Do not alter the list of videos, the gold annotations, the frame
  conversion or a metric definition to improve a result.
- Any change to `segment-any-sign/metrics/` or to the benchmark protocol needs my explicit
  approval first, with the reason recorded where the repository records deviations.
- Experiments are decided on validation data. The test split is used only for results that
  are going to be reported, and only when I ask.
- Tests are the specification. Before adding, changing, deleting or skipping a test, explain
  the change and wait for my approval. When a test fails, say whether the code or the test is
  wrong and why; never weaken an assertion to make it pass.
- Any change to sampling, resampling, frame dropout, windowing or chunking must keep each
  frame's label equal to what the annotation says at that frame's real time, and comes with a
  test that checks it.
- Every run is recorded the way `segment-any-sign/experiments/README.md` describes. Numbers in
  tables come from the scripts' output files, never typed from memory.

## Code conventions

- Do not commit data, poses, checkpoints, model weights, logs or experiment-tracker
  directories. Respect the `.gitignore` files.
- Do not add credentials, personal absolute paths or machine-specific settings to committed
  code; take paths from arguments, environment variables or configuration.
- Do not edit generated files by hand (for example `datasets/*/explore.md`); edit the script
  that produces them and regenerate.
- Follow the style of the surrounding code. Lint must pass.

## Git workflow

The base branch is `segment-any-sign`.

1. Update: `git switch segment-any-sign && git pull`
2. Create a short-lived branch for the task: `git switch -c <short-name>`
3. Commit in small logical steps, one change per commit.
4. Run the relevant tests and the linter.
5. Push the branch: `git push -u origin <short-name>`
6. Open a pull request against `segment-any-sign` describing what changed, why, what you ran
   and what you could not verify. Then stop: I review and merge.

Rules:

- Never commit or push directly to `segment-any-sign` or `main`, never force-push, and never
  rewrite commits that have been pushed. Do not merge your own pull request.
- Syncing from the upstream repository is a pull request of its own, with nothing else in it.
- `segment-any-sign/latex` is a submodule that points to the paper draft. Leave it alone
  unless I ask; a pointer update is a commit of its own.
- Start experiments from a committed state, because each run records the commit it ran from.

Commit messages: a short imperative summary line, then a body saying what changed, why, and
how it was verified.

Authorship: every commit is authored only by the git identity already configured on the
machine. Add no `Co-Authored-By` trailer, no "Generated with" line, no session link and no
other tool or model attribution, in commit messages, pull request titles or pull request
descriptions. Do not change `git config user.name` or `user.email`, and do not use
`--author`.

## Reporting back

Say what you changed, what you ran and what the output was, and what you did not or could not
verify. Report tests as passing only if you ran them in this session.

Ask before: submitting cluster jobs, touching the test split, changing tests, metrics or the
evaluation protocol, modifying the upstream package, deleting files, or anything that cannot
be undone.