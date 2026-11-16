# AGENTS.md

Instructions for coding agents working in this repository on the `segment-any-sign` branch.
"The user" below is the person you are working for in the session.

This file holds conventions only. It contains no project status, plan or list of known
problems on purpose: read those from the READMEs under `segment-any-sign/` and from the code,
and form your own view.

## Layout

- `segment-any-sign/` is the Segment Any Sign project workspace: benchmark, metrics,
  experiments, dataset loaders and literature notes. New project code goes here. Start with
  `segment-any-sign/README.md` and the README of the subfolder you are working in.
- `sign_language_segmentation/` and the root files that also exist in the upstream repository
  (`pyproject.toml`, `dist/`, the Dockerfiles, `.github/`, `.gitignore`, `README.md`,
  `LICENSE`, `.env.example`) are the upstream package, synced from
  `sign-language-processing/segmentation`. Treat them as someone else's code: change them only
  to fix a bug, in a commit of its own, and say so in the pull request so the fix can be sent
  upstream.
- `AGENTS.md`, `CLAUDE.md`, `.claude/` and `.gitmodules` at the root belong to this project.

## Before changing anything

- Read the code you are about to change and its call sites, and the README of that folder.
- The code, the tests and the recorded results are the truth; the READMEs are a guide. When
  they disagree, say which file says what, treat the code as correct, and fix the
  documentation in the same change.
- When a change alters behaviour, commands or results, update the README of that folder in
  the same change.
- Make the smallest change that fully solves the task. Leave unrelated code as it is and list
  other improvements you noticed instead of making them.

## Commands

Environment setup is in the Setup section of `segment-any-sign/README.md`. Project scripts are
run from inside `segment-any-sign/`.

- Metric tests: `cd segment-any-sign && pytest metrics/`
- Upstream package tests: `pytest` from the repository root. They need the dependencies that
  `.github/workflows/test.yaml` installs.
- Lint: `ruff check <the files you changed>`, compared with the same command on the base
  branch. `ruff` is in the `dev` extra of `pyproject.toml`; if it is missing, install it
  with `pip install ruff`.
- Benchmark and scoring: see `segment-any-sign/benchmark/README.md`. The commands in that
  README use the test split and the prediction scripts default to it, so always pass
  `--split` explicitly (see Evaluation rules).
- Training: see the docstring of `segment-any-sign/experiments/train.py` and
  `segment-any-sign/experiments/README.md`.

Training, pose extraction and anything that reads a whole corpus need the cluster and its
data. Where a script offers a quick check (for example `--dry-run` and `--limit` in
`experiments/train.py`), use it first, and ask the user before submitting a job.

If the environment or the data is not available where you are running, say which checks you
could not run. Do not replace missing data with mock data.

## Evaluation rules

These protect the comparability of every number in the repository.

- One evaluation protocol for every model, as documented in
  `segment-any-sign/benchmark/README.md`. An experiment may change the model; it may not
  change how it is scored. Deviate only for a demonstrable bug, record why where that README
  records deviations, and ask the user first.
- Use the validation split, and state the split explicitly wherever a script takes one
  (`--split` in the prediction scripts, `--eval-split` in `experiments/train.py`). Use the
  test split only when the user asks for it.
- When you add or change code that can alter a number without failing, add a test: metrics,
  label construction, sampling, windowing and chunking, decoding, and the construction of
  gold segments and splits. A bug fixed in that code comes with a test that fails before the
  fix. Tests use small hand-made inputs, as the metric tests do. For other code, say in the
  commit how the change was checked.
- Any change to sampling, resampling, frame dropout, windowing or chunking must keep each
  frame's label equal to what the annotation says at that frame's real time.
- Tests are the specification. Before changing, deleting or skipping an existing test,
  explain the change and wait for the user's approval. A new test is proposed in the
  pull request, where the review is the approval. When a test fails, say whether the code or
  the test is wrong and why; never weaken an assertion to make it pass.
- A new test lives next to the code it tests as `test_*.py`, like
  `segment-any-sign/metrics/test_segmentation.py`, and the pull request says how to run it.
- Every training run leaves what is needed to repeat it. `experiments/train.py` writes this
  for each run, and `experiments/add_result_row.py` adds the row of a pretraining run to its
  table in `experiments/README.md`; their docstrings describe what is recorded. Numbers in
  tables come from the scripts' output files.

## Code conventions

- Do not commit data, poses, checkpoints, model weights, logs or experiment-tracker
  directories. Respect the `.gitignore` files.
- Prediction files are the exception. The validation predictions of a recorded run are
  committed under `segment-any-sign/experiments/predictions/`, in the same commit as the
  run's row in `experiments/README.md`. Test predictions under `benchmark/predictions/` are
  ignored and stay uncommitted.
- Keep each data location as a named constant at the top of the module that owns it, with a
  command-line option where a script needs to override it. Do not add credentials, and do
  not hard-code a directory that belongs to one user. Where existing code does, do not copy
  the pattern; changing it is a task of its own.
- Do not edit generated files by hand (for example `datasets/*/explore.md` and its
  `explore_files/`); edit the script that produces them, regenerate, and commit the result.
- Follow the style of the surrounding code, and add no new lint errors in the files you
  touch, compared with the base branch.

## Git workflow

The base branch is `segment-any-sign`.

1. Get the latest base: `git fetch origin`
2. Create a short-lived branch for the task from it:
   `git switch -c <short-name> --no-track origin/segment-any-sign`
3. Commit in small steps, one logical change per commit.
4. Run the tests that cover what you changed, and the linter on the files you changed.
5. Push the branch: `git push -u origin <short-name>`
6. Open a pull request against `segment-any-sign` describing what changed, why, what you ran
   and what you could not verify. Then stop: the user reviews and merges.

Rules:

- Never commit or push directly to `segment-any-sign` or `main`, never force-push, and never
  rewrite commits that have been pushed. Do not merge your own pull request.
- Syncing from the upstream repository is a pull request of its own, with nothing else in it.
- `segment-any-sign/latex` is a submodule that points to the paper draft. Leave it alone
  unless the user asks; a pointer update is a commit of its own.
- Start experiments from a committed state, because each run records the commit it ran from.

Commit messages: a capitalised imperative summary line of at most 72 characters, a blank
line, then a body that says what changed and why, and how it was tested or checked.

Authorship: every commit is authored only by the git identity already configured on the
machine. Add no `Co-Authored-By` trailer, no "Generated with" line, no session link and no
other tool or model attribution, in commit messages, pull request titles or pull request
descriptions. Do not change `git config user.name` or `user.email`, and do not use
`--author`.

## Reporting back

Say what you changed, what you ran and what the output was, and what you did not or could not
verify. Report tests as passing only if you ran them in this session.

Ask the user before: submitting cluster jobs, touching the test split, changing tests,
the evaluation protocol or a metric definition, modifying the upstream package, deleting
files, or anything that cannot be undone.
