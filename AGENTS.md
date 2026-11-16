# AGENTS.md
 
Instructions for coding agents working in this repository on the `segment-any-sign` branch.
"The maintainer" below is Gerard Sant.
 
This file holds conventions only. It contains no project status, plan or list of known
problems on purpose: read those from the READMEs under `segment-any-sign/` and from the code,
and form your own view.
 
## Layout
 
- `segment-any-sign/` is the Segment Any Sign project workspace: benchmark, metrics,
  experiments, dataset loaders and literature notes. New project code goes here. Start with
  `segment-any-sign/README.md` and the README of the subfolder you are working in.
- `sign_language_segmentation/` and the other files at the repository root (`pyproject.toml`,
  `dist/`, the Dockerfiles, `.github/`) are the upstream package, synced from
  `sign-language-processing/segmentation`. Treat them as someone else's code: change them only
  to fix a bug, in a commit of its own, and say so in the pull request so the fix can be sent
  upstream.
- `AGENTS.md`, `CLAUDE.md` and `.claude/` at the root are this project's agent configuration.
## Before changing anything
 
- Read the code you are about to change and its call sites, and the README of that folder.
- The code, the tests and the recorded results are the truth; the READMEs are a guide. When
  they disagree, say which file says what, treat the code as correct, and fix the
  documentation in the same change.
- Make the smallest change that fully solves the task. Leave unrelated code as it is and list
  other improvements you noticed instead of making them.
## Commands
 
Environment setup is in the Setup section of `segment-any-sign/README.md`. Project scripts are
run from inside `segment-any-sign/`.
 
- Metric tests: `cd segment-any-sign && pytest metrics/`
- Upstream package tests: `pytest` from the repository root. They need the dependencies that
  `.github/workflows/test.yaml` installs.
- Lint: `ruff check <the files you changed>`. `ruff` comes from the `dev` extra in
  `pyproject.toml`.
- Benchmark and scoring: see `segment-any-sign/benchmark/README.md`. The prediction scripts
  default to the test split, so always pass `--split` explicitly (see Evaluation rules).
- Training: see the docstring of `segment-any-sign/experiments/train.py` and
  `segment-any-sign/experiments/README.md`.
Training, pose extraction and anything that reads a whole corpus need the cluster and its
data. Where a script offers a quick check (for example `--dry-run` and `--limit` in
`experiments/train.py`), use it first, and ask the maintainer before submitting a job.
 
If the environment or the data is not available where you are running, say which checks you
could not run. Do not replace missing data with mock data.
 
## Evaluation rules
 
These protect the comparability of every number in the repository.
 
- One evaluation protocol for every model, as documented in
  `segment-any-sign/benchmark/README.md`. An experiment may change the model; it may not
  change how it is scored. Deviate only for a demonstrable bug, record why where that README
  records deviations, and ask the maintainer first.
- Use the validation split, and always pass the split explicitly. Use the test split only
  when the maintainer asks for it.
- Any change to sampling, resampling, frame dropout, windowing or chunking must keep each
  frame's label equal to what the annotation says at that frame's real time, and comes with a
  test that checks it.
- Tests are the specification. Before adding, changing, deleting or skipping a test, explain
  the change and wait for the maintainer's approval. When a test fails, say whether the code
  or the test is wrong and why; never weaken an assertion to make it pass.
- A new test lives next to the code it tests as `test_*.py`, like
  `segment-any-sign/metrics/test_segmentation.py`, and the pull request says how to run it.
- Every training run leaves what is needed to repeat it. `experiments/train.py` writes this
  for each run, and `experiments/add_result_row.py` adds result rows to the README table;
  their docstrings describe what is recorded. Numbers in tables come from those output files.
## Code conventions
 
- Do not commit data, poses, checkpoints, model weights, logs or experiment-tracker
  directories. Respect the `.gitignore` files.
- In new or modified code, take paths from arguments, environment variables or configuration
  instead of adding personal absolute paths or credentials.
- Do not edit generated files by hand (for example `datasets/*/explore.md`); edit the script
  that produces them and regenerate.
- Follow the style of the surrounding code, and add no new lint errors in the files you
  touch.
## Git workflow
 
The base branch is `segment-any-sign`.
 
1. Update: `git switch segment-any-sign && git pull`
2. Create a short-lived branch for the task: `git switch -c <short-name>`
3. Commit in small steps, one logical change per commit.
4. Run the tests that cover what you changed, and the linter on the files you changed.
5. Push the branch: `git push -u origin <short-name>`
6. Open a pull request against `segment-any-sign` describing what changed, why, what you ran
   and what you could not verify. Then stop: the maintainer reviews and merges.
Rules:
 
- Never commit or push directly to `segment-any-sign` or `main`, never force-push, and never
  rewrite commits that have been pushed. Do not merge your own pull request.
- Syncing from the upstream repository is a pull request of its own, with nothing else in it.
- `segment-any-sign/latex` is a submodule that points to the paper draft. Leave it alone
  unless the maintainer asks; a pointer update is a commit of its own.
- Start experiments from a committed state, because each run records the commit it ran from.
Commit messages: a short imperative summary line. Add a body when the change needs
explaining: why it was made and how it was checked.
 
Authorship: every commit is authored only by the git identity already configured on the
machine. Add no `Co-Authored-By` trailer, no "Generated with" line, no session link and no
other tool or model attribution, in commit messages, pull request titles or pull request
descriptions. Do not change `git config user.name` or `user.email`, and do not use
`--author`.
 
## Reporting back
 
Say what you changed, what you ran and what the output was, and what you did not or could not
verify. Report tests as passing only if you ran them in this session.
 
Ask the maintainer before: submitting cluster jobs, touching the test split, changing tests,
the evaluation protocol or a metric definition, modifying the upstream package, deleting
files, or anything that cannot be undone.
 