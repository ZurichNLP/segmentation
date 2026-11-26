"""Where a training run writes, and what it records about it.

Run from `segment-any-sign/`:
    pytest tests/test_train_outputs.py

Folders and links are real ones, in pytest's tmp_path: the paths file of each
test puts every output there (see conftest.py), and the working directory is a
folder of tmp_path that stands for `segment-any-sign/`.
Each test imports what it tests, so one missing function cannot stop the others.

The two tests of `main()` import the dataset modules, which are built on the
upstream package. They are marked with `needs` (see test_locations.py) and
skipped, with the missing packages named, where they are not installed.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

# `main()` imports the dataset modules, which are built on the upstream package
from tests.test_locations import UPSTREAM, needs


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A folder that stands for `segment-any-sign/`, as the working directory."""
    folder = tmp_path / "project"
    folder.mkdir()
    monkeypatch.chdir(folder)
    return folder


# --- link_dist: the run folder -----------------------------------------------

def link_dist(project: Path) -> Path:
    from experiments.train import link_dist

    return link_dist(project)


def test_link_dist_creates_the_link_and_the_folder(project, tmp_path):
    target = tmp_path / "outputs" / "dist"
    assert not target.exists()

    assert link_dist(project) == target
    assert target.is_dir()
    assert (project / "dist").is_symlink()
    assert os.readlink(project / "dist") == str(target)

    # so a run folder made as the training code makes it is in the configured folder
    (Path("dist") / "run").mkdir()
    assert (target / "run").is_dir()


def test_link_dist_leaves_a_right_link_alone(project, tmp_path):
    target = tmp_path / "outputs" / "dist"
    link_dist(project)
    (target / "run").mkdir()

    assert link_dist(project) == target
    assert os.readlink(project / "dist") == str(target)
    assert (target / "run").is_dir()


def test_link_dist_accepts_a_link_that_resolves_to_the_folder(project, tmp_path):
    target = tmp_path / "outputs" / "dist"
    target.mkdir(parents=True)
    (tmp_path / "alias").symlink_to(target)
    (project / "dist").symlink_to(tmp_path / "alias")

    assert link_dist(project) == target
    assert os.readlink(project / "dist") == str(tmp_path / "alias")     # as it was


def test_link_dist_accepts_the_folder_itself(project, set_locations):
    # a user who wants the runs in the checkout sets outputs.dist to `dist` itself
    set_locations({"outputs.dist": project / "dist"})

    assert link_dist(project) == project / "dist"
    assert (project / "dist").is_dir()
    assert not (project / "dist").is_symlink()


@pytest.mark.parametrize("kind", ["a folder", "a link to another folder",
                                  "a link to nothing", "a file"])
def test_link_dist_stops_when_dist_leads_elsewhere(project, tmp_path, paths_file, kind):
    dist, target = project / "dist", tmp_path / "outputs" / "dist"
    other = tmp_path / "other"
    if kind == "a folder":
        (dist / "old_run").mkdir(parents=True)
    elif kind == "a link to another folder":
        (other / "old_run").mkdir(parents=True)
        dist.symlink_to(other)
    elif kind == "a link to nothing":
        dist.symlink_to(other)
    else:
        dist.write_text("")

    with pytest.raises(SystemExit) as stop:
        link_dist(project)
    message = str(stop.value)
    assert "outputs.dist" in message
    assert paths_file.name in message
    assert str(target) in message
    assert str(dist) in message
    if dist.is_symlink():
        assert str(other) in message                 # where it leads now

    # and it is left as it was
    if kind == "a folder":
        assert not dist.is_symlink() and (dist / "old_run").is_dir()
    elif kind == "a file":
        assert dist.is_file() and not dist.is_symlink()
    else:
        assert os.readlink(dist) == str(other)
        assert other.exists() == (kind == "a link to another folder")


def test_link_dist_stops_outside_the_project_folder(project, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    with pytest.raises(SystemExit) as stop:
        link_dist(project)
    assert f"{project.name}/" in str(stop.value)
    assert str(elsewhere) in str(stop.value)
    # nothing was made, in either folder or in the configured one
    assert list(elsewhere.iterdir()) == [] and list(project.iterdir()) == []
    assert not (tmp_path / "outputs").exists()


def test_link_dist_without_argument_means_the_real_project_folder(tmp_path, monkeypatch):
    from experiments.train import link_dist

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stop:
        link_dist()
    assert "segment-any-sign/" in str(stop.value)
    assert not os.path.lexists(tmp_path / "dist")


@pytest.mark.parametrize("other_job_links_to", ["the folder", "another folder"])
def test_link_dist_checks_again_when_another_job_makes_the_link_first(
        project, tmp_path, monkeypatch, other_job_links_to):
    target, other = tmp_path / "outputs" / "dist", tmp_path / "other"
    other.mkdir()
    symlink = os.symlink

    def another_job_was_faster(source, link, *args, **kwargs):
        symlink(target if other_job_links_to == "the folder" else other, link)
        symlink(source, link, *args, **kwargs)        # FileExistsError, as it would be

    monkeypatch.setattr(os, "symlink", another_job_was_faster)
    if other_job_links_to == "the folder":
        assert link_dist(project) == target
    else:
        with pytest.raises(SystemExit) as stop:
            link_dist(project)
        assert "outputs.dist" in str(stop.value) and str(other) in str(stop.value)


# --- set_wandb_dir: the files of Weights & Biases ----------------------------

def message_of(call, *args, **kwargs) -> str:
    """The message of the PathsError that the call must raise."""
    import paths

    with pytest.raises(paths.PathsError) as error:
        call(*args, **kwargs)
    return str(error.value)


def test_set_wandb_dir_uses_the_configured_folder_and_creates_it(tmp_path):
    from experiments.train import set_wandb_dir

    # "." is upstream's default: W&B would write into the working directory
    args = argparse.Namespace(no_wandb=False, wandb_dir=".")
    set_wandb_dir(args, ["--run_name", "00_run", "--device", "cpu"])

    assert args.wandb_dir == str(tmp_path / "outputs" / "wandb")
    # created here: W&B falls back to a temporary folder when it cannot write
    assert (tmp_path / "outputs" / "wandb").is_dir()


def test_set_wandb_dir_without_wandb_changes_nothing_and_makes_no_folder(tmp_path, set_locations):
    from experiments.train import set_wandb_dir

    args = argparse.Namespace(no_wandb=True, wandb_dir=".")
    set_wandb_dir(args, ["--no_wandb"])
    assert args.wandb_dir == "."
    assert not (tmp_path / "outputs").exists()

    # and it needs no key
    set_locations({"outputs.wandb": None})
    set_wandb_dir(args, ["--no_wandb"])
    assert args.wandb_dir == "."


@pytest.mark.parametrize("option", ["--wandb_dir {folder}", "--wandb_dir={folder}",
                                    "--wandb_d {folder}"])       # argparse takes a prefix
def test_set_wandb_dir_keeps_a_folder_given_on_the_command_line(tmp_path, option):
    from experiments.train import set_wandb_dir

    mine = tmp_path / "mine"
    args = argparse.Namespace(no_wandb=False, wandb_dir=str(mine))
    set_wandb_dir(args, ["--run_name", "00_run", *option.format(folder=mine).split()])

    assert args.wandb_dir == str(mine)
    assert not (tmp_path / "outputs").exists()


def test_set_wandb_dir_is_not_misled_by_other_wandb_options(tmp_path):
    from experiments.train import set_wandb_dir

    args = argparse.Namespace(no_wandb=False, wandb_dir=".")
    set_wandb_dir(args, ["--wandb_project", "segment-any-sign", "--wandb_entity=team"])
    assert args.wandb_dir == str(tmp_path / "outputs" / "wandb")


def test_set_wandb_dir_without_the_key_names_it(set_locations):
    from experiments.train import set_wandb_dir

    set_locations({"outputs.wandb": None})
    args = argparse.Namespace(no_wandb=False, wandb_dir=".")
    assert "outputs.wandb" in message_of(set_wandb_dir, args, [])
    assert args.wandb_dir == "."


# --- write_run_config: the record of the locations ---------------------------

def test_run_config_records_the_paths_file_and_every_location(tmp_path, paths_file,
                                                               set_locations):
    import paths
    from experiments.train import write_run_config

    set_locations({"data.public_dgs_corpus.downloads": tmp_path / "downloads",
                   "data.youtube_sl25.root": tmp_path / "not_there"})
    args = argparse.Namespace(learning_rate=1e-3)
    mine = argparse.Namespace(phrase="glosses", velocity="off",
                              select_on="mean_mf1s", limit=None)
    write_run_config(tmp_path / "run", args, mine, "validation_mean_mf1s")

    recorded = json.loads((tmp_path / "run" / "run_config.json").read_text())["paths"]
    assert recorded["file"] == str(paths_file)
    locations = recorded["locations"]
    assert locations["output_dir"] == f"{tmp_path}/outputs"
    assert locations["outputs.dist"] == f"{tmp_path}/outputs/dist"
    assert locations["data.public_dgs_corpus.downloads"] == f"{tmp_path}/downloads"
    # a key that is not set is recorded as such, not left out
    assert locations["data.youtube_sl25.videos"] is None
    # it is a record of what the file says: a folder that is not there is recorded too
    assert locations["data.youtube_sl25.root"] == f"{tmp_path}/not_there"
    # every key of the file, and nothing was created to write the record
    assert recorded == paths.describe()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["paths.test.yaml", "run"]


# --- write_run_config: the command and the options of the project -------------

def values_for_every_option(parser) -> list[str]:
    """A command line that gives every option of `parser` a value that is not its default.

    Made from the parser itself, so that an option added to it is in the line
    without a change here. A text has a space, quotes and a `$` in it.
    """
    line = []
    for action in parser._actions:
        flag = action.option_strings[0]
        if action.nargs == 0:                           # a flag such as --dry-run
            line.append(flag)
        elif action.choices:
            line += [flag, next(choice for choice in action.choices
                                if choice != action.default)]
        elif action.type is int:
            line += [flag, str((action.default or 0) + 3)]
        elif action.type is float:
            line += [flag, str((action.default or 0.0) + 0.5)]
        else:
            line += [flag, "two words, 'quoted' and $HOME"]
    return line


def write_record(tmp_path, monkeypatch, typed: list[str], mine) -> dict:
    """The run_config.json that `write_run_config` writes for a run started with `typed`."""
    from experiments.train import write_run_config

    monkeypatch.setattr(sys, "orig_argv", typed)
    write_run_config(tmp_path / "run", argparse.Namespace(), mine, "validation_mean_mf1s")
    return json.loads((tmp_path / "run" / "run_config.json").read_text())


def test_run_config_records_every_option_of_the_project_as_given(tmp_path, monkeypatch):
    from experiments.train import project_parser

    parser = project_parser()
    typed = ["python", "experiments/train.py", "--run_name", "00_run",
             *values_for_every_option(parser)]
    mine, rest = parser.parse_known_args(typed[2:])
    assert rest == ["--run_name", "00_run"]           # upstream's, not recorded here
    defaults = vars(parser.parse_known_args([])[0])

    options = write_record(tmp_path, monkeypatch, typed, mine)["options"]

    # every option of the parser, under its name in the namespace
    assert set(options) == {action.dest for action in parser._actions}
    for name, value in vars(mine).items():
        assert options[name] == value
        assert value != defaults[name]                # so it was read from the line


def test_command_as_typed_gives_the_same_options_when_parsed_again(tmp_path, monkeypatch):
    import shlex

    from experiments.train import project_parser

    parser = project_parser()
    typed = ["python", "experiments/train.py", *values_for_every_option(parser)]
    mine = parser.parse_known_args(typed[2:])[0]

    record = write_record(tmp_path, monkeypatch, typed, mine)

    again = parser.parse_known_args(shlex.split(record["command_as_typed"])[2:])[0]
    assert vars(again) == vars(mine) == record["options"]


@pytest.mark.parametrize("note", ["two words", "it's", 'say "this"', "$HOME and `ls`",
                                  "", "tab\there", "line\nbreak"])
def test_command_as_typed_can_be_pasted_back_into_a_shell(note, tmp_path, monkeypatch):
    import subprocess

    from experiments.train import project_parser

    # the interpreter, by its full name as under SLURM, and its options are part
    # of what was typed
    typed = [str(tmp_path / "sas" / "bin" / "python"), "-u", "experiments/train.py",
             "--note", note, "--limit", "2", "--run_name", "00_run"]
    mine = project_parser().parse_known_args(typed[3:])[0]
    command = write_record(tmp_path, monkeypatch, typed, mine)["command_as_typed"]

    # the words bash makes of the command, without running it
    words = subprocess.run(["bash", "-c", f"set -- {command}; printf '%s\\0' \"$@\""],
                           capture_output=True, text=True, check=True).stdout
    assert words.split("\0")[:-1] == typed


# --- slurm_log_warning: the log folder that SLURM used -----------------------

def test_slurm_log_warning_is_none_for_a_log_in_the_configured_folder(tmp_path):
    from experiments.train import slurm_log_warning

    log = tmp_path / "outputs" / "slurm" / "02_pretrain_youtube-8485766.out"
    assert slurm_log_warning(str(log)) is None
    assert slurm_log_warning(log) is None
    assert not (tmp_path / "outputs").exists()          # it looks, it creates nothing


def test_slurm_log_warning_follows_links(tmp_path, set_locations):
    from experiments.train import slurm_log_warning

    # the configured folder and the folder SLURM names are the same folder
    real = tmp_path / "disk" / "slurm"
    real.mkdir(parents=True)
    (tmp_path / "scratch").symlink_to(tmp_path / "disk")
    set_locations({"outputs.slurm": tmp_path / "scratch" / "slurm"})
    assert slurm_log_warning(str(real / "job-1.out")) is None


def test_slurm_log_warning_names_the_log_the_key_and_the_file(tmp_path, paths_file):
    from experiments.train import slurm_log_warning

    log = tmp_path / "elsewhere" / "slurm" / "02_pretrain_youtube-8485766.out"
    warning = slurm_log_warning(str(log))
    assert str(log) in warning
    assert "outputs.slurm" in warning
    assert str(tmp_path / "outputs" / "slurm") in warning
    assert paths_file.name in warning
    assert "\n" not in warning                          # one line of the log

    # a folder whose name only starts like the configured one is another folder
    assert slurm_log_warning(str(tmp_path / "outputs" / "slurm_old" / "job-1.out")) is not None


def test_slurm_log_warning_is_none_when_the_key_is_not_set(tmp_path, set_locations):
    from experiments.train import slurm_log_warning

    set_locations({"outputs.slurm": None})
    assert slurm_log_warning(str(tmp_path / "anywhere" / "job-1.out")) is None


def test_slurm_log_warning_never_raises(tmp_path):
    import paths
    from experiments.train import slurm_log_warning

    # a paths file that cannot be used must not stop a run at this point
    broken = tmp_path / "broken.yaml"
    broken.write_text("outputs:\n  slurm: relative/folder\n")
    paths.use_file(broken)
    assert slurm_log_warning(str(tmp_path / "anywhere" / "job-1.out")) is None

    paths.use_file(tmp_path / "no_such_file.yaml")
    assert slurm_log_warning(str(tmp_path / "anywhere" / "job-1.out")) is None


def test_slurm_log_warning_never_raises_when_the_user_name_cannot_be_found(tmp_path,
                                                                          no_user_name):
    import paths
    from experiments.train import slurm_log_warning

    file = tmp_path / "with_user.yaml"
    file.write_text(f"outputs:\n  slurm: {tmp_path}/${{user}}/slurm\n")
    paths.use_file(file)
    assert slurm_log_warning(str(tmp_path / "anywhere" / "job-1.out")) is None
    # the paths module says why, to the code that asks for the folder
    with pytest.raises(paths.PathsError):
        paths.lookup("outputs.slurm")


def test_link_slurm_log_prints_the_warning(tmp_path, monkeypatch, capsys):
    import subprocess

    from experiments.train import link_slurm_log

    def scontrol(log: Path):
        """What `scontrol show job` answers for a batch job, cut to the lines used."""
        return lambda *args, **kwargs: (f"JobId=1 JobName=02_pretrain_youtube\n"
                                        f"   StdErr={log}\n   StdIn=(null)\n   StdOut={log}\n")

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    inside = tmp_path / "outputs" / "slurm" / "job-1.out"
    outside = tmp_path / "elsewhere" / "job-1.out"

    monkeypatch.setattr(subprocess, "check_output", scontrol(inside))
    link_slurm_log(tmp_path / "run_a")
    assert os.readlink(tmp_path / "run_a" / "slurm.out") == str(inside)
    assert "outputs.slurm" not in capsys.readouterr().out

    monkeypatch.setattr(subprocess, "check_output", scontrol(outside))
    link_slurm_log(tmp_path / "run_b")
    assert os.readlink(tmp_path / "run_b" / "slurm.out") == str(outside)     # linked all the same
    printed = capsys.readouterr().out
    assert "outputs.slurm" in printed and str(outside) in printed


# --- main: the paths file is checked before any work -------------------------

def run_main(monkeypatch, *arguments: str) -> None:
    """`python experiments/train.py ARGUMENTS`, in this process."""
    from experiments import train

    monkeypatch.setattr(sys, "argv", ["train.py", *arguments])
    # upstream's args.py parses the command line when it is imported, so it is
    # imported anew: the arguments of another test must not be the ones in use
    monkeypatch.delitem(sys.modules, "sign_language_segmentation.args", raising=False)
    train.main()


@needs(*UPSTREAM)
def test_main_stops_before_any_work_on_a_value_that_is_not_a_location(
        project, tmp_path, monkeypatch, paths_file, set_locations):
    import paths

    # in a key that a training run never asks for: run_config.json records every
    # key, and it is written only after the class weights and the data report
    set_locations({"data.ncslgr.root": "relative/folder"})

    with pytest.raises(paths.PathsError) as error:
        run_main(monkeypatch, "--dry-run", "--no_wandb", "--device", "cpu",
                 "--limit", "2", "--datasets", "dgs_corpus")
    message = str(error.value)
    assert "data.ncslgr.root" in message
    assert paths_file.name in message
    assert "relative/folder" in message
    # it stopped before `link_dist`, the first thing that writes: no link, no folder
    assert list(project.iterdir()) == []
    assert not (tmp_path / "outputs").exists()


@needs(*UPSTREAM)
def test_main_with_help_needs_no_location(project, tmp_path, monkeypatch, capsys):
    import paths

    # no paths file at all: asking for any location would be a PathsError
    paths.use_file(tmp_path / "no_such_file.yaml")

    with pytest.raises(SystemExit) as stop:
        run_main(monkeypatch, "--help")
    assert stop.value.code == 0
    assert "usage:" in capsys.readouterr().out
    assert list(project.iterdir()) == []
    assert not (tmp_path / "outputs").exists()


# --- main: the record of a run -----------------------------------------------

def start_training(monkeypatch, *arguments, training=None) -> dict:
    """`main()` with `arguments`, up to where upstream's training starts; the record it wrote.

    `training`, given the monitored metric, stands for upstream's `train()`; by
    default nothing is trained. The rest is `start_main` of test_decoder_use.py:
    no data is read and the names `main()` sets are put back after the test.
    No SLURM job is linked, also when the tests run inside one.
    """
    from tests.test_decoder_use import start_main

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    # upstream's training module parses the command line when it is first imported
    monkeypatch.setattr(sys, "argv", ["train.py"])
    import sign_language_segmentation.train as upstream_train

    monkeypatch.setattr(upstream_train, "train", training or (lambda monitor_metric: None))
    # main() gives the model class the record of the run; no other test may meet it
    from experiments.validation_metrics import ValidationMetricsModel

    monkeypatch.setattr(ValidationMetricsModel, "run_config", None, raising=False)
    # --eval-split none: no end step, whose scripts would read the corpus
    start_main(monkeypatch, "--eval-split", "none", *arguments)
    return written_record()


def written_record() -> dict:
    """The run_config.json of the one run folder under `dist` of the working directory."""
    [written] = Path("dist").glob("*/run_config.json")
    return json.loads(written.read_text())


@needs(*UPSTREAM)
def test_main_records_the_options_as_given_and_the_command_as_typed(project, monkeypatch):
    import shlex

    typed = ["python", "experiments/train.py", "--sampling", "video", "--note", "two words"]
    monkeypatch.setattr(sys, "orig_argv", typed)
    record = start_training(monkeypatch, "--sampling", "video", "--class-weights", "1,5,1",
                            "--note", "two words")

    assert record["command_as_typed"] == shlex.join(typed)
    options = record["options"]
    assert (options["sampling"], options["class_weights"], options["note"],
            options["limit"]) == ("video", "1,5,1", "two words", 2)
    # as given, although main() resolves them before the record is written
    assert options["val_every"] is None and options["max_steps"] is None
    assert options["decode"] is None and options["levels"] == "auto"

    # the keys of before keep what they held
    assert set(record) >= {"run_id", "started_at", "command", "monitor_metric", "ours",
                           "args", "code", "paths", "environment"}
    assert record["ours"] == {"phrase": "glosses", "velocity": "off",
                              "select_on": "mean_mf1s", "limit": 2}
    # "command" is the line upstream parsed: without the options of the project
    assert record["command"].endswith("train.py --no_wandb --device cpu --datasets dgs_corpus")
    assert record["args"]["decode"] == "b-honouring" and record["args"]["levels"] == "both"


def available_cpus() -> int:
    """The CPUs this process may use, counted as `main()` counts them."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else (os.cpu_count() or 8)


@needs(*UPSTREAM)
@pytest.mark.parametrize("options, expected", [
    # 4 clips at batch 64 are one step per epoch: a validation every step
    ([], {"max_steps": 5000, "val_every": 1, "num_workers": min(32, available_cpus()),
          "class_weights": {"sign": [0.0, 1.0, 9.0, 1.0],      # start_main stands in for measuring
                            "sentence": [0.0, 1.0, 9.0, 1.0]}}),
    (["--max-steps", "40", "--val-every", "5", "--num-workers", "3",
      "--class-weights", "1,5,1"],
     {"max_steps": 40, "val_every": 5, "num_workers": 3,
      "class_weights": {"sign": [0.0, 1.0, 5.0, 1.0], "sentence": [0.0, 1.0, 5.0, 1.0]}}),
    (["--class-weights", "none"],
     {"max_steps": 5000, "val_every": 1, "num_workers": min(32, available_cpus()),
      "class_weights": None}),
])
def test_main_records_the_values_it_resolves_that_no_other_key_holds(
        options, expected, project, monkeypatch):
    """The step budget, the validation interval, the workers and the class weights."""
    record = start_training(monkeypatch, *options)
    assert {key: record["args"][key] for key in expected} == expected


@needs(*UPSTREAM)
def test_main_sends_the_options_and_the_command_as_typed_to_wandb(project, monkeypatch):
    """What upstream's `train()` gives W&B as the config of the run, with no connection.

    Upstream's own `train()` runs with the WandbLogger of Lightning, and W&B in
    the mode `disabled`, whose run keeps the config it is given and sends
    nothing. Upstream logs its namespace `args` as that config; the run is
    stopped right after, where it would start to read the data.
    """
    from pytorch_lightning.loggers import WandbLogger

    from tests.test_decoder_use import Reached

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(sys, "argv", ["train.py"])
    import sign_language_segmentation.train as upstream_train

    real_train, loggers = upstream_train.train, []

    def stop(*arguments, **options):
        raise Reached

    def keep(**options):
        loggers.append(WandbLogger(**options))
        return loggers[-1]

    def training(monitor_metric):
        # the namespace that main() filled
        from sign_language_segmentation.args import args

        # Upstream's module keeps the namespace of its first import. In a run the
        # two are one; in a test the first import was another command line.
        monkeypatch.setattr(upstream_train, "args", args)
        # start_main passes --no_wandb, so that main() asks for no W&B folder
        monkeypatch.setattr(args, "no_wandb", False)
        monkeypatch.setattr(upstream_train, "WandbLogger", keep)
        monkeypatch.setattr(upstream_train, "get_dataloader", stop)
        real_train(monitor_metric=monitor_metric)

    typed = ["python", "experiments/train.py", "--sampling", "video"]
    monkeypatch.setattr(sys, "orig_argv", typed)
    with pytest.raises(Reached):
        start_training(monkeypatch, "--sampling", "video", training=training)
    record = written_record()

    config = dict(loggers[0].experiment.config)
    assert config["options"] == record["options"]
    assert config["options"]["sampling"] == "video"
    assert config["command_as_typed"] == record["command_as_typed"]
    # and run_config.json holds them once, outside "args"
    assert "options" not in record["args"] and "command_as_typed" not in record["args"]


# --- the record in every checkpoint ------------------------------------------

#: a record as `write_run_config` returns it, cut to the keys these tests read
RECORD = {"run_id": "00_run-2026.11.26", "options": {"sampling": "video", "decode": None},
          "args": {"decode": "b-honouring", "levels": "both"},
          "paths": {"file": "paths.test.yaml"}, "environment": {"host": "node"}}


def checkpoint_with(record, tmp_path, monkeypatch, name="best.ckpt") -> Path:
    """A checkpoint of a small model of ours, written while the record of the run is `record`."""
    from experiments.validation_metrics import ValidationMetricsModel

    from tests.test_decoder_use import SMALL, checkpoint_of

    monkeypatch.setattr(ValidationMetricsModel, "run_config", record, raising=False)
    return checkpoint_of(ValidationMetricsModel(**SMALL), tmp_path / name)


def stored(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


@needs(*UPSTREAM)
def test_a_checkpoint_keeps_the_record_of_its_run(tmp_path, monkeypatch):
    checkpoint = stored(checkpoint_with(RECORD, tmp_path, monkeypatch))
    assert checkpoint["run_config"] == RECORD
    # a key of its own: the hyperparameters are what they were
    assert "run_config" not in checkpoint["hyper_parameters"]
    assert checkpoint["hyper_parameters"]["decode"] == RECORD["args"]["decode"]


@needs(*UPSTREAM)
def test_a_model_built_without_a_record_writes_none(tmp_path, monkeypatch):
    # a dry run, a test, or upstream's class: no key, rather than an empty one
    assert "run_config" not in stored(checkpoint_with(None, tmp_path, monkeypatch))


@needs(*UPSTREAM)
def test_a_record_that_names_another_decoder_is_refused_when_the_model_is_built(monkeypatch):
    from experiments.validation_metrics import ValidationMetricsModel
    from metrics import get_decoder

    from tests.test_decoder_use import SMALL

    monkeypatch.setattr(ValidationMetricsModel, "decoder", get_decoder("b-honouring"))
    monkeypatch.setattr(ValidationMetricsModel, "run_config",
                        {**RECORD, "args": {"decode": "b-ignoring"}}, raising=False)
    with pytest.raises(ValueError) as refused:
        ValidationMetricsModel(**SMALL)
    assert "b-ignoring" in str(refused.value) and "b-honouring" in str(refused.value)


@needs(*UPSTREAM)
def test_the_loaders_read_a_checkpoint_that_keeps_a_record(tmp_path, monkeypatch):
    import torch
    from benchmark.predict_dgs_2026 import load_checkpoint

    from sign_language_segmentation.model.model import PoseTaggingModel

    path = checkpoint_with(RECORD, tmp_path, monkeypatch)
    weights = stored(path)
    assert "run_config" in weights
    for loaded in (PoseTaggingModel.load_from_checkpoint(path, map_location="cpu"),
                   load_checkpoint(str(path))):
        for key, value in weights["state_dict"].items():
            assert torch.equal(loaded.state_dict()[key], value), key
    # the decoder is still read from the hyperparameters
    assert load_checkpoint(str(path)).stored_decode == "b-honouring"


@needs(*UPSTREAM)
def test_the_export_leaves_the_record_out_of_config_json(tmp_path, monkeypatch):
    """Upstream's export, which `publish.py` uploads: the folders and host stay behind."""
    from sign_language_segmentation.publish.utils import convert_to_safetensors

    path = checkpoint_with(RECORD, tmp_path, monkeypatch)
    assert "run_config" in stored(path)
    (tmp_path / "export").mkdir()
    convert_to_safetensors(str(path), tmp_path / "export")

    config = json.loads((tmp_path / "export" / "config.json").read_text())
    assert set(config) == set(stored(path)["hyper_parameters"])
    assert "run_config" not in config and RECORD["environment"]["host"] not in json.dumps(config)


@needs(*UPSTREAM)
def test_main_gives_the_checkpoints_of_the_run_its_record(project, tmp_path, monkeypatch):
    from experiments.validation_metrics import ValidationMetricsModel

    from tests.test_decoder_use import SMALL, checkpoint_of

    written = []

    def training(monitor_metric):
        # as upstream builds the model: after main() has set the class up
        written.append(checkpoint_of(ValidationMetricsModel(**SMALL), tmp_path / "best.ckpt"))

    record = start_training(monkeypatch, "--sampling", "video", "--decode", "i-runs-kept",
                            training=training)
    checkpoint = stored(written[0])
    assert checkpoint["run_config"] == record
    assert checkpoint["hyper_parameters"]["decode"] == record["args"]["decode"] == "i-runs-kept"


# --- the readers of a run folder: the best checkpoint --------------------------

#: the run folder of these tests, under `outputs.dist` of the paths file
RUN = "00_run-2026.11.26"


@pytest.fixture
def run_folder(tmp_path, monkeypatch):
    """A run folder with a best.ckpt and a later last.ckpt that differ in everything.

    best.ckpt is step 6 and keeps the decoder `b-ignoring`; last.ckpt, written
    after it, is step 12 with other weights and keeps `i-runs-kept`, as when
    the last validation is not the best one. Returns the folder and the steps
    and decoders by file name.
    """
    import paths
    import torch
    from experiments.validation_metrics import ValidationMetricsModel
    from metrics import get_decoder

    from tests.test_decoder_use import SMALL, checkpoint_of

    folder = paths.output("dist") / RUN
    folder.mkdir(parents=True)
    kept = {"best.ckpt": (6, "b-ignoring"), "last.ckpt": (12, "i-runs-kept")}
    for when, (name, (step, decoder)) in enumerate(kept.items()):
        monkeypatch.setattr(ValidationMetricsModel, "decoder", get_decoder(decoder))
        path = checkpoint_of(ValidationMetricsModel(**SMALL), folder / name)
        state = torch.load(path, map_location="cpu", weights_only=False)
        state["global_step"] = step
        torch.save(state, path)
        os.utime(path, (1_000_000 + when, 1_000_000 + when))     # last.ckpt is newer
    return folder, kept


def weights_of(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


@needs(*UPSTREAM)
def test_the_fixture_has_two_different_checkpoints(run_folder):
    import torch

    folder, _ = run_folder
    best, last = weights_of(folder / "best.ckpt"), weights_of(folder / "last.ckpt")
    assert not all(torch.equal(best[key], last[key]) for key in best)
    assert (folder / "last.ckpt").stat().st_mtime > (folder / "best.ckpt").stat().st_mtime


@needs(*UPSTREAM)
def test_newest_checkpoint_takes_best_unless_asked_for_last(run_folder):
    """Used by eval_youtube.py, plot_segmentation.py and add_result_row.py."""
    from experiments.plot_segmentation import newest_checkpoint

    folder, _ = run_folder
    assert newest_checkpoint(folder) == folder / "best.ckpt"
    assert newest_checkpoint(folder, "best") == folder / "best.ckpt"
    assert newest_checkpoint(folder, "last") == folder / "last.ckpt"


@needs(*UPSTREAM)
def test_the_end_of_training_predicts_with_best(run_folder, monkeypatch):
    import subprocess

    from experiments.train import evaluate_best_checkpoint

    folder, _ = run_folder
    steps = []

    def ran(step, **options):
        steps.append(step)
        return subprocess.CompletedProcess(step, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", ran)
    evaluate_best_checkpoint(folder, phrase="glosses", split="validation",
                             plot_datasets="dgs_corpus")
    [predict] = [step for step in steps if step[1].endswith("predict_dgs_2026.py")]
    assert predict[predict.index("--model") + 1] == str(folder / "best.ckpt")
    [plot] = [step for step in steps if step[1].endswith("plot_segmentation.py")]
    assert "--which" not in plot                       # so its default, best


@needs(*UPSTREAM)
def test_the_loaders_given_a_run_folder_load_best(run_folder):
    import torch
    from benchmark.predict_dgs_2026 import load_checkpoint, stored_decode

    from sign_language_segmentation.bin import _load_model_uncached

    folder, kept = run_folder
    best = weights_of(folder / "best.ckpt")
    for model in (load_checkpoint(str(folder)), _load_model_uncached(model_dir=str(folder))):
        assert all(torch.equal(model.state_dict()[key], value) for key, value in best.items())
    assert stored_decode(folder) == kept["best.ckpt"][1]
    assert load_checkpoint(str(folder)).stored_decode == kept["best.ckpt"][1]


@needs(*UPSTREAM)
@pytest.mark.parametrize("script", ["eval_youtube", "plot_segmentation"])
def test_the_scripts_given_a_run_take_best_by_default(script, run_folder, monkeypatch):
    import importlib

    from tests.test_decoder_use import Reached

    module = importlib.import_module(f"experiments.{script}")
    function = {"eval_youtube": "score", "plot_segmentation": "collect"}[script]

    def opened(checkpoint, *arguments, **options):
        raise Reached(checkpoint)

    monkeypatch.setattr(module, function, opened)
    monkeypatch.setattr(sys, "argv", [f"{script}.py", "--run", RUN, "--device", "cpu"]
                        + (["--datasets", "dgs_corpus"] if script == "plot_segmentation"
                           else []))
    with pytest.raises(Reached) as reached:
        module.main()
    assert reached.value.args[0] == run_folder[0] / "best.ckpt"


@needs(*UPSTREAM)
def test_add_result_row_writes_the_step_of_best(run_folder, tmp_path, monkeypatch):
    from experiments import add_result_row

    scores = {"frame_f1": 0.5, "iou": 0.5, "percentage": 1.0, "mf1s": 0.5,
              "decode": "b-honouring"}
    youtube = tmp_path / "youtube_dev.json"
    youtube.write_text(json.dumps({"raw": scores, "filtered": scores}))
    readme = tmp_path / "README.md"
    readme.write_text(f"{add_result_row.TABLE_ANCHOR}\n|---|\n| 1 | `earlier` |\n")
    monkeypatch.setattr(sys, "argv", ["add_result_row.py", "--run", RUN, "--note", "x",
                                      "--youtube", str(youtube),
                                      "--dgs", str(tmp_path / "no_dgs.json"),
                                      "--readme", str(readme)])
    add_result_row.main()

    row = readme.read_text().splitlines()[-1]
    assert row.startswith(f"| 2 | `{RUN}` | x | {run_folder[1]['best.ckpt'][0]} |")
