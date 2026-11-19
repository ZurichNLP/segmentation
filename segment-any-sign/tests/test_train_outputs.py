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
