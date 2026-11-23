"""Unit tests for the paths module, which reads every location from one file.

Run from `segment-any-sign/`:
    pytest tests/test_paths.py

Each test writes a small paths file by hand in pytest's tmp_path and makes the
module read it with `paths.use_file`. No test reads `paths.local.yaml`, and the
only tests that read `paths.uzh.yaml` and `paths.example.yaml` are the ones about
those two committed files; they create no folder outside tmp_path.
The tests of section 8 need a folder that the user cannot enter, so they are
skipped for root.
Each test imports what it tests, so one missing module cannot stop the others.
"""

import ast
import getpass
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.project_files import ROOT, notebooks, project_files

UZH = ROOT / "paths.uzh.yaml"
EXAMPLE = ROOT / "paths.example.yaml"

# values that are not a location, and the part of each that the message must show
NOT_A_LOCATION = [
    ("${home}/data", "${home}"),                       # only ${user} and ${output_dir} exist
    ("relative/folder", "relative/folder"),
    (os.path.join("~", "data"), os.path.join("~", "data")),   # `~` is not a variable
    ("5", "5"),                                        # a number
    ("[a, b]", "['a', 'b']"),                          # a list
]


def use(tmp_path: Path, text: str, name: str = "paths.local.yaml") -> Path:
    """Write a paths file by hand and make the module read it."""
    import paths

    file = tmp_path / name
    file.write_text(textwrap.dedent(text))
    paths.use_file(file)
    return file


def message_of(call, *args, **kwargs) -> str:
    """The message of the PathsError that the call must raise."""
    import paths

    with pytest.raises(paths.PathsError) as error:
        call(*args, **kwargs)
    return str(error.value)


# --- 1. which file is read ---------------------------------------------------

@pytest.fixture
def two_files(tmp_path, monkeypatch):
    """The module with its two files in tmp_path; neither is written yet."""
    import paths

    local, default = tmp_path / "paths.local.yaml", tmp_path / "paths.uzh.yaml"
    monkeypatch.setattr(paths, "LOCAL_FILE", local)
    monkeypatch.setattr(paths, "DEFAULT_FILE", default)
    paths.use_file(None)
    return local, default


def test_the_two_files_are_beside_the_module():
    import paths

    assert paths.ROOT == ROOT
    assert paths.LOCAL_FILE == ROOT / "paths.local.yaml"
    assert paths.DEFAULT_FILE == UZH


def test_default_file_is_read_when_there_is_no_local_file(two_files, tmp_path):
    import paths

    _, default = two_files
    default.write_text(f"data:\n  corpus:\n    root: {tmp_path}/shared\n")
    assert paths.file_in_use() == default
    assert paths.lookup("data.corpus.root") == f"{tmp_path}/shared"


def test_local_file_wins_and_the_two_are_never_merged(two_files, tmp_path):
    import paths

    local, default = two_files
    default.write_text(f"data:\n  corpus:\n    root: {tmp_path}/shared\n"
                       f"    extra: {tmp_path}/extra\n")
    local.write_text(f"data:\n  corpus:\n    root: {tmp_path}/mine\n")

    assert paths.file_in_use() == local
    assert paths.lookup("data.corpus.root") == f"{tmp_path}/mine"
    # a key that only the other file has is not set
    assert paths.lookup("data.corpus.extra") is None
    assert list(paths.describe()["locations"]) == ["data.corpus.root"]
    message = message_of(paths.data, "corpus.extra")
    assert "data.corpus.extra" in message
    assert "paths.local.yaml" in message


def test_no_file_at_all_is_an_error_that_names_the_file(two_files):
    import paths

    assert "paths.uzh.yaml" in message_of(paths.lookup, "outputs.cache")


def test_use_file_replaces_the_rule_and_none_restores_it(two_files, tmp_path):
    import paths

    _, default = two_files
    default.write_text(f"output_dir: {tmp_path}/default\n")
    other = use(tmp_path, f"output_dir: {tmp_path}/other\n", name="other.yaml")
    assert paths.file_in_use() == other
    assert paths.lookup("output_dir") == f"{tmp_path}/other"

    paths.use_file(None)
    assert paths.file_in_use() == default
    assert paths.lookup("output_dir") == f"{tmp_path}/default"


@pytest.mark.parametrize("text", [
    "data: [unclosed\n",              # not YAML
    "- a\n- b\n",                     # YAML, but a list
    "outputs: a string\n",            # `outputs` must hold keys
    "data:\n  corpus: a string\n",    # and so must each corpus
    "data:\n  corpus:\n    root:\n      deeper: x\n",
    "output: a_typo\n",               # the keys are output_dir, outputs and data
])
def test_a_file_that_cannot_be_read_is_an_error_that_names_it(tmp_path, text):
    import paths

    file = use(tmp_path, text)
    assert file.name in message_of(paths.describe)


# --- 2. the two variables, and values that are not a location ----------------

def test_user_and_output_dir_are_substituted(tmp_path):
    import paths

    use(tmp_path, f"""\
        output_dir: {tmp_path}/${{user}}/out
        outputs:
          cache: ${{output_dir}}/cache
        data:
          corpus:
            root: {tmp_path}/shares/${{user}}/${{user}}
            beside: ${{output_dir}}/data
        """)
    user = getpass.getuser()
    assert paths.describe()["locations"] == {
        "output_dir": f"{tmp_path}/{user}/out",
        "outputs.cache": f"{tmp_path}/{user}/out/cache",
        "data.corpus.root": f"{tmp_path}/shares/{user}/{user}",
        "data.corpus.beside": f"{tmp_path}/{user}/out/data",
    }
    assert paths.lookup("outputs.cache") == f"{tmp_path}/{user}/out/cache"


# `getpass.getuser()` answers with the first of LOGNAME, USER, LNAME and USERNAME that is set
@pytest.mark.parametrize("renamed", [("LOGNAME",), ("USER",), ("LOGNAME", "USER")],
                         ids=["LOGNAME", "USER", "LOGNAME and USER"])
def test_user_is_the_account_of_the_process_whatever_the_environment_says(tmp_path, monkeypatch,
                                                                          renamed):
    import pwd

    import paths

    account = pwd.getpwuid(os.getuid()).pw_name          # what SLURM's %u means
    for name in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        if name in renamed:
            monkeypatch.setenv(name, "somebody_else")
        else:
            monkeypatch.delenv(name, raising=False)
    use(tmp_path, f"""\
        output_dir: {tmp_path}/${{user}}/out
        outputs:
          cache: ${{output_dir}}/cache
        data:
          corpus:
            root: {tmp_path}/shares/${{user}}
        """)
    assert paths.describe()["locations"] == {
        "output_dir": f"{tmp_path}/{account}/out",
        "outputs.cache": f"{tmp_path}/{account}/out/cache",
        "data.corpus.root": f"{tmp_path}/shares/{account}",
    }
    assert paths.output("cache") == tmp_path / account / "out" / "cache"
    assert not (tmp_path / "somebody_else").exists()


def test_user_is_taken_from_the_environment_only_without_an_entry_for_the_account(tmp_path,
                                                                                  monkeypatch):
    import pwd

    import paths

    def no_entry(uid):
        raise KeyError(f"getpwuid(): uid not found: {uid}")

    # an account that the password database does not know, as with an arbitrary
    # uid in a container, where the environment is all that names the user
    monkeypatch.setattr(pwd, "getpwuid", no_entry)
    monkeypatch.setenv("LOGNAME", "name_in_the_environment")
    use(tmp_path, f"output_dir: {tmp_path}/${{user}}/out\n")
    assert paths.lookup("output_dir") == f"{tmp_path}/name_in_the_environment/out"


def test_a_value_is_used_as_written_no_link_is_resolved(tmp_path):
    import paths

    real, link = tmp_path / "real", tmp_path / "link"
    real.mkdir()
    link.symlink_to(real)
    use(tmp_path, f"data:\n  corpus:\n    root: {link}\n")
    assert str(paths.data("corpus.root")) == str(link)
    assert paths.lookup("data.corpus.root") == str(link)


@pytest.mark.parametrize("value, shown", NOT_A_LOCATION)
def test_a_data_value_that_is_not_a_location_names_key_and_file(tmp_path, value, shown):
    import paths

    file = use(tmp_path, f"data:\n  corpus:\n    root: {value}\n    other:\n")
    calls = [(paths.data, ("corpus.root",), {}),
             (paths.data, ("corpus.root",), {"required": False, "must_exist": False}),
             (paths.lookup, ("data.corpus.root",), {}),
             (paths.describe, (), {})]
    for call, args, kwargs in calls:
        message = message_of(call, *args, **kwargs)
        assert "data.corpus.root" in message
        assert file.name in message
        assert shown in message
    # only the code that uses the key is stopped
    assert paths.lookup("data.corpus.other") is None


@pytest.mark.parametrize("value, shown", NOT_A_LOCATION)
def test_an_output_value_that_is_not_a_location_names_key_and_file(tmp_path, value, shown):
    import paths

    file = use(tmp_path, f"outputs:\n  cache: {value}\n")
    message = message_of(paths.output, "cache")
    assert "outputs.cache" in message
    assert file.name in message
    assert shown in message
    assert sorted(path.name for path in tmp_path.iterdir()) == [file.name, "paths.test.yaml"]


@pytest.mark.parametrize("value, shown", NOT_A_LOCATION)
def test_an_output_dir_that_is_not_a_location_names_key_and_file(tmp_path, value, shown):
    import paths

    file = use(tmp_path, f"output_dir: {value}\noutputs:\n  cache: ${{output_dir}}/cache\n")
    for call, key in ((paths.output, "cache"), (paths.lookup, "outputs.cache"),
                      (paths.lookup, "output_dir")):
        message = message_of(call, key)
        assert "output_dir" in message
        assert file.name in message
        assert shown in message


def test_output_dir_cannot_use_itself(tmp_path):
    import paths

    file = use(tmp_path, "output_dir: ${output_dir}/more\n")
    message = message_of(paths.lookup, "output_dir")
    assert "output_dir" in message
    assert file.name in message


@pytest.mark.parametrize("written", [
    "{tmp}/$USER/x",                # an environment variable: none is read
    "{tmp}/$user/x",                # a variable is written with braces
    "{tmp}/${{user}}/$HOME",        # beside a variable that exists
    "{tmp}/${{user/x",              # braces that are not closed
    "{tmp}/$(id -un)/x",            # a command
    "{tmp}/$$/x",
    "{tmp}/x$",
])
def test_a_dollar_outside_the_two_variables_is_rejected_and_creates_nothing(tmp_path, written):
    import paths

    value = written.format(tmp=tmp_path)
    file = use(tmp_path, f"""\
        output_dir: {value}
        outputs:
          cache: ${{output_dir}}/cache
          dist: {value}
        data:
          corpus:
            root: {value}
            other: {tmp_path}/other
        """)
    # the call, and the key its message names: the one where the `$` is written
    calls = [(paths.output, ("dist",), {}, "outputs.dist"),
             (paths.output, ("cache",), {}, "output_dir"),
             (paths.lookup, ("outputs.dist",), {}, "outputs.dist"),
             (paths.lookup, ("output_dir",), {}, "output_dir"),
             (paths.data, ("corpus.root",), {"must_exist": False}, "data.corpus.root"),
             (paths.data, ("corpus.root",), {"required": False, "must_exist": False},
              "data.corpus.root"),
             (paths.lookup, ("data.corpus.root",), {}, "data.corpus.root"),
             (paths.describe, (), {}, "output_dir")]
    for call, args, kwargs, key in calls:
        message = message_of(call, *args, **kwargs)
        assert key in message
        assert file.name in message
        assert value in message
        assert "${user}" in message and "${output_dir}" in message     # what can be written
    # only the code that uses such a key is stopped, and no folder was created
    assert paths.lookup("data.corpus.other") == f"{tmp_path}/other"
    assert sorted(path.name for path in tmp_path.iterdir()) == [file.name, "paths.test.yaml"]


def user_name_file(tmp_path: Path) -> Path:
    """A paths file where `${user}` is written in `output_dir` and in one data key."""
    return use(tmp_path, f"""\
        output_dir: {tmp_path}/${{user}}/out
        outputs:
          cache: ${{output_dir}}/cache
          plain: {tmp_path}/plain
        data:
          corpus:
            root: {tmp_path}/shares/${{user}}
            other: {tmp_path}/other
        """)


def test_a_user_name_that_cannot_be_found_names_key_and_file(tmp_path, no_user_name):
    import paths

    file = user_name_file(tmp_path)
    # the call, and the key its message names: the one where `${user}` is written
    calls = [(paths.lookup, ("output_dir",), {}, "output_dir"),
             (paths.output, ("cache",), {}, "output_dir"),
             (paths.lookup, ("outputs.cache",), {}, "output_dir"),
             (paths.data, ("corpus.root",), {}, "data.corpus.root"),
             (paths.data, ("corpus.root",), {"required": False, "must_exist": False},
              "data.corpus.root"),
             (paths.lookup, ("data.corpus.root",), {}, "data.corpus.root"),
             (paths.describe, (), {}, "output_dir")]
    for call, args, kwargs, key in calls:
        message = message_of(call, *args, **kwargs)
        assert key in message
        assert file.name in message
        assert "user name" in message
        assert "without ${user}" in message                # what to do about it
    # a key that does not use ${user} is not stopped
    assert paths.lookup("data.corpus.other") == f"{tmp_path}/other"
    assert paths.output("plain") == tmp_path / "plain"
    assert sorted(path.name for path in tmp_path.iterdir()) == [file.name, "paths.test.yaml",
                                                                "plain"]


def test_a_user_name_that_cannot_be_found_in_the_default_file_points_to_a_local_file(
        two_files, tmp_path, no_user_name):
    import paths

    # the default file is committed: the folder is written in a local file instead
    _, default = two_files
    default.write_text(f"output_dir: {tmp_path}/${{user}}/out\n")
    message = message_of(paths.lookup, "output_dir")
    assert "output_dir" in message and "paths.uzh.yaml" in message
    assert "paths.local.yaml" in message and "without ${user}" in message


# what `getpass.getuser()` raises without a user name depends on the Python version.
# The account has no entry in the password database (`no_user_name`), so the name
# is asked from `getpass.getuser()`, and the error is the one it raises.
@pytest.mark.parametrize("error", [KeyError("getpwuid(): uid not found: 1234"),
                                   ImportError("No module named 'pwd'"),
                                   OSError("No username set in the environment")],
                         ids=["KeyError", "ImportError", "OSError"])
def test_a_user_name_error_of_any_python_version_is_a_paths_error(tmp_path, monkeypatch,
                                                                  no_user_name, error):
    import paths

    asked = []

    def no_user():
        asked.append("getpass.getuser")
        raise error

    monkeypatch.setattr(getpass, "getuser", no_user)
    file = user_name_file(tmp_path)
    message = message_of(paths.lookup, "output_dir")
    assert "output_dir" in message and file.name in message
    assert "without ${user}" in message
    assert asked == ["getpass.getuser"]


def test_command_without_a_key_goes_on_when_the_user_name_cannot_be_found(tmp_path, capsys,
                                                                         no_user_name):
    import paths

    file = user_name_file(tmp_path)
    assert paths.main([]) != 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0].split() == ["file", str(file)]
    # the keys that do not use ${user} are listed
    assert [line.split() for line in lines[1:]] == [
        ["missing", "outputs.plain", f"{tmp_path}/plain"],
        ["missing", "data.corpus.other", f"{tmp_path}/other"],
    ]
    # and one line for each of the three others: output_dir, outputs.cache, data.corpus.root
    errors = captured.err.splitlines()
    assert len(errors) == 3
    for key, error in zip(("output_dir", "output_dir", "data.corpus.root"), errors):
        assert key in error and file.name in error and "without ${user}" in error


# --- 3. nothing set, nothing on disk -----------------------------------------

def test_with_every_key_empty_describe_and_lookup_succeed(tmp_path):
    import paths

    file = use(tmp_path, """\
        output_dir:
        outputs:
          cache: ${output_dir}/cache
          dist:
        data:
          corpus:
            root:
            videos:
          other:
        """)
    assert paths.describe() == {
        "file": str(file),
        "locations": {"output_dir": None, "outputs.cache": None, "outputs.dist": None,
                      "data.corpus.root": None, "data.corpus.videos": None},
    }
    for key in ("output_dir", "outputs.cache", "outputs.dist", "data.corpus.root",
                "data.missing.root", "outputs.missing"):
        assert paths.lookup(key) is None
    assert sorted(path.name for path in tmp_path.iterdir()) == [file.name, "paths.test.yaml"]


def test_describe_and_lookup_check_no_folder(tmp_path):
    import paths

    use(tmp_path, f"""\
        output_dir: {tmp_path}/out
        outputs:
          cache: ${{output_dir}}/cache
        data:
          corpus:
            root: {tmp_path}/not/there
        """)
    assert paths.describe()["locations"] == {
        "output_dir": f"{tmp_path}/out",
        "outputs.cache": f"{tmp_path}/out/cache",
        "data.corpus.root": f"{tmp_path}/not/there",
    }
    assert paths.lookup("data.corpus.root") == f"{tmp_path}/not/there"
    assert not (tmp_path / "out").exists()


def test_import_needs_neither_a_file_nor_yaml(tmp_path):
    # a copy of the module alone in a folder: no paths file beside it, and `yaml`
    # made impossible to import. Importing must work; asking for a location says why not.
    shutil.copy(ROOT / "paths.py", tmp_path / "paths.py")
    code = textwrap.dedent("""\
        import sys
        sys.modules["yaml"] = None          # `import yaml` now raises ImportError
        import paths
        print("imported")
        open("paths.uzh.yaml", "w").write("output_dir:\\n")
        try:
            paths.lookup("output_dir")
        except paths.PathsError as error:
            print(error)
        """)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=False,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "imported"
    assert "pyyaml" in lines[1]


# --- 4. data() ---------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path):
    """A paths file with one folder that exists, one that does not, one key not set."""
    (tmp_path / "poses").mkdir()
    return use(tmp_path, f"""\
        data:
          corpus:
            root: {tmp_path}/poses
            gone: {tmp_path}/gone
            empty:
        """)


def test_data_returns_the_folder_that_exists(corpus, tmp_path):
    import paths

    assert paths.data("corpus.root") == tmp_path / "poses"
    assert isinstance(paths.data("corpus.root"), Path)


@pytest.mark.parametrize("key", ["corpus.empty", "corpus.absent", "absent.root"])
def test_data_not_set_names_key_and_file(corpus, key):
    import paths

    message = message_of(paths.data, key)
    assert f"data.{key}" in message
    assert corpus.name in message
    assert "not set" in message


@pytest.mark.parametrize("key", ["corpus.empty", "corpus.absent", "absent.root"])
def test_data_not_set_is_none_when_not_required(corpus, key):
    import paths

    assert paths.data(key, required=False) is None


def test_data_missing_folder_names_key_folder_and_file(corpus, tmp_path):
    import paths

    # also for a key that is not required: it is set, so it must be right
    for kwargs in ({}, {"required": False}):
        message = message_of(paths.data, "corpus.gone", **kwargs)
        assert "data.corpus.gone" in message
        assert str(tmp_path / "gone") in message
        assert corpus.name in message
        assert "does not exist" in message
    assert not (tmp_path / "gone").exists()


def test_data_missing_folder_is_returned_when_it_need_not_exist(corpus, tmp_path):
    import paths

    assert paths.data("corpus.gone", must_exist=False) == tmp_path / "gone"
    assert not (tmp_path / "gone").exists()


# --- 5. output() -------------------------------------------------------------

def test_output_creates_the_folder_and_its_parents(tmp_path):
    import paths

    use(tmp_path, f"output_dir: {tmp_path}/a/b\noutputs:\n  cache: ${{output_dir}}/cache\n")
    assert not (tmp_path / "a").exists()
    assert paths.output("cache") == tmp_path / "a" / "b" / "cache"
    assert (tmp_path / "a" / "b" / "cache").is_dir()
    assert paths.output("cache") == tmp_path / "a" / "b" / "cache"      # and again


@pytest.mark.parametrize("text, key", [
    ("outputs:\n  cache:\n", "outputs.cache"),                   # empty
    ("outputs:\n  dist: {tmp}/dist\n", "outputs.cache"),         # absent
    ("output_dir: {tmp}/out\n", "outputs.cache"),                # no `outputs` at all
    ("output_dir:\noutputs:\n  cache: ${{output_dir}}/cache\n", "output_dir"),
])
def test_output_not_set_names_key_and_file(tmp_path, text, key):
    import paths

    file = use(tmp_path, text.format(tmp=tmp_path))
    message = message_of(paths.output, "cache")
    assert key in message
    assert "outputs.cache" in message
    assert file.name in message
    assert "not set" in message


def test_output_that_cannot_be_created_names_key_folder_and_file(tmp_path):
    import paths

    (tmp_path / "a_file").write_text("in the way\n")
    file = use(tmp_path, f"outputs:\n  cache: {tmp_path}/a_file/cache\n")
    message = message_of(paths.output, "cache")
    assert "outputs.cache" in message
    assert str(tmp_path / "a_file" / "cache") in message
    assert file.name in message


# --- 6. the two committed files ----------------------------------------------

def keys_of(file: Path) -> list[str]:
    import paths

    paths.use_file(file)
    return list(paths.describe()["locations"])


def keys_asked_for(source: str) -> tuple[set[str], list[str]]:
    """Keys that Python code asks the paths module for, and the calls it cannot read.

    `paths.data("a.b")` asks for `data.a.b`, `paths.output("c")` for `outputs.c`,
    and `paths.lookup("k")` for `k`. A key that is not a string literal cannot be
    checked here, so it is reported.
    """
    prefix = {"data": "data.", "output": "outputs.", "lookup": ""}
    keys, unreadable = set(), []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "paths"
                and node.func.attr in prefix):
            continue
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            keys.add(prefix[node.func.attr] + first.value)
        else:
            unreadable.append(f"line {node.lineno}: {ast.unparse(node)}")
    return keys, unreadable


def keys_asked_for_in_shell(source: str) -> set[str]:
    """Keys given to `python -m paths` on a command line of a shell script."""
    keys = set()
    for line in source.replace("\\\n", " ").splitlines():
        _, found, arguments = line.partition("-m paths ")
        if found and not line.lstrip().startswith("#"):
            keys |= set(re.findall(r"(?<!\S)(?:outputs|data)\.[\w.]+", arguments))
    return keys


def test_key_collector_on_hand_made_snippets():
    source = textwrap.dedent("""\
        import paths

        def f(name):
            a = paths.data("corpus.root")
            b = paths.data("corpus.videos", required=False)
            c = paths.output("cache")
            d = paths.lookup("data.other.root")
            e = paths.output(name)
            return paths.describe()
        """)
    assert keys_asked_for(source) == (
        {"data.corpus.root", "data.corpus.videos", "outputs.cache", "data.other.root"},
        ["line 8: paths.output(name)"])
    shell = ('# "$PYTHON" -m paths outputs.old\n'
             '"$PYTHON" -m paths outputs.slurm outputs.dist \\\n'
             '    data.corpus.root \\\n'
             '    || die "a location cannot be used"\n')
    assert keys_asked_for_in_shell(shell) == {"outputs.slurm", "outputs.dist",
                                              "data.corpus.root"}


def test_the_two_committed_files_have_the_same_keys():
    assert keys_of(UZH) == keys_of(EXAMPLE)


def test_example_file_fills_in_the_outputs_and_nothing_else():
    import paths
    import yaml

    example = yaml.safe_load(EXAMPLE.read_text())
    assert list(example) == ["output_dir", "outputs", "data"]
    assert example["output_dir"] is None
    assert example["outputs"] == {name: "${output_dir}/" + name
                                  for name in ("cache", "dist", "wandb", "slurm")}
    assert example["data"]
    for corpus, names in example["data"].items():
        assert names and all(value is None for value in names.values()), corpus

    # so, as it is, the example sets nothing
    paths.use_file(EXAMPLE)
    assert set(paths.describe()["locations"].values()) == {None}


def test_uzh_file_resolves_every_key():
    import paths

    paths.use_file(UZH)
    locations = paths.describe()["locations"]
    for key, value in locations.items():
        assert value is not None, key
        assert os.path.isabs(value) and "$" not in value and "~" not in value, key
    # the outputs belong to the user who runs the code
    output_dir = locations["output_dir"]
    assert getpass.getuser() in Path(output_dir).parts
    for name in ("cache", "dist", "wandb", "slurm"):
        assert locations[f"outputs.{name}"] == f"{output_dir}/{name}"


def test_every_key_the_code_asks_for_is_in_both_committed_files():
    asked, unreadable = set(), []
    for name in project_files(".py"):
        if name.startswith("tests/"):
            continue
        keys, calls = keys_asked_for((ROOT / name).read_text())
        asked |= keys
        unreadable += [f"{name}, {call}" for call in calls]
    for name in project_files(".slurm", ".sh"):
        asked |= keys_asked_for_in_shell((ROOT / name).read_text())

    assert not unreadable, "write the key as a string literal:\n  " + "\n  ".join(unreadable)
    for file in (UZH, EXAMPLE):
        missing = sorted(asked - set(keys_of(file)))
        assert not missing, f"{file.name} lacks {missing}"


def test_every_notebook_asks_paths_for_a_location():
    # no table of notebooks and keys, so a notebook for a new corpus needs no change
    # here. That its keys are in both committed files is the test above, and that it
    # names no folder itself is the scan of tests/test_no_absolute_paths.py
    found = notebooks()
    assert found, "no notebook datasets/*/explore.py was found"
    asking_for_nothing = [name for name in found
                          if not keys_asked_for((ROOT / name).read_text())[0]]
    assert not asking_for_nothing, (
        "a notebook reads its folders from the paths file, with paths.data(...) at "
        f"module level. These ask for no location: {asking_for_nothing}")


def test_git_ignores_the_local_file():
    result = subprocess.run(["git", "check-ignore", "-q", "paths.local.yaml"],
                            cwd=ROOT, check=False)
    assert result.returncode == 0
    tracked = subprocess.run(["git", "ls-files", "paths.local.yaml"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout
    assert tracked == ""


# --- 7. python -m paths ------------------------------------------------------

def test_command_prints_each_folder_it_was_asked_for(corpus, tmp_path, capsys):
    import paths

    corpus.write_text(corpus.read_text() + f"outputs:\n  cache: {tmp_path}/out/cache\n")
    paths.use_file(corpus)
    assert paths.main(["outputs.cache", "data.corpus.root"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{tmp_path}/out/cache",
                                                    f"{tmp_path}/poses"]
    assert (tmp_path / "out" / "cache").is_dir()        # checked as output() checks


@pytest.mark.parametrize("key, reason", [
    ("data.corpus.empty", "not set"),
    ("data.corpus.gone", "does not exist"),
    ("outputs.cache", "not set"),
])
def test_command_prints_the_error_and_fails(corpus, tmp_path, capsys, key, reason):
    import paths

    # the keys around the bad one are still checked and printed
    assert paths.main(["data.corpus.root", key, "data.corpus.root"]) != 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [f"{tmp_path}/poses"] * 2
    assert key in captured.err
    assert reason in captured.err
    assert corpus.name in captured.err


@pytest.mark.parametrize("key", ["output_dir", "cache", "corpus.root"])
def test_command_takes_only_output_and_data_keys(corpus, capsys, key):
    import paths

    assert paths.main([key]) != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert repr(key) in captured.err
    assert "outputs.<name>" in captured.err and "data.<corpus>.<name>" in captured.err


def test_command_without_a_key_shows_the_file_and_every_location(tmp_path, capsys):
    import paths

    (tmp_path / "poses").mkdir()
    file = use(tmp_path, f"""\
        output_dir: {tmp_path}/out
        outputs:
          cache: ${{output_dir}}/cache
        data:
          corpus:
            root: {tmp_path}/poses
            gone: {tmp_path}/gone
            empty:
        """)
    assert paths.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["file", str(file)]
    rows = [re.fullmatch(r"(ok|missing|not set)\s+(\S+)\s*(\S*)", line).groups()
            for line in lines[1:]]
    assert rows == [
        ("missing", "output_dir", f"{tmp_path}/out"),
        ("missing", "outputs.cache", f"{tmp_path}/out/cache"),
        ("ok", "data.corpus.root", f"{tmp_path}/poses"),
        ("missing", "data.corpus.gone", f"{tmp_path}/gone"),
        ("not set", "data.corpus.empty", ""),
    ]
    assert not (tmp_path / "out").exists()              # it shows, it creates nothing


def test_command_without_a_key_fails_on_a_file_it_cannot_read(tmp_path, capsys):
    import paths

    file = use(tmp_path, "data:\n  corpus:\n    root: relative/folder\n")
    assert paths.main([]) != 0
    captured = capsys.readouterr()
    assert "data.corpus.root" in captured.err
    assert file.name in captured.err


def test_python_m_paths_from_the_shell(tmp_path):
    # the real command line, on a copy of the module with a paths file beside it
    shutil.copy(ROOT / "paths.py", tmp_path / "paths.py")
    (tmp_path / "paths.uzh.yaml").write_text(
        f"outputs:\n  cache: {tmp_path}/out/cache\ndata:\n  corpus:\n    root:\n")

    def run(*keys: str):
        return subprocess.run([sys.executable, "-m", "paths", *keys], cwd=tmp_path,
                              check=False, capture_output=True, text=True)

    good = run("outputs.cache")
    assert (good.returncode, good.stdout, good.stderr) == (0, f"{tmp_path}/out/cache\n", "")
    assert (tmp_path / "out" / "cache").is_dir()

    bad = run("outputs.cache", "data.corpus.root")
    assert bad.returncode != 0
    assert bad.stdout == f"{tmp_path}/out/cache\n"
    assert "data.corpus.root" in bad.stderr and "paths.uzh.yaml" in bad.stderr
    assert "Traceback" not in bad.stderr


# --- 8. a folder that cannot be reached --------------------------------------

@pytest.fixture
def closed(tmp_path):
    """A folder that exists, inside a parent of mode 000: only root can enter it."""
    if os.geteuid() == 0:
        pytest.skip("root enters a folder of mode 000, so no folder is out of its reach")
    parent = tmp_path / "closed"
    (parent / "folder").mkdir(parents=True)
    parent.chmod(0o000)
    yield parent / "folder"
    parent.chmod(0o700)                                 # so that pytest can remove it


def test_data_with_a_folder_that_cannot_be_reached_names_key_and_file(closed, tmp_path):
    import paths

    file = use(tmp_path, f"data:\n  corpus:\n    root: {closed}\n    deeper: {closed}/a/b\n")
    for key, folder in (("corpus.root", closed), ("corpus.deeper", closed / "a" / "b")):
        for kwargs in ({}, {"required": False}):
            message = message_of(paths.data, key, **kwargs)
            assert f"data.{key}" in message
            assert str(folder) in message
            assert file.name in message
            assert "cannot be reached" in message
    # asked not to look, it does not look; and neither do lookup() and describe()
    assert paths.data("corpus.root", must_exist=False) == closed
    assert paths.lookup("data.corpus.root") == str(closed)
    assert paths.describe()["locations"]["data.corpus.deeper"] == f"{closed}/a/b"


def test_output_with_a_folder_that_cannot_be_reached_names_key_and_file(closed, tmp_path):
    import paths

    read_only = tmp_path / "read_only"
    read_only.mkdir()
    file = use(tmp_path, f"""\
        outputs:
          cache: {closed}
          dist: {closed.parent}/new/dist
          wandb: {read_only}/wandb
        """)
    for name, folder in (("cache", closed), ("dist", closed.parent / "new" / "dist")):
        message = message_of(paths.output, name)
        assert f"outputs.{name}" in message
        assert str(folder) in message
        assert file.name in message
        assert "cannot be reached" in message

    # a parent that can be entered but not written is another case: nothing is out of reach
    read_only.chmod(0o500)
    try:
        message = message_of(paths.output, "wandb")
    finally:
        read_only.chmod(0o700)
    assert "outputs.wandb" in message and str(read_only / "wandb") in message
    assert file.name in message
    assert "cannot be created" in message and "cannot be reached" not in message


def test_command_prints_the_error_for_a_folder_that_cannot_be_reached(closed, tmp_path, capsys):
    import paths

    (tmp_path / "poses").mkdir()
    file = use(tmp_path, f"""\
        outputs:
          cache: {closed}
        data:
          corpus:
            root: {tmp_path}/poses
            closed: {closed}
        """)
    # the keys around the two bad ones are still checked and printed
    assert paths.main(["data.corpus.root", "data.corpus.closed", "outputs.cache",
                       "data.corpus.root"]) != 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [f"{tmp_path}/poses"] * 2
    errors = captured.err.splitlines()
    assert len(errors) == 2
    for key, error in zip(("data.corpus.closed", "outputs.cache"), errors):
        assert key in error and str(closed) in error and file.name in error
        assert "cannot be reached" in error


def test_command_without_a_key_goes_on_after_a_folder_that_cannot_be_reached(closed, tmp_path,
                                                                            capsys):
    import paths

    (tmp_path / "poses").mkdir()
    file = use(tmp_path, f"""\
        output_dir: {tmp_path}/out
        outputs:
          cache: {closed}
          dist: ${{output_dir}}/dist
        data:
          corpus:
            closed: {closed}/deeper
            root: {tmp_path}/poses
            empty:
        """)
    assert paths.main([]) != 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0].split() == ["file", str(file)]
    # every key whose folder can be looked at is listed, also after one that cannot
    rows = [re.fullmatch(r"(ok|missing|not set)\s+(\S+)\s*(\S*)", line).groups()
            for line in lines[1:]]
    assert rows == [
        ("missing", "output_dir", f"{tmp_path}/out"),
        ("missing", "outputs.dist", f"{tmp_path}/out/dist"),
        ("ok", "data.corpus.root", f"{tmp_path}/poses"),
        ("not set", "data.corpus.empty", ""),
    ]
    # and the two that cannot are named, each with its folder and the file
    errors = captured.err.splitlines()
    assert len(errors) == 2
    for (key, folder), error in zip((("outputs.cache", closed),
                                     ("data.corpus.closed", closed / "deeper")), errors):
        assert key in error and str(folder) in error and file.name in error
        assert "cannot be reached" in error
    assert not (tmp_path / "out").exists()              # it shows, it creates nothing


def test_python_m_paths_shows_no_traceback_for_a_folder_that_cannot_be_reached(closed,
                                                                               tmp_path):
    # the real command line, on a copy of the module with a paths file beside it
    shutil.copy(ROOT / "paths.py", tmp_path / "paths.py")
    (tmp_path / "poses").mkdir()
    (tmp_path / "paths.uzh.yaml").write_text(
        f"data:\n  corpus:\n    closed: {closed}\n    root: {tmp_path}/poses\n")

    def run(*keys: str):
        return subprocess.run([sys.executable, "-m", "paths", *keys], cwd=tmp_path,
                              check=False, capture_output=True, text=True)

    for result, printed in ((run(), ["file", "ok"]),
                            (run("data.corpus.closed", "data.corpus.root"),
                             [f"{tmp_path}/poses"])):
        assert result.returncode != 0
        assert [line.split()[0] for line in result.stdout.splitlines()] == printed
        assert "Traceback" not in result.stderr
        assert len(result.stderr.splitlines()) == 1
        assert "data.corpus.closed" in result.stderr and "paths.uzh.yaml" in result.stderr
        assert "cannot be reached" in result.stderr


# --- 9. the paths file that conftest.py gives every test ---------------------

def with_a_comment_on_every_line(text: str) -> str:
    """The same YAML, with a comment added above it and at the end of each line."""
    return "# added above\n" + "".join(f"{line}  # added\n" if line.strip() else "\n"
                                       for line in text.splitlines())


@pytest.mark.parametrize("text_of_the_example", ["as committed", "with a comment on every line",
                                                 "as a YAML writer writes it"])
def test_the_paths_file_of_the_tests_does_not_depend_on_the_text_of_the_example(
        tmp_path, text_of_the_example):
    import yaml

    committed = EXAMPLE.read_text()
    text = {"as committed": committed,
            "with a comment on every line": with_a_comment_on_every_line(committed),
            "as a YAML writer writes it": yaml.safe_dump(yaml.safe_load(committed),
                                                         sort_keys=False)}[text_of_the_example]
    assert yaml.safe_load(text) == yaml.safe_load(committed)    # the same keys and values
    assert (text == committed) == (text_of_the_example == "as committed")

    # a copy of the project that holds what the fixture needs, and one test that uses
    # it, run by pytest as the tests of the project are
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    shutil.copy(ROOT / "paths.py", project / "paths.py")
    shutil.copy(ROOT / "tests" / "conftest.py", project / "tests" / "conftest.py")
    (project / "tests" / "__init__.py").write_text("")
    (project / "pytest.ini").write_text("[pytest]\n")
    (project / "paths.example.yaml").write_text(text)
    (project / "tests" / "test_fixture.py").write_text(textwrap.dedent("""\
        from pathlib import Path

        import paths


        def test_every_output_is_inside_tmp_path(tmp_path):
            assert Path(paths.__file__).parent == Path(__file__).parents[1]
            assert paths.file_in_use().parent == tmp_path
            locations = paths.describe()["locations"]
            assert locations.pop("output_dir") == str(tmp_path / "outputs")
            outputs = {key: folder for key, folder in locations.items()
                       if key.startswith("outputs.")}
            assert outputs == {f"outputs.{name}": str(tmp_path / "outputs" / name)
                               for name in ("cache", "dist", "wandb", "slurm")}
            data = {key: folder for key, folder in locations.items() if key not in outputs}
            assert data and all(key.startswith("data.") for key in data)
            assert set(data.values()) == {None}
            assert paths.output("cache") == tmp_path / "outputs" / "cache"
        """))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"--basetemp={tmp_path / 'tmp'}", "tests"],
        cwd=project, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
