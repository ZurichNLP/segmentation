"""Every location outside the repository, read from one file.

The file is `paths.local.yaml` when it exists, otherwise `paths.uzh.yaml`, both
beside this module. Exactly one of them is read: they are never merged.
`paths.example.yaml` is the template, and says what each folder must contain.

Code asks for a location inside the function that uses it:

    import paths

    poses = paths.data("youtube_sl25.root")                    # must be set, must exist
    videos = paths.data("youtube_sl25.videos", required=False)  # None when not set
    cache = paths.output("cache")                              # created when missing

Nothing is read when this module is imported, and a location is checked only
when it is asked for, so a missing one stops only the code that needs it. The
file is chosen and parsed on the first call and kept in memory.

A value is an absolute path. Two variables exist: `${user}`, the name of the
account the process runs as, and `${output_dir}`, the value of the key
`output_dir`. `${user}` is the entry of the process in the password database,
which is what SLURM writes for `%u`: LOGNAME and USER do not change it, and
are read only for an account that has no such entry. Any other `$` is an
error: a value cannot name an environment variable. A value is used as
written after substitution: no link is resolved and `~` is not expanded.

From the command line, inside `segment-any-sign/`:

    python -m paths                  # the file in use, and each location
    python -m paths outputs.cache data.youtube_sl25.root

Without a key it shows every location as `ok`, `missing` or `not set`, and
creates nothing; an output shown as `missing` is created when it is first used.
A key whose value cannot be used, or whose folder cannot be reached, gets its
error instead of a line: the other keys are still shown, and the exit status
is non-zero. With keys it checks each one as `output()` or `data()` does and
prints the folder, or prints the error and exits with a non-zero status.
"""

from __future__ import annotations

import argparse
import errno
import getpass
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # segment-any-sign/
LOCAL_FILE = ROOT / "paths.local.yaml"
DEFAULT_FILE = ROOT / "paths.uzh.yaml"
EXAMPLE_NAME = "paths.example.yaml"

_VARIABLE = re.compile(r"\$\{([^}]*)\}")
# the errors of `os.stat` that mean "not there", the ones `Path.exists()` reads
# so on Python 3.11; any other error means that the folder cannot be reached
_NOT_THERE = (errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP)

_override: Path | None = None                     # set by use_file()
_read: tuple[Path, dict[str, object]] | None = None   # the file read, and its keys as written


class PathsError(RuntimeError):
    """A location cannot be used. The message names the key and the file."""


class _NotSet(PathsError):
    """The key has no value in the file."""


def file_in_use() -> Path:
    """`paths.local.yaml` if it exists, otherwise `paths.uzh.yaml`. Never both."""
    if _read is not None:
        return _read[0]
    if _override is not None:
        return _override
    return LOCAL_FILE if LOCAL_FILE.exists() else DEFAULT_FILE


def use_file(path: Path | None) -> None:
    """For tests only: read this file instead. None restores the rule of `file_in_use`."""
    global _override, _read
    _override = None if path is None else Path(path)
    _read = None


def data(key: str, *, required: bool = True, must_exist: bool = True) -> Path | None:
    """The folder of the key `data.<key>`, where `key` is `<corpus>.<name>`.

    Not set: PathsError, or None when `required` is false. Set, but the folder is
    not on this machine or cannot be reached: PathsError, unless `must_exist` is
    false, which looks at nothing.
    """
    file, written = _load()
    try:
        folder = Path(_resolve(f"data.{key}", written, file))
    except _NotSet:
        if required:
            raise
        return None
    if must_exist and not _is_there(f"data.{key}", folder, file):
        raise PathsError(f"data.{key} is {folder} in {file.name}, but that folder does not "
                         f"exist on this machine. {_hint(file)}")
    return folder


def output(name: str) -> Path:
    """The folder of the key `outputs.<name>`, created with its parents when missing.

    PathsError when it cannot be reached or cannot be created.
    """
    file, written = _load()
    folder = Path(_resolve(f"outputs.{name}", written, file))
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        _is_there(f"outputs.{name}", folder, file)      # PathsError when it cannot be reached
        raise PathsError(f"outputs.{name} is {folder} in {file.name}, but that folder "
                         f"cannot be created: {error}") from error
    return folder


def lookup(key: str) -> str | None:
    """The location of `output_dir`, `outputs.<name>` or `data.<corpus>.<name>`, or None.

    Nothing is checked on disk and nothing is created: for records and messages.
    """
    file, written = _load()
    try:
        return _resolve(key, written, file)
    except _NotSet:
        return None


def describe() -> dict:
    """The file in use and the location of every key in it; None for a key not set.

    Nothing is checked on disk. This is what a training run records.
    """
    file, written = _load()
    return {"file": str(file), "locations": {key: lookup(key) for key in written}}


# --- reading the file --------------------------------------------------------

def _load() -> tuple[Path, dict[str, object]]:
    """The file in use and its keys as written: `output_dir`, `outputs.x`, `data.x.y`."""
    global _read
    if _read is not None:
        return _read
    file = file_in_use()
    try:
        import yaml
    except ImportError as error:
        raise PathsError(f"reading {file.name} needs the package pyyaml, which {sys.executable} "
                         f"does not have: pip install pyyaml") from error
    try:
        content = yaml.safe_load(file.read_text())
    except OSError as error:
        raise PathsError(f"the paths file {file} cannot be read: {error}") from error
    except yaml.YAMLError as error:
        raise PathsError(f"the paths file {file} is not valid YAML: {error}") from error

    content = {} if content is None else content
    if not isinstance(content, dict):
        raise PathsError(f"the paths file {file} must hold the keys output_dir, outputs and "
                         f"data, as {EXAMPLE_NAME} does")
    unknown = sorted(str(key) for key in content if key not in ("output_dir", "outputs", "data"))
    if unknown:
        raise PathsError(f"the paths file {file} has the unknown key {', '.join(unknown)}: "
                         f"the keys are output_dir, outputs and data")

    written: dict[str, object] = {}
    if "output_dir" in content:
        written["output_dir"] = content["output_dir"]
    for name, value in _keys(content, "outputs", file).items():
        written[f"outputs.{name}"] = value
    for corpus in _keys(content, "data", file):
        for name, value in _keys(content["data"], corpus, file, inside="data.").items():
            written[f"data.{corpus}.{name}"] = value
    _read = (file, written)
    return _read


def _keys(content: dict, name: str, file: Path, inside: str = "") -> dict:
    """The keys under `name`; an empty value holds none, anything but keys is an error."""
    value = content.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PathsError(f"{inside}{name} is {value!r} in the paths file {file}, but it must "
                         f"hold keys, as in {EXAMPLE_NAME}")
    return value


def _resolve(key: str, written: dict[str, object], file: Path) -> str:
    """The location of one key, with the variables substituted. `_NotSet` when it has none."""
    value = written.get(key)
    if value is None:
        raise _NotSet(f"{key} is not set in {file.name}. {_hint(file)}")
    if not isinstance(value, str):
        raise PathsError(f"{key} is {value!r} in {file.name}, which is not a path.")
    if "$" in _VARIABLE.sub("", value):
        raise PathsError(f"{key} is {value} in {file.name}, but a `$` in it is not part of "
                         f"${{user}} or ${{output_dir}}, the two variables. A value cannot "
                         f"name an environment variable: for the user name, write ${{user}}.")

    def variable(match: re.Match) -> str:
        name = match.group(1)
        if name == "user":
            return _user(key, value, file)
        if name == "output_dir" and key != "output_dir":
            try:
                return _resolve("output_dir", written, file)
            except _NotSet:
                raise _NotSet(f"{key} is {value} in {file.name}, but output_dir is not set "
                              f"there. {_hint(file)}") from None
        if name == "output_dir":
            raise PathsError(f"output_dir is {value} in {file.name}: it cannot use "
                             f"${{output_dir}}, which is itself.")
        raise PathsError(f"{key} is {value} in {file.name}, but ${{{name}}} is not a "
                         f"variable. The two variables are ${{user}} and ${{output_dir}}.")

    location = _VARIABLE.sub(variable, value)
    if not os.path.isabs(location):
        raise PathsError(f"{key} is {value} in {file.name}, which is not an absolute path. "
                         f"`~` is not expanded: write the folder in full, or with ${{user}}.")
    return str(Path(location))


def _user(key: str, value: str, file: Path) -> str:
    """The name of the account the process runs as, for `${user}` in the value of `key`.

    It is the entry of the user id of the process in the password database,
    which is what SLURM writes for `%u`, so LOGNAME and USER do not change it.
    Only an account without such an entry gets the name `getpass.getuser()`
    gives: the first of LOGNAME, USER, LNAME and USERNAME that is set.
    """
    try:
        # imported here: the password database is not a module of every system
        import pwd

        return pwd.getpwuid(os.getuid()).pw_name
    except (ImportError, KeyError, OSError):
        pass                                          # no entry: ask as before
    try:
        return getpass.getuser()
    except (KeyError, ImportError, OSError) as error:
        # nothing in the environment names the user either, as with an arbitrary
        # uid in a container. Which of the three is raised depends on the Python
        # version.
        where = (f"Copy {file.name} to {LOCAL_FILE.name} and write the folder there"
                 if file == DEFAULT_FILE else "Write the folder")
        raise PathsError(f"{key} is {value} in {file.name}, but the user name of this "
                         f"account cannot be found. {where} without ${{user}}.") from error


def _is_there(key: str, folder: Path, file: Path) -> bool:
    """Whether the folder of `key` exists. PathsError when it cannot be reached.

    A folder cannot be reached when a folder above it cannot be entered, for
    example the scratch folder of another user. `Path.exists()` is not used:
    on Python 3.11 it raises PermissionError there, on 3.14 it answers False.
    """
    try:
        os.stat(folder)
    except OSError as error:
        if error.errno in _NOT_THERE:
            return False
        raise PathsError(f"{key} is {folder} in {file.name}, but that folder cannot be "
                         f"reached: {error}") from error
    except ValueError:                                # a path that the system cannot hold
        return False
    return True


def _hint(file: Path) -> str:
    """What to do about a key that is not set or a folder that is not there."""
    if file == DEFAULT_FILE:
        return (f"{file.name} holds the UZH locations: on another machine, copy "
                f"{EXAMPLE_NAME} to {LOCAL_FILE.name} and set it there.")
    return (f"{file.name} is the only file read, it is never merged with {DEFAULT_FILE.name}: "
            f"set the key there. {EXAMPLE_NAME} says what each folder must contain.")


# --- python -m paths ---------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m paths",
        description="Show the paths file in use and its locations, or check the given keys.")
    parser.add_argument("keys", nargs="*", metavar="KEY",
                        help="outputs.<name>, created when missing, or data.<corpus>.<name>, "
                             "which must exist. Without a key: every location, nothing created")
    keys = parser.parse_args(argv).keys
    return _check(keys) if keys else _show()


def _show() -> int:
    """Every location of the file with `ok`, `missing` or `not set`. Creates nothing.

    A key that cannot be shown so gets its error, and the listing goes on.
    """
    try:
        file, written = _load()
    except PathsError as error:
        print(f"paths: {error}", file=sys.stderr)
        return 1
    width = max((len(key) for key in written), default=0)
    failed = False
    print(f"{'file':<7}  {file}", flush=True)
    for key in written:
        try:
            location = lookup(key)
            state = ("not set" if location is None
                     else "ok" if _is_there(key, Path(location), file) else "missing")
        except PathsError as error:
            print(f"paths: {error}", file=sys.stderr)
            failed = True
            continue
        print(f"{state:<7}  {key:<{width}}  {location or ''}".rstrip(), flush=True)
    return 1 if failed else 0


def _check(keys: list[str]) -> int:
    """Each key as `output()` or `data()` checks it: the folder, or the error."""
    failed = False
    for key in keys:
        section, _, rest = key.partition(".")
        try:
            if section == "outputs" and rest:
                folder = output(rest)
            elif section == "data" and rest:
                folder = data(rest)
            else:
                raise PathsError(f"{key!r} is not a key this command checks: give "
                                 f"outputs.<name> or data.<corpus>.<name>.")
        except PathsError as error:
            print(f"paths: {error}", file=sys.stderr)
            failed = True
        else:
            print(folder, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
