"""Project code names no folder: every location comes from the paths file.

Run from `segment-any-sign/`:
    pytest tests/test_no_absolute_paths.py

Three scans of the project's own files, and the detectors behind them on small
hand-made snippets:

* no absolute path in a string literal of a Python file, or on a command line or
  an `#SBATCH` line of a shell script. Comments and docstrings are not scanned;
* no call into the `paths` module that runs at import (module level, class body,
  default argument, decorator), and `paths.use_file` only in tests;
* the shell scripts parse (`bash -n`).

Not scanned: the notebooks `datasets/*/explore.py`, which keep their locations in
their configuration cell, and this file, which holds the snippets.
"""

import ast
import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path

import pytest

from tests.project_files import NOTEBOOKS, ROOT, project_files

THIS_FILE = "tests/" + Path(__file__).name
PATHS_MODULE = "paths.py"                    # it defines use_file and calls itself by bare names
ALLOWED_IN_SHELL = {"/dev/null"}
SLURM_USER = "%u"                            # the only way an #SBATCH line may name a folder

HOLE = "\ufffc"                              # stands for a {value} of an f-string

# One name of a path: letters and digits, and what SLURM (%u), the shell ($USER, *)
# and an f-string (HOLE) put in a name. The first name after the root starts with a
# letter, a digit or `_`, so that "/", "//" and "/{}" are not paths.
_NAME = rf"[\w.%$*?{HOLE}-]"
_HEAD = rf"(?:~/{_NAME}*|/[A-Za-z0-9_]{_NAME}*)"
_TAIL = rf"(?:/{_NAME}*)*"
_ONE = _HEAD + _TAIL                                             # /name or ~/
_TWO = rf"(?:~/{_NAME}*|/[A-Za-z0-9_]{_NAME}*/{_NAME}+)" + _TAIL    # /name/name or ~/

# A path is absolute where a new word starts: at the start of the text, after one
# of `= : - ( ,` (an option, a shell default, a list), or after white space. After
# white space two names are asked for, so that prose such as "frames /second" is
# not a path. In a shell word a path also follows a redirection, `>` or `<`; in a
# Python literal those two close a tag or a placeholder ("</b>", "<run>/x.json").
ABSOLUTE = re.compile(rf"(?:^|(?<=[=:\-(,]))({_ONE})|(?<=\s)({_TWO})")
ABSOLUTE_IN_SHELL = re.compile(rf"(?:^|(?<=[=:\-(,><]))({_ONE})|(?<=\s)({_TWO})")


# --- the detectors -----------------------------------------------------------

def absolute_paths(text: str, shell: bool = False) -> list[str]:
    """The absolute paths written in one string literal, or in one shell word."""
    pattern = ABSOLUTE_IN_SHELL if shell else ABSOLUTE
    return [match.group(1) or match.group(2) for match in pattern.finditer(text)]


class _Literals(ast.NodeVisitor):
    """Every string literal of a module with its line, docstrings left out."""

    def __init__(self):
        self.found: list[tuple[int, str]] = []

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return                               # a docstring, or a string used as a comment
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            self.found.append((node.lineno, node.value))

    def visit_JoinedStr(self, node):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(value.value)
            else:
                parts.append(HOLE)
                self.visit(value)                # a literal inside the braces
        self.found.append((node.lineno, "".join(parts)))


def python_hits(source: str) -> list[tuple[int, str]]:
    """(line, path) for each absolute path in a string literal of Python source."""
    visitor = _Literals()
    visitor.visit(ast.parse(source))
    return [(line, path) for line, text in visitor.found for path in absolute_paths(text)]


def _without_comment(line: str) -> str:
    quote = None
    for index, char in enumerate(line):
        if quote:
            quote = None if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def shell_hits(source: str) -> list[tuple[int, str]]:
    """(line, path) for each absolute path on a command line or an `#SBATCH` line.

    Quotes are dropped before the line is cut into words, so `"$REPO"/a/b` is one
    word that starts with a variable, and `"/a/b"` is a word that starts with `/`.
    """
    hits = []
    for number, line in enumerate(source.splitlines(), start=1):
        sbatch = line.lstrip().startswith("#SBATCH")
        text = line.lstrip()[len("#SBATCH"):] if sbatch else _without_comment(line)
        for word in text.replace('"', "").replace("'", "").split():
            for path in absolute_paths(word, shell=True):
                if sbatch and SLURM_USER in path:
                    continue
                if not sbatch and path in ALLOWED_IN_SHELL:
                    continue
                hits.append((number, path))
    return hits


class _ImportTime(ast.NodeVisitor):
    """Calls into the paths module that run when the file is imported."""

    def __init__(self, modules: set[str], functions: set[str]):
        self.modules, self.functions = modules, functions
        self.depth = 0                           # number of function bodies around the node
        self.found: list[tuple[int, str]] = []

    def _function(self, node):
        # decorators and defaults are evaluated where the function is defined
        outside = [*getattr(node, "decorator_list", []), *node.args.defaults,
                   *[default for default in node.args.kw_defaults if default is not None]]
        for expression in outside:
            self.visit(expression)
        self.depth += 1
        for statement in (node.body if isinstance(node.body, list) else [node.body]):
            self.visit(statement)
        self.depth -= 1

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _function

    def visit_Call(self, node):
        func = node.func
        into_paths = (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                      and func.value.id in self.modules) \
            or (isinstance(func, ast.Name) and func.id in self.functions)
        if into_paths and self.depth == 0:
            self.found.append((node.lineno, ast.unparse(node)))
        self.generic_visit(node)


def _paths_names(tree: ast.AST) -> tuple[set[str], set[str], list[tuple[int, str]]]:
    """Names bound to the paths module, names imported from it, and each `use_file`."""
    modules, functions, use_file = set(), set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.asname or alias.name for alias in node.names
                        if alias.name == "paths"}
        elif isinstance(node, ast.ImportFrom) and node.module == "paths":
            functions |= {alias.asname or alias.name for alias in node.names}
            use_file += [(node.lineno, "from paths import use_file")
                         for alias in node.names if alias.name == "use_file"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "use_file" \
                and isinstance(node.value, ast.Name) and node.value.id in modules:
            use_file.append((node.lineno, ast.unparse(node)))
    return modules, functions, use_file


def import_time_calls(source: str) -> list[tuple[int, str]]:
    """(line, call) for each call into the paths module outside a function body."""
    tree = ast.parse(source)
    modules, functions, _ = _paths_names(tree)
    visitor = _ImportTime(modules, functions)
    visitor.visit(tree)
    return visitor.found


def use_file_references(source: str) -> list[tuple[int, str]]:
    """(line, text) for each use of the test hook `paths.use_file`."""
    return _paths_names(ast.parse(source))[2]


# --- the files ---------------------------------------------------------------

def scanned(*suffixes: str) -> list[str]:
    """The project's files with one of the suffixes, without this file."""
    return [name for name in project_files(*suffixes) if name != THIS_FILE]


def report(hits: list[tuple[str, int, str]]) -> str:
    return "\n" + "\n".join(f"  {name}:{line}  {text}" for name, line, text in hits)


# --- the detectors on hand-made snippets -------------------------------------

@pytest.mark.parametrize("source, expected", [
    ('folder = Path("/scratch/u/x")', ["/scratch/u/x"]),
    ('folder = "~/data"', ["~/data"]),
    ('command = "sbatch --output=/scratch/x/y job.slurm"', ["/scratch/x/y"]),
    ('message = "the poses are in /shares/group/poses, read-only"', ["/shares/group/poses"]),
    ('folder = f"/scratch/{user}/cache"', ["/scratch/" + HOLE + "/cache"]),
    ('folder = os.path.join("/scratch", user)', ["/scratch"]),
    ('folders = ["/a/b", "/c/d"]', ["/a/b", "/c/d"]),
])
def test_detector_flags_an_absolute_path_in_a_python_literal(source, expected):
    assert [path for _, path in python_hits(source)] == expected


@pytest.mark.parametrize("source", [
    'separator = "/"',
    'folder = f"{root}/mediapipe"',
    'file = f"{root}/{video}.pose"',
    'url = "https://example.org/corpus/poses.zip"',
    'out = "benchmark/predictions/x.json"',
    'unit = "frames /second, or it/s"',
    'label = "B/I/O"',
    'pattern = "/{}"',
    'tag = "<cmdp:Task>Joke</cmdp:Task>"',
    'help = "eval_youtube.py JSON (default: <run>/youtube_dev.json)"',
    '"""A module docstring may say /scratch/u/x."""',
    'def f():\n    """And so may the docstring of a function: ~/data."""',
    'x = 1  # a comment may name /scratch/u/x',
])
def test_detector_passes_what_is_not_an_absolute_path_in_python(source):
    assert python_hits(source) == []


def test_detector_gives_the_line_of_a_python_literal():
    source = 'A = "a/b"\n\nB = Path(\n    "/scratch/u/x")\n'
    assert python_hits(source) == [(4, "/scratch/u/x")]


@pytest.mark.parametrize("line, expected", [
    ("#SBATCH --output=/scratch/x/y/%x-%j.out", ["/scratch/x/y/%x-%j.out"]),
    ("mkdir -p /scratch/x/slurm", ["/scratch/x/slurm"]),
    ("cd /scratch", ["/scratch"]),
    ("PYTHON=/home/x/envs/sas/bin/python   # no conda", ["/home/x/envs/sas/bin/python"]),
    ('REPO="${SLURM_SUBMIT_DIR:-/home/x/repo}"', ["/home/x/repo"]),
    ('[ -f "$REPO/train.py" ] || REPO=/home/x/repo', ["/home/x/repo"]),
    ('echo "logs are in /scratch/$USER/slurm"', ["/scratch/$USER/slurm"]),
    ("cp x.out ~/logs", ["~/logs"]),
    ("python train.py > /scratch/x/log 2>&1", ["/scratch/x/log"]),
    ("python train.py >/scratch/x/log 2>&1", ["/scratch/x/log"]),
])
def test_detector_flags_an_absolute_path_in_a_shell_line(line, expected):
    assert [path for _, path in shell_hits(line)] == expected


@pytest.mark.parametrize("line", [
    "#SBATCH --output=/scratch/%u/segment-any-sign/slurm/%x-%j.out",
    "#SBATCH --error=/scratch/%u/segment-any-sign/slurm/%x-%j.out",
    "#!/bin/bash",
    "# the logs used to be in /scratch/x/slurm",
    'cd "$REPO"   # not /home/x/repo',
    '[ -f "$REPO/experiments/train.py" ] || exit 1',
    '"$REPO"/experiments/train.py --split validation',
    'srun "$PYTHON" experiments/train.py --max_time 01:11:00:00',
    "command -v nvidia-smi > /dev/null || exit 1",
    "python -m paths outputs.slurm 2>/dev/null",
    'NOTE="${4:-weights ${WEIGHTS}, fps aug ${FPS_AUG}}"',
])
def test_detector_passes_what_is_allowed_in_a_shell_line(line):
    assert shell_hits(line) == []


def test_sbatch_line_may_name_a_folder_only_through_the_slurm_user():
    # %u excuses the path that contains it, not the line
    line = "#SBATCH --output=/scratch/%u/slurm/%j.out --chdir=/home/x/repo"
    assert shell_hits(line) == [(1, "/home/x/repo")]
    # and it excuses nothing outside an #SBATCH line
    assert shell_hits("mkdir -p /scratch/%u/slurm") == [(1, "/scratch/%u/slurm")]


@pytest.mark.parametrize("source, expected", [
    ("import paths\nCACHE = paths.output('cache')", ["paths.output('cache')"]),
    ("import paths\nclass A:\n    root = paths.data('a.root')", ["paths.data('a.root')"]),
    ("import paths\ndef f(root=paths.data('a.root')):\n    return root",
     ["paths.data('a.root')"]),
    ("import paths\ndef f(*, root=paths.data('a.root')):\n    return root",
     ["paths.data('a.root')"]),
    ("import paths\nf = lambda root=paths.lookup('a.root'): root", ["paths.lookup('a.root')"]),
    ("import paths as locations\nROOT = Path(locations.data('a.root'))",
     ["locations.data('a.root')"]),
    ("from paths import data\nROOT = data('a.root')", ["data('a.root')"]),
    ("import paths\nif __name__ == '__main__':\n    print(paths.describe())",
     ["paths.describe()"]),
])
def test_detector_flags_a_paths_call_that_runs_at_import(source, expected):
    assert [call for _, call in import_time_calls(source)] == expected


@pytest.mark.parametrize("source", [
    "import paths\ndef f():\n    return paths.output('cache')",
    "import paths\nclass A:\n    def f(self, root=None):\n        return root or paths.data('a.root')",
    "import paths\nf = lambda: paths.data('a.root')",
    "import paths\ndef f():\n    def g(root=paths.data('a.root')):\n        return root\n    return g",
    "from paths import data\ndef f():\n    return data('a.root')",
    # a list that happens to be called `paths`, in a file that does not import the module
    "paths = []\npaths.append('a')",
])
def test_detector_passes_a_paths_call_inside_a_function(source):
    assert import_time_calls(source) == []


def test_detector_finds_the_test_hook():
    assert use_file_references("import paths\ndef f(p):\n    paths.use_file(p)") \
        == [(3, "paths.use_file")]
    assert use_file_references("from paths import use_file") \
        == [(1, "from paths import use_file")]
    assert use_file_references("import paths\ndef f():\n    return paths.output('cache')") == []


# --- the scans of the project ------------------------------------------------

def test_the_scan_reads_the_project_files():
    # a scan that finds no file would pass whatever the code says
    python, shell = scanned(".py"), scanned(".slurm", ".sh")
    assert "experiments/train.py" in python
    assert "datasets/public_dgs_corpus/load.py" in python
    assert "experiments/pretrain_youtube.slurm" in shell
    assert not [name for name in python if fnmatch(name, NOTEBOOKS)]
    assert THIS_FILE not in python


def test_no_absolute_path_in_project_code():
    hits = [(name, line, path) for name in scanned(".py")
            for line, path in python_hits((ROOT / name).read_text())]
    hits += [(name, line, path) for name in scanned(".slurm", ".sh")
             for line, path in shell_hits((ROOT / name).read_text())]
    assert not hits, (
        f"{len(hits)} absolute paths in project code. Every location is a key of the "
        f"paths file, read through paths.py:" + report(hits))


def test_no_paths_call_runs_at_import():
    hits = [(name, line, call) for name in scanned(".py") if name != PATHS_MODULE
            for line, call in import_time_calls((ROOT / name).read_text())]
    assert not hits, (
        "a location is asked for inside the function that uses it, so that a missing "
        "one stops only the code that needs it. Calls that run at import:" + report(hits))


def test_use_file_is_only_used_in_tests():
    hits = [(name, line, text) for name in scanned(".py")
            if name != PATHS_MODULE and not name.startswith("tests/")
            for line, text in use_file_references((ROOT / name).read_text())]
    assert not hits, "paths.use_file is the hook of the tests:" + report(hits)


def test_shell_scripts_parse():
    for name in scanned(".slurm", ".sh"):
        result = subprocess.run(["bash", "-n", name], cwd=ROOT, check=False,
                                capture_output=True, text=True)
        assert result.returncode == 0, f"bash -n {name}:\n{result.stderr}"
