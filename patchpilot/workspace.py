"""Pin a GitHub tree and reconstruct a disposable workspace from a reviewed diff."""

import hashlib
import os
import subprocess
import tempfile
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath

from .config import COMMIT_RE, repo_url

MAX_TREE = 48 * 1024 * 1024
MAX_FILE = 8 * 1024 * 1024
MAX_PATCH = 96 * 1024
MAX_FILES = 6000


class WorkspaceError(ValueError):
    pass


class ImportError(WorkspaceError):
    pass


def safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or str(path) != value
        or len(value) > 512
        or "\\" in value
        or ":" in value
        or any(ord(char) < 32 for char in value)
        or any(part.casefold() in {".", "..", ".git"} for part in path.parts)
    ):
        raise WorkspaceError("Unsupported repository path")
    return value


def patch_sha(commit: str, patch: str) -> str:
    return hashlib.sha256(commit.encode() + b"\0" + patch.encode()).hexdigest()


@dataclass(frozen=True)
class File:
    path: str
    oid: str
    mode: str


def git(*args: str, input: bytes | None = None, timeout=30, extra=None) -> bytes:
    env = {
        "PATH": os.environ["PATH"],
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_ALLOW_PROTOCOL": "https",
        "LC_ALL": "C",
        **(extra or {}),
    }
    try:
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", *args],
            input=input,
            stdin=None if input is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkspaceError("Git operation unavailable or timed out") from exc
    if result.returncode:
        raise WorkspaceError("Git could not read the requested object or apply the patch")
    return result.stdout


def fetch(root: Path, repository_id: str, url: str, commit: str, token: str | None = None):
    if not COMMIT_RE.fullmatch(commit):
        raise WorkspaceError("Expected a full commit SHA")
    url = repo_url(url)
    cache = root / "git" / repository_id / commit
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_dir():
        with tempfile.TemporaryDirectory(dir=cache.parent) as name:
            git("init", "--bare", name)
            auth = {}
            if token:
                auth = {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraHeader",
                    "GIT_CONFIG_VALUE_0": f"Authorization: Bearer {token}",
                }
            try:
                git(
                    f"--git-dir={name}",
                    "-c",
                    "http.followRedirects=false",
                    "fetch",
                    "--depth=1",
                    "--no-tags",
                    "--no-recurse-submodules",
                    url,
                    commit,
                    timeout=90,
                    extra=auth,
                )
                received = git(f"--git-dir={name}", "rev-parse", "FETCH_HEAD").decode().strip()
                if received != commit:
                    raise WorkspaceError("Commit pin mismatch")
                git(f"--git-dir={name}", "update-ref", "refs/heads/pinned", commit)
                Path(name).rename(cache)
            except WorkspaceError as exc:
                raise ImportError("GitHub fetch failed for the exact commit") from exc
    return Tree(cache, commit)


class Tree:
    def __init__(self, cache: Path, commit: str):
        if not COMMIT_RE.fullmatch(commit):
            raise WorkspaceError("Expected a full commit SHA")
        self.cache = cache.resolve()
        self.commit = commit
        self.files = {}
        total = 0
        aliases = {}
        entries = git(f"--git-dir={self.cache}", "ls-tree", "-rlz", commit).split(b"\0")[:-1]
        if len(entries) > MAX_FILES:
            raise WorkspaceError("Repository file count exceeds the local limit")
        for entry in entries:
            meta, path_bytes = entry.split(b"\t", 1)
            mode, kind, oid, size = meta.decode().split()
            if kind != "blob" or mode not in {"100644", "100755"}:
                raise WorkspaceError("Repository contains a symlink or submodule")
            try:
                path = safe_path(path_bytes.decode())
            except UnicodeError as exc:
                raise WorkspaceError("Repository path is not UTF-8") from exc
            for parent in (PurePosixPath(path), *PurePosixPath(path).parents):
                name = str(parent)
                alias = unicodedata.normalize("NFC", name).casefold()
                if aliases.setdefault(alias, name) != name:
                    raise WorkspaceError("Repository has colliding paths")
            total += int(size)
            if total > MAX_TREE or int(size) > MAX_FILE:
                raise WorkspaceError("Repository tree exceeds the local size limit")
            self.files[path] = File(path, oid, mode)

    @contextmanager
    def workspace(self, patch: str = ""):
        if len(patch.encode()) > MAX_PATCH:
            raise WorkspaceError("Patch exceeds size limit")
        with tempfile.TemporaryDirectory(prefix="patchpilot-work-") as temporary:
            base = Path(temporary)
            work = base / "source"
            work.mkdir()
            objects = base / "objects"
            objects.mkdir()
            env = {
                "GIT_INDEX_FILE": str(base / "index"),
                "GIT_OBJECT_DIRECTORY": str(objects),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(self.cache / "objects"),
            }

            def call(*args, **kwargs):
                return git(f"--git-dir={self.cache}", f"--work-tree={work}", *args, extra=env, **kwargs)

            call("read-tree", self.commit)
            if patch:
                call("apply", "--cached", "--whitespace=nowarn", "-", input=patch.encode())
                if diff(call, self.commit) != patch:
                    raise WorkspaceError("Checkpoint patch is not the canonical tree diff")
            materialize(call, self, work)
            yield Workspace(self, work, call)


def diff(call, commit):
    return call(
        "-c",
        "core.quotePath=true",
        "diff",
        "--cached",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--text",
        "--no-renames",
        "--full-index",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        commit,
    ).decode()


def index_files(call, tree):
    entries = call("ls-files", "--stage", "-z").split(b"\0")[:-1]
    actual = {}
    for entry in entries:
        meta, name = entry.split(b"\t", 1)
        mode, oid, stage = meta.decode().split()
        path = name.decode()
        if path not in tree.files or stage != "0" or mode != tree.files[path].mode:
            raise WorkspaceError("Patch changed repository paths or modes")
        actual[path] = oid
    if set(actual) != set(tree.files):
        raise WorkspaceError("Patch changed the repository file set")
    return actual


def materialize(call, tree, work):
    files = index_files(call, tree)
    if not files:
        return
    objects = BytesIO(call("cat-file", "--batch", input=("\n".join(files.values()) + "\n").encode()))
    total = 0
    for path, oid in files.items():
        found, kind, size = objects.readline().decode().split()
        size = int(size)
        total += size
        if found != oid or kind != "blob" or size > MAX_FILE or total > MAX_TREE + MAX_PATCH:
            raise WorkspaceError("Unexpected or oversized workspace object")
        content = objects.read(size)
        if len(content) != size or objects.read(1) != b"\n":
            raise WorkspaceError("Incomplete workspace object")
        file = work / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
        file.chmod(int(tree.files[path].mode[-3:], 8))


class Workspace:
    def __init__(self, tree: Tree, root: Path, call):
        self.tree = tree
        self.root = root
        self.call = call

    def text(self, path: str, limit: int = 256 * 1024) -> str:
        file = self.root / safe_path(path)
        if path not in self.tree.files or file.is_symlink() or file.stat().st_size > limit:
            raise WorkspaceError("File is unavailable or too large")
        try:
            value = file.read_bytes().decode()
        except UnicodeError as exc:
            raise WorkspaceError("File is not UTF-8 text") from exc
        if "\0" in value:
            raise WorkspaceError("Binary file")
        return value

    def edit(self, path: str, old: str, new: str) -> str:
        source = self.text(path)
        stem = PurePosixPath(path)
        if (
            any(
                part.lower() in {"test", "tests", "testing", "fixtures", ".github"}
                for part in stem.parts
            )
            or stem.name.lower().startswith(("test_", "."))
            or stem.name.lower().endswith(("_test.py", ".test.js", ".spec.js"))
            or stem.name.lower()
            in {"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml"}
        ):
            raise WorkspaceError("Tests and verification configuration are read-only")
        if not old or source.count(old) != 1 or old == new or "\0" in new:
            raise WorkspaceError("Provide one unique literal match and a changed replacement")
        updated = source.replace(old, new, 1).encode()
        if len(updated) > 256 * 1024:
            raise WorkspaceError("Updated file exceeds size limit")
        oid = self.call("hash-object", "--no-filters", "-w", "--stdin", input=updated).decode().strip()
        self.call("update-index", "--cacheinfo", f"{self.tree.files[path].mode},{oid},{path}")
        patch = diff(self.call, self.tree.commit)
        if len(patch.encode()) > MAX_PATCH:
            raise WorkspaceError("Patch exceeds size limit")
        return patch
