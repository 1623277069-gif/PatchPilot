import os
import subprocess
import sys
from pathlib import Path

import pytest

from patchpilot.config import Settings
from patchpilot.workspace import Tree

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "synthetic-operator-token-for-tests-only"


def local_commit(directory, files):
    directory.mkdir(exist_ok=True)
    subprocess.run(["git", "init", "-q", directory], check=True)
    for path, content in files.items():
        file = directory / path
        file.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            file.unlink()
        else:
            file.write_text(content)
    subprocess.run(["git", "-C", directory, "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            directory,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.check_output(["git", "-C", directory, "rev-parse", "HEAD"], text=True).strip()


@pytest.fixture
def fixture_tree(tmp_path):
    directory = tmp_path / "repository"
    commit = local_commit(
        directory,
        {
            "src/sample/simple.py": "def add_one(number):\n    return number + 1\n",
            "tests/test_old.py": "def test_old():\n    assert 1\n",
            "notes.md": "Legacy widget conversion.\n",
        },
    )
    return Tree(directory / ".git", commit)


@pytest.fixture(scope="session")
def mysql_url():
    url = os.environ.get("PATCHPILOT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set PATCHPILOT_TEST_DATABASE_URL to a dedicated MySQL test database")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env={**os.environ, "PATCHPILOT_DATABASE_URL": url, "PATCHPILOT_API_TOKEN": TOKEN},
        capture_output=True,
        check=False,
    )
    if result.returncode:
        pytest.fail("MySQL integration database migration failed")
    return url


@pytest.fixture
def mysql_settings(mysql_url, tmp_path):
    return Settings(database_url=mysql_url, api_token=TOKEN, state_dir=tmp_path, _env_file=None)
