import os
from uuid import uuid4

import pytest
from conftest import local_commit

from patchpilot.sandbox import DockerSandbox
from patchpilot.workspace import Tree

pytestmark = pytest.mark.sandbox


def test_untrusted_process_gets_only_offline_disposable_source(tmp_path, monkeypatch):
    if os.environ.get("PATCHPILOT_TEST_DOCKER") != "1":
        pytest.skip("Set PATCHPILOT_TEST_DOCKER=1 after building the sandbox image")
    monkeypatch.setenv("PATCHPILOT_DATABASE_URL", "synthetic-canary")
    root = tmp_path / "repo"
    sha = local_commit(
        root,
        {
            "test_boundary.py": """
import os
import socket
import unittest
from pathlib import Path

class Boundary(unittest.TestCase):
    def test_scope(self):
        self.assertNotIn('PATCHPILOT_DATABASE_URL', os.environ)
        self.assertNotIn('DOCKER_HOST', os.environ)
        self.assertEqual(os.getuid(), 65532)
        self.assertFalse(Path('/var/run/docker.sock').exists())
        self.assertFalse(Path('/work/.git').exists())
        self.assertEqual(set(socket.if_nameindex()), {(1, 'lo')})
        self.assertEqual(Path('/sys/fs/cgroup/pids.max').read_text().strip(), '64')
        self.assertEqual(int(Path('/sys/fs/cgroup/memory.max').read_text()), 384 * 1024 * 1024)
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        self.assertEqual(int(quota), int(period))
        with self.assertRaises(OSError):
            Path('/opt/sandbox_write_probe').write_text('probe')
        Path('/work/ephemeral').write_text('transient')
"""
        },
    )
    with Tree(root / ".git", sha).workspace() as workspace:
        result = DockerSandbox("patchpilot-sandbox:local", 20).run(str(uuid4()), workspace, "unittest")
        assert result.code == 0, result.output
        assert not (workspace.root / "ephemeral").exists()


def test_deadline_kills_sandbox_test(tmp_path):
    if os.environ.get("PATCHPILOT_TEST_DOCKER") != "1":
        pytest.skip("Set PATCHPILOT_TEST_DOCKER=1 after building the sandbox image")
    root = tmp_path / "repo"
    sha = local_commit(
        root,
        {
            "test_slow.py": """
import time
import unittest

class TestSlow(unittest.TestCase):
    def test_slow(self):
        time.sleep(20)
"""
        },
    )
    with Tree(root / ".git", sha).workspace() as workspace:
        result = DockerSandbox("patchpilot-sandbox:local", 2).run(str(uuid4()), workspace, "unittest")
        assert (result.code, result.timed_out) == (124, True)
