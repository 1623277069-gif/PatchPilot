"""Run a fixed test profile in an ephemeral, offline Docker container."""

import os
import selectors
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from uuid import UUID

from .workspace import Workspace

OUTPUT_LIMIT = 8000


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class TestResult:
    code: int
    output: str
    timed_out: bool = False


class DockerSandbox:
    def __init__(self, image: str, timeout: int, runtime: str = ""):
        self.image = image
        self.timeout = timeout
        self.runtime = runtime

    def run(self, job_id: str, workspace: Workspace, profile: str) -> TestResult:
        name = f"patchpilot-test-{UUID(job_id)}"
        if profile not in {"demo", "unittest", "pytest"}:
            raise SandboxError("Unknown test profile")
        flags = [
            "docker",
            "run",
            "--pull=never",
            "--rm",
            "-i",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user=65532:65532",
            "--pids-limit=64",
            "--memory=384m",
            "--memory-swap=384m",
            "--cpus=1",
            "--ulimit=nofile=256:256",
            "--ulimit=core=0:0",
            "--log-driver=none",
            "--tmpfs=/work:rw,noexec,nosuid,nodev,size=192m,mode=1777",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
            "--workdir=/work",
        ]
        if self.runtime:
            flags.extend(["--runtime", self.runtime])
        cleanup = ["docker", "rm", "-f", name]
        # Docker CLI only receives connection and certificate settings from the operator.
        env = {
            key: os.environ[key]
            for key in (
                "PATH",
                "HOME",
                "DOCKER_HOST",
                "DOCKER_CONTEXT",
                "DOCKER_CONFIG",
                "DOCKER_TLS_VERIFY",
                "DOCKER_CERT_PATH",
            )
            if key in os.environ
        }
        try:
            subprocess.run(cleanup, env=env, timeout=10, capture_output=True, check=False)
            with tempfile.TemporaryFile() as archive:
                with tarfile.open(fileobj=archive, mode="w") as tar:
                    for path in workspace.tree.files:
                        tar.add(workspace.root / path, arcname=path, recursive=False)
                archive.seek(0)
                with (
                    subprocess.Popen(
                        [*flags, self.image, profile, str(self.timeout)],
                        env=env,
                        stdin=archive,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    ) as process,
                    selectors.DefaultSelector() as selector,
                ):
                    selector.register(process.stdout, selectors.EVENT_READ)
                    output = bytearray()
                    deadline = time.monotonic() + self.timeout + 10
                    while (remaining := deadline - time.monotonic()) > 0:
                        if not selector.select(remaining):
                            break
                        data = os.read(process.stdout.fileno(), 4096)
                        if not data:
                            break
                        output.extend(data)
                        del output[:-OUTPUT_LIMIT]
                    try:
                        code = process.wait(timeout=max(0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        process.kill()
                        return TestResult(124, "Sandbox exceeded its time limit", True)
            if code in {125, 126, 127}:
                raise SandboxError("Sandbox image or Docker runtime unavailable")
            return TestResult(code, output.decode(errors="replace"), code == 124)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxError("Sandbox unavailable") from exc
        finally:
            try:
                subprocess.run(cleanup, env=env, timeout=10, capture_output=True, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
