"""Trusted entry point. The source archive is the only input from the worker."""

import subprocess
import sys
import tarfile

PROFILES = {
    "demo": ["python", "-I", "/opt/demo_check.py"],
    "unittest": ["python", "-m", "unittest", "discover", "-v"],
    "pytest": ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
}

with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tar:
    tar.extractall("/work", filter="data")

try:
    result = subprocess.run(
        PROFILES[sys.argv[1]],
        timeout=int(sys.argv[2]),
        check=False,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "/work/src:/work",
        },
    )
except subprocess.TimeoutExpired:
    raise SystemExit(124)
raise SystemExit(result.returncode if result.returncode >= 0 else 128 - result.returncode)
