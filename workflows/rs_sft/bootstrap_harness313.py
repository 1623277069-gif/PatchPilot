"""Offline official Harness environment, separate from the model-serving ABI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--supplement", type=Path, required=True)
    p.add_argument("--shared-runtime", type=Path, required=True)
    p.add_argument("--official-wheels", type=Path, required=True)
    p.add_argument("--task-wheels", type=Path, required=True)
    p.add_argument("--role", choices=["generator", "validator"], required=True)
    p.add_argument("--environment", type=Path, default=Path("/tmp/harness313"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Official task tool environment requires Python 3.13")
    official_tasks = Path("/kaggle/input/competitions/gemma-4-developer-agent/tasks.jsonl")
    if args.role == "generator" and official_tasks.exists():
        raise RuntimeError("Official answers mounted in Generator")
    if args.role == "validator" and not official_tasks.is_file():
        raise RuntimeError("Validator official task input unavailable")
    for base in [args.supplement, args.shared_runtime]:
        manifest = json.loads((base / "manifest.json").read_text())
        for rel, expected in manifest["files"].items():
            path = base / rel
            if not path.resolve().is_relative_to(base.resolve()) or sha(path) != expected:
                raise RuntimeError("Input bundle changed: " + rel)
    args.output.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("No local uv; no online fallback")
    env = dict(os.environ, UV_OFFLINE="1", UV_CACHE_DIR="/tmp/gemma-uv-cache",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    commands = [
        [uv, "venv", "--python", sys.executable, str(args.environment)],
        [uv, "pip", "install", "--python", str(args.environment / "bin/python"),
         "--no-index", "--find-links", str(args.supplement / "wheels"),
         "--find-links", str(args.shared_runtime / "wheels"),
         "--find-links", str(args.official_wheels),
         "--find-links", str(args.task_wheels),
         "-r", str(args.supplement / "harness313-requirements.txt"),
         "swegemma==0.2.11", "adk-submission==0.2.13", "adk-eval-core==0.1.0",
         "pdm-backend==2.4.9"],
        [uv, "pip", "check", "--python", str(args.environment / "bin/python")],
        [str(args.environment / "bin/python"), "-c",
         "import sys,torch,swegemma,adk_submission,adk_eval_core;"
         "from swegemma.models import Task;"
         "from swegemma.config import EvalConfig;"
         "from swegemma.sandbox.subprocess import SubprocessManager;"
         "from swegemma.harness.agent_runner import run_agent_sandbox;"
         "from swegemma.harness.verification import verify_task;"
         "print(sys.version);print(torch.__version__);"
         "print('OFFICIAL_HARNESS_IMPORTS_PASSED')"],
    ]
    receipt = {"status": "INSTALLING", "role": args.role, "started_unix": time.time(),
               "model_calls": 0, "training_steps": 0, "commands": []}
    for i, cmd in enumerate(commands):
        with (args.output / f"harness-bootstrap-{i:02d}.log").open("w") as log:
            run = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=1800)
        receipt["commands"].append({"command": cmd, "returncode": run.returncode})
        receipt["status"] = "FAILED" if run.returncode else "IN_PROGRESS"
        (args.output / "harness_bootstrap.json").write_text(json.dumps(receipt, indent=2))
        if run.returncode:
            return run.returncode
    receipt.update(status="IMPORTS_PASSED_REAL_TASK_CALIBRATION_PENDING",
                   ended_unix=time.time())
    (args.output / "harness_bootstrap.json").write_text(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
