"""Install an audited offline auxiliary runtime without mounting official tasks.

Controller-prepared wheel bytes are checked before uv is invoked. This is not
an isolation receipt and never starts a model, agent, validator, or training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--official-wheels", type=Path, required=True)
    p.add_argument("--environment", type=Path, default=Path("/tmp/gemma312"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if Path("/kaggle/input/competitions/gemma-4-developer-agent").exists():
        raise RuntimeError("Official answer-bearing input mounted; no setup started")
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    for rel, expected in manifest["files"].items():
        path = args.bundle / rel
        if not path.resolve().is_relative_to(args.bundle.resolve()) or sha(path) != expected:
            raise RuntimeError("Offline dependency bundle hash mismatch: " + rel)
    uv = shutil.which("uv")
    if not uv or not Path("/usr/bin/python3.12").is_file():
        raise RuntimeError("Required local uv/Python 3.12 unavailable; no online fallback")
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, UV_CACHE_DIR="/tmp/gemma-uv-cache", UV_OFFLINE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    commands = [
        [uv, "venv", "--python", "/usr/bin/python3.12", str(args.environment)],
        [uv, "pip", "install", "--python", str(args.environment / "bin/python"),
         "--no-index", "--find-links", str(args.bundle / "wheels"),
         "--find-links", str(args.official_wheels),
         "-r", str(args.bundle / "linux312-requirements.txt"),
         "swegemma==0.2.11", "adk-submission==0.2.13", "adk-eval-core==0.1.0"],
        [uv, "pip", "check", "--python", str(args.environment / "bin/python")],
        [str(args.environment / "bin/python"), "-c",
         "import torch,vllm,transformers,swegemma,adk_submission,adk_eval_core;"
         "from swegemma.harness.agent_runner import run_agent_sandbox;"
         "from swegemma.harness.verification import verify_task;"
         "print(torch.__version__,vllm.__version__,transformers.__version__);"
         "print(torch.cuda.device_count(),torch.cuda.get_device_capability(0))"],
    ]
    receipt = {"status": "INSTALLING", "started_unix": time.time(),
               "model_loads": 0, "inference_calls": 0, "training_steps": 0,
               "commands": []}
    for i, command in enumerate(commands):
        with (args.output / f"bootstrap-{i:02d}.log").open("w") as log:
            run = subprocess.run(command, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=1800)
        receipt["commands"].append({"command": command, "returncode": run.returncode})
        receipt["status"] = "FAILED" if run.returncode else "IN_PROGRESS"
        (args.output / "bootstrap.json").write_text(json.dumps(receipt, indent=2))
        if run.returncode:
            return run.returncode
    receipt.update(status="IMPORTS_PASSED_ISOLATION_AND_MODEL_SMOKE_PENDING",
                   ended_unix=time.time())
    (args.output / "bootstrap.json").write_text(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
