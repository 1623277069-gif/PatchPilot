"""Validator-only official blank/reference calibration and byte-frozen evaluation.

Never upload this role's task table, logs or calibration outputs to Generator.
The reference solution is used only for the Validator's positive control.
"""
import argparse
import asyncio
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


async def run(args):
    from adk_submission import ModelRegistry
    from swegemma.config import EvalConfig
    from swegemma.harness.verification import verify_task
    from swegemma.models import Task
    from swegemma.sandbox.subprocess import SubprocessManager
    official = args.official_dir
    tasks_path = official / "tasks.jsonl"
    if sha(tasks_path) != args.tasks_sha256:
        raise RuntimeError("Official task table changed")
    data = next(row for row in map(json.loads, tasks_path.read_text().splitlines())
                if row["instance_id"] == args.task_id)
    task = Task(**{f.name: data[f.name] for f in fields(Task) if f.name in data})
    archive = args.snapshot
    if sha(archive) != args.snapshot_sha256:
        raise RuntimeError("P0-proven official snapshot changed")
    if args.output.exists():
        raise FileExistsError("Do not overwrite Validator evidence")
    args.output.mkdir(parents=True)
    git_config = args.output.resolve() / "validator.gitconfig"
    for key, value in [("user.name", "RS Validator"),
                       ("user.email", "rs-validator@example.invalid"),
                       ("core.hooksPath", "/dev/null")]:
        subprocess.run(["git", "config", "--file", str(git_config), key, value], check=True)
    os.environ["GIT_CONFIG_GLOBAL"] = str(git_config)
    if args.mode == "calibrate":
        cases = [("blank", ""), ("reference_PRIVATE", task.patch)]
    else:
        calibration = json.loads(args.calibration.read_text())
        transfer = json.loads(args.transfer_receipt.read_text())
        raw = args.candidate.read_bytes()
        if (calibration.get("calibration_passed") is not True
                or transfer.get("generator_stopped_before_transfer") is not True
                or transfer.get("controller_cmp_equal") is not True
                or transfer.get("sha256") != hashlib.sha256(raw).hexdigest()
                or transfer.get("bytes") != len(raw) or not raw.strip()):
            raise RuntimeError("Calibration or byte-level transfer gate missing")
        cases = [("candidate", raw.decode("utf-8"))]
    results = {}
    for label, patch in cases:
        directory = args.output / label
        directory.mkdir()
        cfg = EvalConfig(tasks_path=tasks_path, snapshots_dir=official / "snapshots",
            results_dir=directory, submission_dir=args.agent_dir, models=ModelRegistry(),
            sandbox="subprocess", wheels_dir=official / "wheels", max_time_minutes=7,
            max_tool_calls=28, max_turns=80, timeout_seconds=60,
            verification_timeout_seconds=300, display_mode="quiet")
        manager = SubprocessManager(timeout_seconds=300,
            base_dir=directory / "sandboxes", system_site_packages=True)
        started = time.perf_counter()
        result = await verify_task(manager, cfg, task, archive, agent_patch=patch,
                                   start_time=started)
        obj = result.model_dump(mode="json") if hasattr(result, "model_dump") else vars(result)
        (directory / "result_PRIVATE.json").write_text(json.dumps(obj, indent=2, default=str))
        results[label] = {"resolved": bool(result.resolved),
                          "test_exit_code": result.test_exit_code,
                          "error": getattr(result, "error_message", None),
                          "wall_seconds": time.perf_counter()-started}
        print(label, json.dumps(results[label]), flush=True)
    report = {"mode": args.mode, "task": args.task_id, "results": results,
              "max_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              "model_calls": 0, "feedback_sent_to_generator": False}
    if args.mode == "calibrate":
        blank, gold = results["blank"], results["reference_PRIVATE"]
        report["calibration_passed"] = (blank["resolved"] is False
            and blank["test_exit_code"] == 1 and not blank["error"]
            and gold["resolved"] is True and gold["test_exit_code"] == 0 and not gold["error"])
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    if args.mode == "calibrate" and not report["calibration_passed"]:
        raise RuntimeError("Official full-task calibration did not pass")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["calibrate", "evaluate"])
    p.add_argument("--official-dir", type=Path, required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--tasks-sha256", required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--snapshot-sha256", required=True)
    p.add_argument("--agent-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--transfer-receipt", type=Path)
    args = p.parse_args()
    if args.mode == "evaluate" and not all([args.candidate, args.calibration, args.transfer_receipt]):
        p.error("evaluate requires candidate, calibration and transfer receipt")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
