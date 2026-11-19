"""Generation-only P1 runner in a clean Generator, using the frozen Agent.

Official SubprocessManager supplies the tools; physical Kaggle environment
separation supplies the answer boundary. No verifier, answers, full task table,
test patch, or validation IDs are accepted by this program.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from common import MODEL, journal, read_json, write_json
from recording import recording_proxy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_manager(directory):
    from swegemma.sandbox.subprocess import SubprocessManager
    directory.mkdir(parents=True, exist_ok=True)
    manager = SubprocessManager(timeout_seconds=60, base_dir=directory,
                                system_site_packages=True)
    original_start, original_exec, original_copy = manager.start, manager.exec, manager.copy_to
    lock = threading.RLock()
    logfile = directory.parent / "execution.jsonl"

    def snapshot(sid):
        original_exec(sid, "cd /workspace && git add -N .", timeout=30)
        out = original_exec(sid, "cd /workspace && git diff --binary _swegemma_baseline", timeout=30)
        if out.exit_code == 0:
            journal(logfile, {"type": "workspace_diff", "sandbox": sid, "patch": out.stdout})

    def start():
        sid = original_start()
        # Clean source-only archives intentionally have no history. Initialize a
        # new empty repository before the official setup creates its baseline.
        out = original_exec(sid, "cd /workspace && git init -q && "
            "git config user.name 'RS experiment' && "
            "git config user.email 'rs@example.invalid' && "
            "git config core.hooksPath /dev/null", timeout=30)
        if out.exit_code:
            raise RuntimeError("Fresh Git initialization failed: " + out.stderr)
        journal(logfile, {"type": "start", "sandbox": sid,
                          "manager": type(manager).__name__})
        return sid

    def execute(*args, **kwargs):
        with lock:
            out = original_exec(*args, **kwargs)
            journal(logfile, {"type": "exec", "arguments": list(args), "options": kwargs,
                              "exit_code": out.exit_code, "stdout": out.stdout,
                              "stderr": out.stderr})
            if args:
                snapshot(args[0])
            return out

    def copy(*args, **kwargs):
        with lock:
            out = original_copy(*args, **kwargs)
            journal(logfile, {"type": "copy_to", "arguments": [str(x) for x in args]})
            if len(args) == 3:
                snapshot(args[0])
            return out

    # Keep the official manager's actual type. Its task setup has intentional
    # SubprocessManager-specific behavior that a renamed subclass would change.
    manager.start, manager.exec, manager.copy_to = start, execute, copy
    return manager


async def run(args):
    if Path("/kaggle/input/competitions/gemma-4-developer-agent").exists():
        raise RuntimeError("Official answer-bearing input mounted")
    isolation = read_json(args.isolation_receipt)
    if (isolation.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            or isolation.get("controller_approved") is not True):
        raise RuntimeError("Fresh controller-approved isolation receipt required")
    readiness = read_json(args.model_receipt)
    if readiness.get("status") != "MODEL_AND_TOOL_PARSER_SMOKE_PASSED":
        raise RuntimeError("Exact model and real tool parser smoke have not passed")
    task_data = read_json(args.task_dir / "task.json")
    if (set(task_data) != {"instance_id", "repo", "base_commit", "problem_statement"}
            or task_data["instance_id"] != args.task_id):
        raise RuntimeError("P1 accepts only the authorized sanitized training task")
    if sha(args.task_dir / "source.tgz") != args.source_sha256:
        raise RuntimeError("P0-proven source archive changed")
    bundle = read_json(args.bundle_manifest)
    agent_hashes = {n.removeprefix("code/agent/"): h for n, h in bundle["files"].items()
                    if n.startswith("code/agent/")}
    if len(agent_hashes) != 6 or any(sha(args.agent_dir / n) != h for n, h in agent_hashes.items()):
        raise RuntimeError("Frozen six-file Agent changed")
    os.environ.update(KAGGLE_SANDBOX_DIR=str(args.setup_dir), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", OPENAI_API_KEY="EMPTY")
    from adk_submission import ModelRegistry
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.config import EvalConfig
    from swegemma.context import SwegemmaContext
    from swegemma.harness.agent_runner import run_agent_sandbox
    from swegemma.models import Task
    import litellm
    litellm.drop_params = True
    args.output.mkdir(parents=True, exist_ok=True)
    task = Task(**task_data)
    for attempt in range(args.attempt_start, args.attempt_start + args.attempts):
        directory = args.output / f"attempt-{attempt:02d}"
        if directory.exists():
            raise FileExistsError("Do not overwrite an existing attempt: " + str(directory))
        directory.mkdir()
        state = {"status": "GENERATING", "task": task.instance_id, "attempt": attempt,
                 "model": MODEL, "variant": "base", "base": readiness["base"],
                 "split_sha256": bundle["split_freeze_sha256"], "agent_files": agent_hashes,
                 "validator_feedback_used": False, "predeclared_attempts": args.attempts,
                 "budget": {"seconds": 420, "tool_calls": 28, "turns": 80,
                            "per_command_seconds": 60}}
        write_json(directory / "state.json", state)
        registry = ModelRegistry()
        manager = recorded_manager(directory / "sandboxes")
        cfg = EvalConfig(tasks_path=args.task_dir / "task.json", snapshots_dir=args.task_dir,
            results_dir=directory, submission_dir=args.agent_dir, models=registry,
            sandbox="subprocess", wheels_dir=args.wheels_dir, max_tool_calls=28,
            max_turns=80, max_time_minutes=7, timeout_seconds=60,
            verification_timeout_seconds=300, graph_dir="/nonexistent",
            embeddings_dir="/nonexistent", display_mode="quiet")
        class RecordedContext(SwegemmaContext):
            def handle_adk_event(self, event, **kwargs):
                journal(directory / "adk_events.jsonl", {
                    "event": event.model_dump(mode="json", exclude_none=True)})
                return super().handle_adk_event(event, **kwargs)
        context = RecordedContext(docker_manager=manager, task=task,
            problem_statement=task.problem_statement, budget=cfg.budget, harness=cfg.harness,
            graph_dir="/nonexistent", embeddings_dir="/nonexistent")
        before = time.perf_counter()
        with recording_proxy("http://127.0.0.1:8000/v1", readiness["base"]["path"],
                             directory / "model.jsonl") as proxy:
            registry.register(MODEL, LiteLlm(model="openai/" + MODEL, api_base=proxy,
                                            api_key="EMPTY", num_retries=0))
            patch, error, trace = await run_agent_sandbox(manager, cfg, task,
                args.task_dir / "source.tgz", context=context)
        raw = patch.encode("utf-8")
        (directory / "candidate.patch").write_bytes(raw)
        trace.save(directory / "official_trace.json")
        state.update(status="GENERATED_NOT_VALIDATED", error=error,
                     patch_bytes=len(raw), patch_sha256=hashlib.sha256(raw).hexdigest(),
                     wall_seconds=time.perf_counter()-before,
                     tool_calls=context.tool_calls_used, llm_calls=context.llm_calls_used)
        write_json(directory / "state.json", state)
        print(json.dumps({k: state[k] for k in ["task", "attempt", "status", "error",
                         "patch_bytes", "patch_sha256", "wall_seconds"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", required=True)
    p.add_argument("--source-sha256", required=True)
    for name in ["task-dir", "agent-dir", "wheels-dir", "setup-dir", "bundle-manifest",
                 "isolation-receipt", "model-receipt", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--attempt-start", type=int, default=0)
    args = p.parse_args()
    if not 1 <= args.attempts <= 3 or args.attempt_start < 0:
        p.error("P1 budget is at most three fresh attempts")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
