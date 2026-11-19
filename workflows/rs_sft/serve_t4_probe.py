"""Compatibility probe of the original CT checkpoint on isolated free T4s.

This explicit FP16 serving experiment is not an L4/BF16 equivalence claim.
Agent, token limits, original submission and model weight bytes are unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import time

from common import model_identity, write_json
from recording import local_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--isolation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stay-running", action="store_true")
    parser.add_argument("--cpu-offload-gb", type=float, default=2)
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--attention-backend", choices=["TRITON_ATTN", "FLEX_ATTENTION"])
    args = parser.parse_args()
    if Path("/kaggle/input/competitions/gemma-4-developer-agent").exists():
        raise RuntimeError("Official answer-bearing input must never be mounted")
    receipt = json.loads(args.isolation_receipt.read_text())
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if (not receipt.get("passed") or receipt.get("boot_id") != boot
            or receipt.get("controller_approved") is not True
            or receipt.get("validator_positive_control_passed") is not True
            or receipt.get("runtime_identities_distinct") is not True):
        raise RuntimeError("Fresh full isolation receipt for this boot required")
    import torch
    from adk_submission import VllmConfig, VllmServer
    if torch.cuda.device_count() != 2 or any(
            torch.cuda.get_device_capability(i) != (7, 5) for i in range(2)):
        raise RuntimeError("This probe is specifically configured for the observed 2x T4")
    identity = model_identity(args.model_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
    cfg = VllmConfig(
        model=str(args.model_dir.resolve()), host="127.0.0.1", port=8000,
        tool_call_parser="gemma4", reasoning_parser="gemma4",
        max_model_len=32768, tensor_parallel_size=2, dtype="float16",
        gpu_memory_utilization=0.92, enable_auto_tool_choice=True,
        startup_timeout=1200,
        extra_args=["--max-num-seqs", "1", "--max-num-batched-tokens", "1024",
                    "--cpu-offload-gb", str(args.cpu_offload_gb), "--enforce-eager"]
                   + (["--language-model-only"] if args.language_model_only else [])
                   + (["--attention-backend", args.attention_backend]
                      if args.attention_backend else []),
    )
    server = VllmServer(cfg)
    report = {"status": "STARTING", "base": identity,
              "configuration": {k: getattr(cfg, k) for k in [
                  "dtype", "tensor_parallel_size", "max_model_len",
                  "gpu_memory_utilization", "extra_args"]},
              "inference_calls": 0, "equivalent_to_L4_BF16": "NOT_ESTABLISHED"}
    write_json(args.output / "model_smoke.json", report)
    started = time.perf_counter()
    try:
        server.start()
        report["startup_seconds"] = time.perf_counter() - started
        endpoint = "http://127.0.0.1:8000/v1"
        names = [m["id"] for m in local_json(endpoint + "/models")["data"]]
        name = str(args.model_dir.resolve())
        if name not in names:
            raise RuntimeError("Exact local CT model not listed by official server")
        tools = [{"type": "function", "function": {"name": "run_command",
                  "description": "Execute a shell command",
                  "parameters": {"type": "object", "properties": {
                      "command": {"type": "string"}}, "required": ["command"]}}}]
        cases = [
            {"messages": [{"role": "user", "content": "Return only the number 2."}],
             "max_tokens": 32, "temperature": 0},
            {"messages": [{"role": "user", "content":
                "Use run_command to execute exactly: printf P1_TOOL_SMOKE"}],
             "tools": tools, "tool_choice": "auto", "max_tokens": 192, "temperature": 0},
        ]
        outputs = []
        for payload in cases:
            before = time.perf_counter()
            response = local_json(endpoint + "/chat/completions", dict(
                payload, model=name, chat_template_kwargs={"enable_thinking": False}))
            outputs.append({"request": payload, "response": response,
                            "seconds": time.perf_counter() - before})
            report["inference_calls"] += 1
            write_json(args.output / "model_smoke_outputs.json", outputs)
        calls = outputs[1]["response"]["choices"][0]["message"].get("tool_calls", [])
        if not calls or calls[0]["function"]["name"] != "run_command":
            raise RuntimeError("No correctly parsed run_command call in real model response")
        arguments = calls[0]["function"]["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if arguments != {"command": "printf P1_TOOL_SMOKE"}:
            raise RuntimeError("Unexpected tool smoke arguments; no command was executed")
        report.update(status="MODEL_AND_TOOL_PARSER_SMOKE_PASSED", models=names,
                      smoke_outputs="model_smoke_outputs.json")
        write_json(args.output / "model_smoke.json", report)
        if args.stay_running:
            while True:
                time.sleep(1)
    except BaseException as exc:
        if getattr(exc, "output", None):
            (args.output / "server_failure.log").write_text(str(exc.output))
        report.update(status="FAILED", error=repr(exc))
        write_json(args.output / "model_smoke.json", report)
        raise
    finally:
        server.stop()


if __name__ == "__main__":
    main()
