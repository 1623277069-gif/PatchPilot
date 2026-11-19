"""Repeat P0's measured boundary checks on the current Generator boot.

Run after dependency setup, before starting a model. The controller must also
compare the Generator/Validator identities and confirm a live Validator service
positive control. This script alone cannot establish those remote facts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validator-host", required=True)
    p.add_argument("--history-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    result = {"passed": False, "boot_id": boot, "hostname": socket.gethostname(),
              "scope": "measured current Generator, known paths and TCP endpoints",
              "requires_controller_identity_comparison": True,
              "requires_live_validator_positive_control": True}
    history = json.loads(args.history_receipt.read_text())
    if history.get("boot_id") != boot or history.get("unknown_rows") != []:
        raise RuntimeError("Notebook history review missing or from another boot")
    result["history"] = history
    prohibited = ["/kaggle/input/competitions/gemma-4-developer-agent/tasks.jsonl",
                  "/tmp/p0-validator-private-20261118/canary",
                  "/tmp/p1-validator-private-20261118/canary",
                  "/kaggle/working/p0-validator/answers_PRIVATE.json"]
    child = "\n".join([
        "import json,sys", "out={}", "for p in sys.argv[1:]:", " try:",
        "  with open(p,'rb') as f:f.read(1)", "  out[p]=True",
        " except OSError:out[p]=False", "print(json.dumps(out))"])
    result["child_reads"] = json.loads(subprocess.check_output(
        [sys.executable, "-c", child, *prohibited], close_fds=True, timeout=15))
    if any(result["child_reads"].values()):
        raise RuntimeError("Generator child can access prohibited data")
    result["fds"] = []
    for proc in sorted(Path("/proc").iterdir()):
        if not proc.name.isdigit():
            continue
        for name in prohibited:
            if (proc / "root" / name.lstrip("/")).is_file():
                raise RuntimeError("Prohibited data visible through process root")
        try:
            fds = list((proc / "fd").iterdir())
        except FileNotFoundError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
                info = fd.stat()
                rec = {"pid": proc.name, "target": target, "regular": stat.S_ISREG(info.st_mode)}
                if rec["regular"]:
                    # Do not mark unread files or denied FDs as passed.
                    handle = os.open(fd, os.O_RDONLY | os.O_NONBLOCK)
                    try:
                        h = hashlib.sha256()
                        count = 0
                        while count < info.st_size:
                            data = os.pread(handle, min(1024 * 1024, info.st_size-count), count)
                            if not data:
                                break
                            h.update(data)
                            count += len(data)
                    finally:
                        os.close(handle)
                    if count != info.st_size:
                        raise RuntimeError("FD read coverage incomplete")
                    rec.update(bytes=count, sha256=h.hexdigest())
                    allowed = (target.endswith("/history.sqlite") or target.startswith(
                        ("/usr/", "/opt/", "/kaggle/input/datasets/metric/",
                         "/kaggle/input/models/", "/tmp/gemma312/")))
                    if not allowed:
                        raise RuntimeError("Unreviewed regular FD: " + target)
                result["fds"].append(rec)
            except FileNotFoundError:
                result["fds"].append({"pid": proc.name, "closed_during_enumeration": True})
    result["network"] = {}
    for host, port in [(args.validator_host, 18761), ("127.0.0.1", 18761),
                       ("::1", 18761), ("1.1.1.1", 443),
                       ("example.com", 443), ("169.254.169.254", 80)]:
        try:
            conn = socket.create_connection((host, port), timeout=3)
        except OSError as exc:
            result["network"][f"{host}:{port}"] = type(exc).__name__
        else:
            conn.close()
            raise RuntimeError("Unexpected reachable endpoint: " + host)
    for root in ["/kaggle/working", "/root/.ipython", "/tmp"]:
        if list(Path(root).rglob("*.ipynb")):
            raise RuntimeError("Unreviewed notebook file accessible to Generator")
    result["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print("MEASURED_GENERATOR_CHECKS_PASSED_CONTROLLER_GATE_PENDING", flush=True)


if __name__ == "__main__":
    main()
