"""Run in the separate Kaggle notebook; no LLM or training requests.

Uses the official subprocess backend when installed. Prints only booleans and
metadata for answer-read tests, never reference patch/test content.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

OUT = Path('/kaggle/working/gemma-rs-runtime-probe')
OUT.mkdir(parents=True, exist_ok=True)


def command(argv, timeout=12):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {'code': result.returncode, 'stdout': result.stdout[-6000:], 'stderr': result.stderr[-2000:]}
    except Exception as exc:
        return {'error': str(exc)}


report = {'python': sys.version, 'platform': platform.platform(), 'uid': os.getuid(),
          'model_inference_requests': 0, 'training_steps': 0, 'new_cost_cny': 0}
report['gpu'] = command(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'])
report['ram'] = command(['free', '-b'])
report['disk'] = command(['df', '-B1', '/kaggle/working', '/kaggle/temp'])
report['binaries'] = {x: shutil.which(x) for x in ['docker', 'bwrap', 'unshare', 'chroot', 'setpriv']}
report['namespace_probe'] = command(['unshare', '--user', '--map-root-user', '--mount', 'true']) if shutil.which('unshare') else None
proc = Path('/proc/self/status')
report['process_security'] = [line for line in proc.read_text().splitlines()
                             if line.startswith(('Uid:', 'Gid:', 'CapEff:', 'NoNewPrivs:', 'Seccomp:'))] if proc.exists() else []
report['packages'] = {}
for name in ['torch', 'transformers', 'peft', 'bitsandbytes', 'compressed-tensors', 'swegemma', 'adk-submission']:
    try:
        report['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        report['packages'][name] = None
tasks = list(Path('/kaggle/input').glob('**/tasks.jsonl'))
# Import the exact official pure-Python wheels without installing or changing
# the notebook's packages. Never add all wheel files (native dependencies may
# target a different Python/CUDA runtime).
for wheel in Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse').glob('*.whl'):
    if wheel.name.startswith(('swegemma-', 'adk_submission-', 'adk_eval_core-')):
        sys.path.insert(0, str(wheel))
report['official_data'] = []
for path in tasks:
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    snapshots = list((path.parent / 'snapshots').glob('*.tgz'))
    report['official_data'].append({'path': str(path), 'count': len(records),
        'sha256': hashlib.sha256(raw).hexdigest(), 'snapshots': len(snapshots),
        'wheels': len(list((path.parent / 'wheels').glob('*.whl'))),
        'snapshot_bytes': sum(p.stat().st_size for p in snapshots),
        'wheels_bytes': sum(p.stat().st_size for p in (path.parent / 'wheels').glob('*.whl'))})
    # A real read-permission test, no reference content returned to the notebook.
    code = "import json;from pathlib import Path;t=json.loads(Path(" + repr(str(path)) + ").read_text().splitlines()[0]);print(json.dumps({'reference_patch_readable':bool(t.get('patch')),'test_patch_readable':bool(t.get('test_patch'))}))"
    report['ordinary_child_answer_read'] = command([sys.executable, '-c', code])
    try:
        from swegemma.sandbox.subprocess import SubprocessManager
        manager = SubprocessManager(timeout_seconds=15)
        sid = manager.start()
        try:
            result = manager.exec(sid, shlex.quote(sys.executable) + ' -c ' + shlex.quote(code))
            report['official_subprocess_answer_read'] = {'exit_code': result.exit_code,
                'stdout': result.stdout, 'stderr': result.stderr[-1000:]}
        finally:
            manager.stop(sid)
    except Exception as exc:
        report['official_subprocess_answer_read'] = {'error': repr(exc)}
    break
models = list(Path('/kaggle/input').glob('**/gemma-4-31b-it-qat-w4a16-ct/*/config.json'))
report['qat_models'] = [{'path': str(p.parent), 'weights_bytes': sum(x.stat().st_size for x in p.parent.glob('*.safetensors'))}
                        for p in models]
report['generation_authorized_by_probe'] = False
report['reason'] = 'An ordinary subprocess/venv is not a security boundary. A negative read test alone would not establish isolation either.'
(OUT / 'runtime_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
