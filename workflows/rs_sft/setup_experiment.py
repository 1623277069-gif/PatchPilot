"""Fingerprint the bundled agent and report resources; never install or buy."""
import argparse
import ast
import hashlib
import importlib.util
import platform
from pathlib import Path
import shutil
import subprocess
from common import ROOT, MODEL, digest, read_json, write_json
from split import make_split

SOURCE = ROOT / 'agent'


def snapshot():
    hashes = {}
    for file in sorted(SOURCE.rglob('*')):
        if not file.is_file() or file.suffix not in {'.py', '.md', '.yaml'}:
            continue
        rel = str(file.relative_to(SOURCE))
        hashes[rel] = hashlib.sha256(file.read_bytes()).hexdigest()
    if len(hashes) != 6:
        raise ValueError('Expected the six-file Agent bundle')
    manifest = {'source': str(SOURCE.relative_to(ROOT)), 'files': hashes,
                'fingerprint': digest(hashes), 'model': MODEL, 'single_agent': True}
    path = ROOT / 'evidence/agent_manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Original agent changed; new experiment identity required')
    write_json(path, manifest)
    return manifest


def audit(official):
    files = []
    for package in official.glob('*'):
        for file in sorted(package.rglob('*.py')):
            if '__pycache__' in file.parts:
                continue
            ast.parse(file.read_bytes())
            files.append({'path': str(file.relative_to(official)),
                          'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
    write_json(ROOT / 'evidence/official_source_manifest.json', {
        'versions': ['swegemma==0.2.11', 'adk-submission==0.2.13', 'adk-eval-core==0.1.0'],
        'latest_version_verified': False, 'files': files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path)
    parser.add_argument('--official-sources', type=Path)
    args = parser.parse_args()
    agent = snapshot()
    if args.official_sources:
        audit(args.official_sources)
    tasks = args.tasks
    split_status = 'BLOCKED_MISSING_COMPLETE_TASKS'
    if tasks:
        from common import rows
        proposed = make_split(rows(tasks))
        path = ROOT / 'data/split.json'
        if path.exists() and read_json(path) != proposed:
            raise ValueError('Frozen split cannot be changed')
        write_json(path, proposed)
        split_status = 'FROZEN_99_30'
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ['torch', 'transformers', 'peft', 'bitsandbytes', 'swegemma']}
    gpu = None
    if shutil.which('nvidia-smi'):
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=15)
        gpu = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    state = {'scheme': 'Gemma 4 31B QAT rejection-sampling SFT', 'qwen_teacher': 'PAUSED',
             'cost_usd': 0, 'external_teacher_calls': 0,
             'host': platform.platform(), 'machine': platform.machine(),
             'gpu_inventory': gpu, 'docker_cli': shutil.which('docker'), 'packages': packages,
             'tasks_found': [str(tasks)] if tasks else [], 'split_status': split_status,
             'agent_fingerprint': agent['fingerprint'], 'generation_runs': 0,
             'training_steps': 0, 'official_adapter_load_test': 'NOT_RUN',
             'base_vs_lora_evaluation': 'NOT_RUN'}
    write_json(ROOT / 'evidence/local_status.json', state)
    print(state)


if __name__ == '__main__':
    main()
