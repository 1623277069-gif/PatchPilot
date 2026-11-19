"""Inspect the current Kaggle/Linux runtime without purchasing or installing."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
from common import ROOT, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path)
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    blockers = []
    for name in ['swegemma', 'torch', 'transformers', 'peft', 'bitsandbytes', 'compressed_tensors', 'litellm']:
        if importlib.util.find_spec(name) is None:
            blockers.append('missing_package:' + name)
    if not args.tasks or not args.tasks.is_file():
        blockers.append('missing_complete_tasks_jsonl')
    if not args.model_dir or not (args.model_dir / 'config.json').is_file():
        blockers.append('missing_local_QAT31B_weights')
    docker = shutil.which('docker')
    if not docker:
        blockers.append('missing_Docker_for_isolated_official_Sandbox')
    else:
        result = subprocess.run([docker, 'info', '--format', '{{json .ServerVersion}}'],
                                capture_output=True, text=True, timeout=15)
        if result.returncode:
            blockers.append('Docker_daemon_unavailable')
    cuda = None
    if importlib.util.find_spec('torch'):
        import torch
        cuda = {'available': torch.cuda.is_available(), 'count': torch.cuda.device_count(),
                'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        if not cuda['available']:
            blockers.append('CUDA_unavailable')
    else:
        blockers.append('CUDA_unverified_no_torch')
    result = {'status': 'BLOCKED' if blockers else 'READY_FOR_RUNTIME_SMOKE_ONLY',
              'blockers': blockers, 'cuda': cuda, 'paid_calls': 0,
              'note': 'No inference, training, or automatic dependency installation performed.'}
    write_json(ROOT / 'evidence/runtime_check.json', result)
    print(result)
    return 2 if blockers else 0


if __name__ == '__main__':
    raise SystemExit(main())
