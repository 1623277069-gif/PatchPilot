"""Use the competition's VllmServer. No model download or external endpoint."""
import argparse
import os
from pathlib import Path
import time
from common import ROOT, MODEL, model_identity, read_json, write_json
from recording import local_json
from train import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--adapter-name', default='rs_lora')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--compatibility-only', action='store_true')
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no server or paid fallback started')
    from adk_submission import VllmConfig, VllmServer
    identity = model_identity(args.model_dir)
    os.environ['VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS'] = '1'
    modules = None
    if args.adapter:
        receipt = read_json(args.adapter / 'training_receipt.json')
        if not receipt.get('passed') or receipt['base_fingerprint'] != identity['fingerprint']:
            raise ValueError('Adapter training/reload gate missing or base mismatch')
        if receipt['adapter_sha256'] != file_sha(args.adapter / 'adapter_model.safetensors'):
            raise ValueError('Adapter changed')
        modules = {args.adapter_name: args.adapter}
    server = VllmServer(VllmConfig(
        model=str(args.model_dir.resolve()), host='127.0.0.1', port=args.port,
        tool_call_parser='gemma4', reasoning_parser='gemma4',
        max_model_len=32768, tensor_parallel_size=torch.cuda.device_count(),
        dtype='bfloat16', gpu_memory_utilization=0.8,
        enable_auto_tool_choice=True, enable_lora=bool(modules),
        max_loras=1, max_lora_rank=8, startup_timeout=1200), lora_modules=modules)
    url = f'http://127.0.0.1:{args.port}/v1'
    try:
        server.start()
        listed = local_json(url + '/models')['data']
        names = [m['id'] for m in listed]
        base_name = str(args.model_dir.resolve())
        if base_name not in names:
            raise RuntimeError('Expected exact local base model not listed')
        if args.adapter:
            if args.adapter_name not in names:
                raise RuntimeError('Adapter not listed in official inference server')
            differences = []
            for prompt in ['Return the Python expression for one plus two.',
                           'Explain how to check a patch using repository tests.',
                           'List three prime numbers.']:
                payload = {'messages': [{'role': 'user', 'content': prompt}],
                           'max_tokens': 1, 'temperature': 0, 'logprobs': True, 'top_logprobs': 20,
                           'chat_template_kwargs': {'enable_thinking': False}}
                base = local_json(url + '/chat/completions', dict(payload, model=base_name))
                lora = local_json(url + '/chat/completions', dict(payload, model=args.adapter_name))
                a = base['choices'][0]['logprobs']['content'][0]['top_logprobs']
                b = lora['choices'][0]['logprobs']['content'][0]['top_logprobs']
                am = {x['token']: x['logprob'] for x in a}
                bm = {x['token']: x['logprob'] for x in b}
                differences.append(max((abs(am[k] - bm[k]) for k in am.keys() & bm.keys()), default=0))
            if max(differences) <= 1e-3:
                raise RuntimeError('No clear adapter effect on original CT base; do not train formally')
            write_json(args.adapter / 'official_load_receipt.json', {
                'passed': True, 'base_fingerprint': identity['fingerprint'],
                'adapter_sha256': file_sha(args.adapter / 'adapter_model.safetensors'),
                'max_logprob_deltas': differences,
                'scope': 'official server load and measurable effect; not task-level quality'})
        write_json(ROOT / 'evidence/server_readiness.json', {'base': identity, 'models': names,
                   'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                   'tensor_parallel_size': torch.cuda.device_count(), 'gpu_memory_utilization': 0.8,
                   'max_model_len': 32768,
                   'adapter': str(args.adapter) if args.adapter else None, 'url': url})
        print('Official server ready at', url, flush=True)
        if not args.compatibility_only:
            while True:
                time.sleep(1)
    finally:
        server.stop()


if __name__ == '__main__':
    main()
