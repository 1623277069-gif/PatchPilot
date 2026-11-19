"""Official swegemma two-phase orchestration; no alternate tools or teachers."""
import argparse
import asyncio
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time
from common import ROOT, MODEL, digest, journal, model_identity, read_json, rows, write_json
from split import validate_split
from recording import local_json, loopback, manager_with_journal, recording_proxy


def sanitized(task):
    return {k: task[k] for k in ['instance_id', 'repo', 'base_commit', 'problem_statement']}


def snapshot_agent_check():
    manifest = read_json(ROOT / 'evidence/agent_manifest.json')
    for rel, expected in manifest['files'].items():
        if hashlib.sha256((ROOT / 'agent' / rel).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen agent changed: ' + rel)
    return manifest['fingerprint']


async def run(args):
    scope = read_json(ROOT / 'configs/experiment.json').get('execution_scope', {})
    if args.stage == 'generate' and not scope.get('batch_generation_allowed', False):
        if not args.task_id or args.attempts != 1:
            raise RuntimeError('Current scope permits one explicitly selected training task only; batch generation disabled')
    if args.stage == 'evaluate' and not scope.get('validation_evaluation_allowed', False):
        raise RuntimeError('Frozen validation evaluation is disabled until the real single-task and single-step gates pass')
    from adk_submission import ModelRegistry
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.config import EvalConfig
    from swegemma.context import SwegemmaContext
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.harness.agent_runner import run_agent_sandbox
    from swegemma.harness.verification import verify_task
    from swegemma.models import Task
    from swegemma.sandbox.docker import ContainerManager, ContainerConfig
    import litellm
    litellm.drop_params = True
    split = read_json(args.split)
    tasks = rows(args.tasks)
    validate_split(split, tasks)
    identity = model_identity(args.model_dir)
    agent_hash = snapshot_agent_check()
    endpoint = loopback(args.endpoint)
    readiness = read_json(ROOT / 'evidence/server_readiness.json')
    if readiness['base']['fingerprint'] != identity['fingerprint'] or readiness['url'] != endpoint:
        raise ValueError('Official server readiness receipt does not match model/endpoint')
    if args.stage == 'generate' and readiness.get('adapter') is not None:
        raise ValueError('Self-generation requires the original base-only service')
    listed = local_json(endpoint + '/models')['data']
    allowed_names = {MODEL, str(args.model_dir.resolve())}
    adapter_fingerprint = None
    if args.variant == 'lora':
        allowed_names = {args.adapter_name}
        receipt = read_json(args.adapter / 'official_load_receipt.json')
        if receipt.get('passed') is not True or receipt['base_fingerprint'] != identity['fingerprint']:
            raise ValueError('Adapter official loading gate missing or mismatched')
        if receipt['adapter_sha256'] != hashlib.sha256((args.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest():
            raise ValueError('Adapter changed after loading gate')
        adapter_fingerprint = receipt['adapter_sha256']
    matched = [m['id'] for m in listed if m['id'] in allowed_names]
    if len(matched) != 1:
        raise ValueError('Local service must expose the exact requested base/adapter model')
    served_model = matched[0]
    if args.stage == 'generate' and args.variant != 'base':
        raise ValueError('RS samples must originate from original base model')
    ids = split['train_ids' if args.stage == 'generate' else 'validation_ids']
    if args.task_id:
        if args.task_id not in ids:
            raise ValueError('Task not in requested frozen partition')
        ids = [args.task_id]
    by_id = {t['instance_id']: t for t in tasks}
    if args.stage == 'evaluate' and (args.attempts != 1 or args.attempt_start != 0):
        raise ValueError('Validation uses attempt-00 only, one fixed attempt per model/task')
    for iid in ids:
        for attempt in range(args.attempt_start, args.attempt_start + args.attempts):
            directory = ROOT / 'runs' / args.stage / args.variant / iid / f'attempt-{attempt:02d}'
            directory.mkdir(parents=True, exist_ok=True)
            fingerprint = digest([split['freeze_sha256'], identity['fingerprint'], agent_hash,
                                  args.stage, args.variant, iid, attempt, args.adapter_name,
                                  adapter_fingerprint])
            state_path = directory / 'state.json'
            state = read_json(state_path) if state_path.exists() else {}
            if state and state.get('fingerprint') != fingerprint:
                raise ValueError('Resume fingerprint mismatch')
            if state.get('status') == 'complete':
                continue
            if state.get('status') == 'generating':
                raise RuntimeError(f'Interrupted Phase 1 in {directory}; kept intact. Use a new attempt index; in-flight sandbox replay is not supported.')
            task = Task(**{f.name: by_id[iid][f.name] for f in fields(Task) if f.name in by_id[iid]})
            safe_task = Task(**sanitized(by_id[iid]))
            snapshot, base, patch = resolve_task_snapshot_paths(args.snapshots, iid, task.repo)
            if not snapshot.is_file():
                raise FileNotFoundError(snapshot)
            registry = ModelRegistry()
            cfg = EvalConfig(tasks_path=args.tasks, snapshots_dir=args.snapshots,
                             results_dir=directory, submission_dir=ROOT / 'agent', models=registry,
                             sandbox='docker', image=args.image, wheels_dir=args.wheels,
                             max_tool_calls=28, max_turns=80, max_time_minutes=7,
                             timeout_seconds=60, verification_timeout_seconds=300,
                             graph_dir=str(args.graphs), embeddings_dir=str(args.embeddings),
                             display_mode='quiet')
            if state.get('status') != 'generated':
                write_json(state_path, {'status': 'generating', 'fingerprint': fingerprint})
                recorder = manager_with_journal(directory / 'execution.jsonl', args.image)
                # No host mounts. The official Docker backend copies only the selected
                # base snapshot into the container; secret tasks stay on the host.
                with recording_proxy(endpoint, served_model, directory / 'model.jsonl') as proxy:
                    registry.register(MODEL, LiteLlm(model='openai/' + MODEL, api_base=proxy,
                                                    api_key='EMPTY', num_retries=0))
                    class RecordedContext(SwegemmaContext):
                        def handle_adk_event(self, event, **kwargs):
                            journal(directory / 'adk_events.jsonl', {'event': event.model_dump(mode='json', exclude_none=True)})
                            return super().handle_adk_event(event, **kwargs)
                    context = RecordedContext(docker_manager=recorder, task=safe_task,
                                             problem_statement=safe_task.problem_statement,
                                             budget=cfg.budget, harness=cfg.harness,
                                             graph_dir=str(args.graphs), embeddings_dir=str(args.embeddings))
                    started = time.perf_counter()
                    generated, error, trace = await run_agent_sandbox(
                        recorder, cfg, safe_task, snapshot, base_snapshot_path=base,
                        patch_path=patch, context=context)
                    duration = time.perf_counter() - started
                    trace.save(directory / 'official_trace.json')
                    (directory / 'agent.patch').write_text(generated)
                    write_json(directory / 'generation.json', {
                        'error': error, 'duration_seconds': duration,
                        'tool_calls': context.tool_calls_used, 'llm_calls': context.llm_calls_used,
                        'model': MODEL, 'variant': args.variant, 'base_fingerprint': identity['fingerprint'],
                        'adapter_sha256': adapter_fingerprint,
                        'split_sha256': split['freeze_sha256'], 'agent_fingerprint': agent_hash,
                        'serving': readiness,
                        'patch_sha256': hashlib.sha256(generated.encode()).hexdigest()})
                write_json(state_path, {'status': 'generated', 'fingerprint': fingerprint})
            generated = (directory / 'agent.patch').read_text()
            generation = read_json(directory / 'generation.json')
            # New managers for verification, never connected to the model. Do NOT
            # call Evaluator.evaluate_task: its hydration would give Phase 1 secrets.
            verifier = ContainerManager(ContainerConfig(image=args.image, network_mode='none'))
            kwargs = {'base_snapshot_path': base, 'patch_path': patch}
            baseline = await verify_task(verifier, cfg, task, snapshot, agent_patch='', **kwargs)
            candidate = await verify_task(verifier, cfg, task, snapshot, agent_patch=generated,
                                          agent_error=generation['error'], **kwargs)
            def result_data(result):
                return result.model_dump(mode='json', exclude={'trace', 'agent_patch'})
            write_json(directory / 'baseline_verification.json', result_data(baseline))
            write_json(directory / 'verification.json', result_data(candidate))
            accepted = (bool(generated.strip()) and not generation['error']
                        and baseline.resolved is False and baseline.test_exit_code == 1
                        and candidate.resolved is True and candidate.test_exit_code == 0
                        and not (getattr(candidate, 'error', None) or getattr(candidate, 'error_message', None)))
            write_json(state_path, {'status': 'complete', 'fingerprint': fingerprint,
                                   'accepted': accepted, 'base_fingerprint': identity['fingerprint']})
            journal(ROOT / 'runs/ledger.jsonl', {'stage': args.stage, 'variant': args.variant,
                                                'task_id': iid, 'attempt': attempt, 'accepted': accepted})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['generate', 'evaluate'])
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=ROOT / 'data/split.json')
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--graphs', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--wheels', type=Path, required=True)
    parser.add_argument('--image', default='swebench-sandbox:latest')
    parser.add_argument('--attempts', type=int, default=1)
    parser.add_argument('--task-id')
    parser.add_argument('--variant', choices=['base', 'lora'], default='base')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--adapter-name', default='rs_lora')
    parser.add_argument('--attempt-start', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.attempts <= 8 or args.attempt_start < 0:
        parser.error('attempts must be 1..8 and attempt-start must be nonnegative')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
