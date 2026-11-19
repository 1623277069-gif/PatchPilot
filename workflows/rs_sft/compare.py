"""Report task-level rates on the complete frozen validation set only."""
import argparse
from statistics import mean
from common import ROOT, digest, read_json, rows, write_json
from split import validate_split
from export_sft import failed_action


def metrics(root, ids, freeze_hash):
    results, missing = [], []
    for iid in ids:
        directory = root / iid / 'attempt-00'
        if not (directory / 'state.json').is_file():
            missing.append(iid)
            continue
        state = read_json(directory / 'state.json')
        generation = read_json(directory / 'generation.json')
        if state.get('status') != 'complete' or generation['split_sha256'] != freeze_hash:
            missing.append(iid)
            continue
        result = read_json(directory / 'verification.json')
        baseline = read_json(directory / 'baseline_verification.json')
        if baseline.get('resolved') is not False or baseline.get('test_exit_code') != 1:
            missing.append(iid)
            continue
        calls, observed, errors = 0, {}, {}
        call_ids = set()
        for event in rows(directory / 'adk_events.jsonl'):
            for part in (event['event'].get('content') or {}).get('parts', []):
                call = part.get('function_call')
                response = part.get('function_response')
                if call:
                    if not call.get('id'):
                        raise ValueError('Missing ADK tool call ID: cannot measure reliably')
                    call_ids.add(call['id'])
                if response:
                    key = response.get('id')
                    if not key:
                        raise ValueError('Missing ADK tool response ID: cannot measure reliably')
                    flag = failed_action(response.get('response'))
                    observed[key] = flag
                    if flag is True:
                        errors[key] = True
        calls = len(call_ids)
        results.append({'task_id': iid, 'resolved': result.get('resolved') is True
                        and result.get('test_exit_code') == 0 and not generation.get('error'),
                        'duration_seconds': generation['duration_seconds'],
                        'tool_calls': calls, 'observed_tool_results': len(observed),
                        'classified_tool_results': sum(value is not None for value in observed.values()),
                        'tool_errors': len(errors)})
    tool_total = sum(x['tool_calls'] for x in results)
    observed_total = sum(x['observed_tool_results'] for x in results)
    classified = sum(x['classified_tool_results'] for x in results)
    return {'complete': not missing, 'missing': missing, 'finished_tasks': len(results),
            'resolved': sum(x['resolved'] for x in results),
            'success_rate': sum(x['resolved'] for x in results) / len(ids) if not missing else None,
            'mean_agent_seconds': mean(x['duration_seconds'] for x in results) if results else None,
            'tool_error_rate': sum(x['tool_errors'] for x in results) / tool_total if tool_total and not missing and classified == tool_total else None,
            'confirmed_error_lower_bound': sum(x['tool_errors'] for x in results) / tool_total if tool_total else None,
            'observed_tool_fraction': observed_total / tool_total if tool_total else None,
            'tool_metric_note': 'Official ADK event IDs deduplicate calls/results including skills. Unknown outcomes are not assumed successful.',
            'rows': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    split = read_json(ROOT / 'data/split.json')
    validate_split(split)
    variants = {name: metrics(ROOT / 'runs/evaluate' / name, split['validation_ids'], split['freeze_sha256'])
                for name in ['base', 'lora']}
    complete = all(x['complete'] for x in variants.values())
    report = {'status': 'COMPLETE' if complete else 'INCOMPLETE_NO_COMPARATIVE_CLAIM',
              'split_sha256': split['freeze_sha256'], 'variants': variants,
              'success_rate_delta': (variants['lora']['success_rate'] - variants['base']['success_rate']) if complete else None}
    write_json(ROOT / 'evidence/comparison.json', report)
    print(report['status'])


if __name__ == '__main__':
    main()
