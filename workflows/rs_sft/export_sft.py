"""Only verified successful base trajectories; per-turn assistant-only targets."""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from common import ROOT, MODEL, digest, read_json, rows, write_json
from split import validate_split


def normalize_messages(messages):
    messages = copy.deepcopy(messages)
    for message in messages:
        if message.get('role') not in {'system', 'developer', 'user', 'assistant', 'tool'}:
            raise ValueError('Unknown message role')
        for call in message.get('tool_calls', []):
            function = call['function']
            if isinstance(function['arguments'], str):
                function['arguments'] = json.loads(function['arguments'])
            if not isinstance(function['arguments'], dict):
                raise ValueError('Tool arguments must be objects for Gemma template')
    return messages


def failed_action(result):
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return None  # Unstructured outcomes require review; never assume success.
    if not isinstance(result, dict):
        return None
    if result.get('status') in {'error', 'failed', 'failure'} or result.get('error') or result.get('error_message'):
        return True
    if result.get('exit_code', 0) != 0:
        return True
    details = result.get('details')
    if isinstance(details, dict) and details.get('exit_code', 0) != 0:
        return True
    if result.get('status') in {'ok', 'success'} or result.get('exit_code') == 0:
        return False
    return None


def annotate_template(template):
    """Mark only the final assistant iteration without rewriting its contents.

    Gemma's template can merge assistant/tool/assistant into the same turn and
    looks ahead at tool results. Prefix tokenization alone is therefore unsafe.
    Capture each original loop iteration and mark only the last one; the caller
    proves the instrumented and canonical rendered bytes/token IDs are equal.
    """
    anchor = re.search(r'{%[-+]?\s*for\s+message\s+in\s+loop_messages\s*[-+]?%}', template)
    if not anchor:
        raise ValueError('Unsupported canonical Gemma message loop; do not guess labels')
    depth = 1
    ending = None
    for tag in re.finditer(r'{%[-+]?\s*(for\b|endfor\b)[\s\S]*?%}', template[anchor.end():]):
        depth += 1 if tag.group(1) == 'for' else -1
        if depth == 0:
            ending = anchor.end() + tag.start()
            break
    if ending is None:
        raise ValueError('Unbalanced canonical template loop')
    body = template[anchor.end():ending]
    wrapped = ("{% set rs_sft_turn %}" + body + "{% endset %}"
               "{% if loop.last and message['role'] == 'assistant' %}"
               "{% generation %}{{ rs_sft_turn }}{% endgeneration %}"
               "{% else %}{{ rs_sft_turn }}{% endif %}")
    return template[:anchor.end()] + wrapped + template[ending:]


def loss_sample(tokenizer, messages, target, tools, template_kwargs, max_length):
    """Tokenize one original context + one actual assistant response.

    Byte/token equality with the canonical template plus a final-message-only
    generation block proves context tokens cannot receive loss. Never guess
    offsets or silently truncate on chat-template incompatibility.
    Error actions remain masked in later contexts; they are never targets.
    """
    kwargs = {k: v for k, v in template_kwargs.items() if k in {'enable_thinking', 'preserve_thinking'}}
    context = normalize_messages(messages)
    target = normalize_messages([target])[0]
    if target.get('tool_responses'):
        raise ValueError('Assistant target must not embed tool observations')
    if target['role'] != 'assistant':
        raise ValueError('Only assistant targets')
    sequence = context + [target]
    template = tokenizer.get_chat_template(tools=tools or None)
    annotated = annotate_template(template)
    shared = dict(tools=tools or None, add_generation_prompt=False, **kwargs)
    canonical_text = tokenizer.apply_chat_template(sequence, chat_template=template, tokenize=False, **shared)
    marked_text = tokenizer.apply_chat_template(sequence, chat_template=annotated, tokenize=False, **shared)
    if canonical_text != marked_text:
        raise ValueError('Mask annotation changed canonical rendered bytes')
    # Transformers 5 returns BatchEncoding by default. Request IDs explicitly;
    # list(BatchEncoding) would compare field names with actual token IDs.
    full = list(tokenizer.apply_chat_template(sequence, chat_template=template, tokenize=True,
                                              return_dict=False, **shared))
    marked = tokenizer.apply_chat_template(sequence, chat_template=annotated, tokenize=True,
                                           return_dict=True, return_assistant_tokens_mask=True, **shared)
    mask = list(marked.get('assistant_masks', []))
    if list(marked['input_ids']) != full or len(mask) != len(full) or not any(mask):
        raise ValueError('Tokenizer does not provide a valid assistant mask')
    if len(full) > max_length:
        raise ValueError('Overlength sample rejected, not truncated')
    if any(value not in {0, 1} for value in mask):
        raise ValueError('Invalid mask values')
    # The marker covers exactly the final assistant's canonical serialized turn,
    # including its role delimiter and tool-call/termination tokens. No role:tool
    # observation is appended to the target, hence none is supervised.
    labels = [token if flag else -100 for token, flag in zip(full, mask)]
    return {'input_ids': full, 'attention_mask': [1] * len(full), 'labels': labels,
            'messages': context + [target], 'tools': tools,
            'loss_mask': [int(x != -100) for x in labels],
            'assistant_prefix_length': next(i for i, x in enumerate(mask) if x)}


def eligible_directory(directory, split):
    generation = read_json(directory / 'generation.json')
    verify = read_json(directory / 'verification.json')
    baseline = read_json(directory / 'baseline_verification.json')
    iid = directory.parent.name
    if iid not in split['train_ids'] or iid in split['validation_ids']:
        raise ValueError('Validation data must never be exported')
    if (generation.get('model') != MODEL or generation.get('variant') != 'base'
            or generation.get('split_sha256') != split['freeze_sha256']
            or generation.get('error')):
        raise ValueError('Invalid generation provenance')
    if (verify.get('resolved') is not True or verify.get('test_exit_code') != 0
            or verify.get('error') or verify.get('error_message')
            or baseline.get('resolved') is not False or baseline.get('test_exit_code') != 1):
        raise ValueError('Success/baseline verification gate failed')
    patch = (directory / 'agent.patch').read_bytes()
    if not patch.strip() or hashlib.sha256(patch).hexdigest() != generation['patch_sha256']:
        raise ValueError('Patch missing or changed')
    if read_json(directory / 'state.json').get('status') != 'complete':
        raise ValueError('Attempt incomplete')
    return generation


def candidates(events):
    requests = {e['call_id']: e['body'] for e in events if e['type'] == 'request'}
    responses = {e['call_id']: e['body'] for e in events if e['type'] == 'response'}
    # Tool results are matched by call ID, never by function name or array position.
    results = {}
    for request in requests.values():
        for message in request['messages']:
            if message.get('role') == 'tool':
                key = message.get('tool_call_id')
                value = message.get('content')
                if key in results and results[key] != value:
                    raise ValueError('Conflicting tool observations')
                results[key] = value
    for cid in sorted(requests):
        response = responses.get(cid)
        if not response or len(response.get('choices', [])) != 1:
            continue
        choice = response['choices'][0]
        if choice.get('finish_reason') not in {'stop', 'tool_calls'}:
            continue
        target = choice['message']
        calls = target.get('tool_calls', [])
        if any(c.get('id') not in results or failed_action(results[c['id']]) is not False for c in calls):
            continue  # Reject full action turn, retaining original context later.
        yield cid, requests[cid], target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=ROOT / 'data/split.json')
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/sft.jsonl')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    split = read_json(args.split)
    validate_split(split)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'data'):
        raise ValueError('SFT output must remain inside the independent data directory')
    if output.exists():
        raise FileExistsError('SFT data already frozen; do not overwrite')
    samples, rejected, seen = [], [], set()
    for directory in sorted((ROOT / 'runs/generate/base').glob('*/attempt-*')):
        try:
            provenance = eligible_directory(directory, split)
            for call_id, request, target in candidates(rows(directory / 'model.jsonl')):
                try:
                    sample = loss_sample(tokenizer, request['messages'], target, request.get('tools', []),
                                         request.get('chat_template_kwargs', {}), args.max_length)
                    key = digest([sample['input_ids'], sample['labels']])
                    if key in seen:
                        continue
                    seen.add(key)
                    sample.update(task_id=directory.parent.name, attempt=directory.name,
                                  call_id=call_id, provenance=provenance)
                    samples.append(sample)
                except ValueError as exc:
                    rejected.append({'directory': str(directory), 'call_id': call_id, 'reason': str(exc)})
        except (ValueError, FileNotFoundError) as exc:
            rejected.append({'directory': str(directory), 'reason': str(exc)})
    write_json(ROOT / 'evidence' / (output.stem + '_export_report.json'), {'accepted_turns': len(samples), 'rejections': rejected,
               'tokenizer_template_sha256': digest(tokenizer.chat_template), 'max_length': args.max_length})
    if not samples:
        raise RuntimeError('No eligible real samples. No empty/fabricated SFT file written.')
    temporary = output.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in samples))
    temporary.replace(output)
    write_json(ROOT / 'data/sft_manifest.json', {'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'samples': len(samples), 'split_sha256': split['freeze_sha256'],
               'base_fingerprint': samples[0]['provenance']['base_fingerprint']})


if __name__ == '__main__':
    main()
