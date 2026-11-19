"""Real local Gemma tokenizer probe. Fixtures never become SFT training data."""
import argparse
import hashlib
from pathlib import Path
from common import ROOT, write_json
from export_sft import loss_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tools = [{'type': 'function', 'function': {'name': 'run_command',
              'description': 'Execute command', 'parameters': {'type': 'object',
              'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]
    action = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_1',
              'type': 'function', 'function': {'name': 'run_command',
              'arguments': {'command': 'echo MASK_PROBE'}}}]}
    cases = [([{'role': 'user', 'content': 'USER_SENTINEL'}], action),
             ([{'role': 'user', 'content': 'USER_SENTINEL'}, action,
               {'role': 'tool', 'name': 'run_command', 'tool_call_id': 'call_1',
                'content': 'TOOL_SENTINEL'}],
              {'role': 'assistant', 'content': 'FINAL_SENTINEL'})]
    results = []
    for context, target in cases:
        sample = loss_sample(tokenizer, context, target, tools, {}, 8192)
        supervised = tokenizer.decode([x for x in sample['labels'] if x != -100])
        assert 'USER_SENTINEL' not in supervised and 'TOOL_SENTINEL' not in supervised
        assert ('MASK_PROBE' if target.get('tool_calls') else 'FINAL_SENTINEL') in supervised
        results.append({'passed': True, 'tokens': len(sample['input_ids']),
                        'supervised_tokens': sum(sample['loss_mask']),
                        'supervised_text': supervised})
    receipt = {'fixture_only': True, 'training_rows_written': 0, 'results': results,
               'template_sha256': hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()}
    write_json(ROOT / 'evidence/tokenizer_smoke.json', receipt)
    print(receipt)


if __name__ == '__main__':
    main()
