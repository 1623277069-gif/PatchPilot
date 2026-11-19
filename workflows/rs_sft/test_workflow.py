"""Synthetic fixtures only. No fixture is ever exported as training data."""
import json
import unittest
from split import components, make_split, validate_split
from recording import loopback
from export_sft import annotate_template, candidates, failed_action, loss_sample
from pipeline import sanitized


def fixtures():
    result = []
    for i in range(129):
        repo = 'example/' + ('a' if i < 66 else 'b' if i < 114 else 'c' if i < 127 else 'd')
        result.append({'instance_id': f'task_{i}', 'repo': repo, 'base_commit': f'commit{i}',
                       'problem_statement': f'Unique bug number {i}',
                       'patch': 'SECRET_REFERENCE', 'test_patch': 'SECRET_TEST',
                       'FAIL_TO_PASS': ['SECRET_TEST_NODE'], 'hints_text': 'SECRET_HINT'})
    return result


class WorkflowTests(unittest.TestCase):
    def test_exact_split_deterministic_and_grouped(self):
        tasks = fixtures()
        tasks[1]['base_commit'] = tasks[0]['base_commit']
        split = make_split(tasks)
        validate_split(split, tasks)
        self.assertEqual(split, make_split(list(reversed(tasks))))
        self.assertEqual(len(split['train_ids']), 99)
        self.assertEqual(len(split['validation_ids']), 30)
        self.assertEqual('task_0' in split['train_ids'], 'task_1' in split['train_ids'])
        self.assertNotIn('SECRET_REFERENCE', json.dumps(split))

    def test_tampered_freeze_is_rejected(self):
        split = make_split(fixtures())
        split['train_ids'][0] = 'new_task'
        with self.assertRaises(ValueError):
            validate_split(split)

    def test_missing_real_tasks_is_rejected(self):
        with self.assertRaises(ValueError):
            make_split(fixtures()[:128])

    def test_impossible_group_split_is_rejected(self):
        tasks = fixtures()
        for task in tasks:
            task.update(repo='same/repo', base_commit='same_commit')
        with self.assertRaises(ValueError):
            make_split(tasks)

    def test_issue_family_transitively_grouped(self):
        tasks = fixtures()[:3]
        tasks[0]['problem_statement'] += ' https://github.com/example/a/issues/15'
        tasks[1]['problem_statement'] += ' https://github.com/example/a/issues/15'
        tasks[2]['base_commit'] = tasks[1]['base_commit']
        self.assertEqual(len(components(tasks)), 1)

    def test_no_secrets_in_agent_task(self):
        clean = sanitized(fixtures()[0])
        self.assertNotIn('SECRET', json.dumps(clean))
        self.assertEqual(set(clean), {'instance_id', 'repo', 'base_commit', 'problem_statement'})

    def test_no_external_api_or_redirect_target(self):
        for url in ['https://openrouter.ai/api/v1', 'http://localhost:8000/v1',
                    'http://127.0.0.1.evil:8000/v1', 'http://user@127.0.0.1:8000/v1',
                    'http://127.0.0.1:8000/v1?url=evil']:
            with self.assertRaises(ValueError):
                loopback(url)
        self.assertEqual(loopback('http://127.0.0.1:8000/v1'), 'http://127.0.0.1:8000/v1')

    def test_tool_error_classification(self):
        self.assertTrue(failed_action({'status': 'error'}))
        self.assertTrue(failed_action({'details': {'exit_code': 1}}))
        self.assertFalse(failed_action({'status': 'ok', 'exit_code': 0}))
        self.assertIsNone(failed_action('unstructured outcome'))

    def test_failed_actions_filtered_by_id_not_name(self):
        target = {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'bad', 'type': 'function', 'function': {'name': 'edit_file', 'arguments': '{}'}}]}
        events = [{'type': 'request', 'call_id': 1, 'body': {'messages': []}},
                  {'type': 'response', 'call_id': 1, 'body': {'choices': [
                      {'finish_reason': 'tool_calls', 'message': target}]}},
                  {'type': 'request', 'call_id': 2, 'body': {'messages': [
                      {'role': 'tool', 'tool_call_id': 'bad', 'content': '{"status":"error"}'},
                      {'role': 'tool', 'tool_call_id': 'good', 'content': '{"status":"ok"}'}]}}]
        self.assertEqual(list(candidates(events)), [])
        events[1]['body']['choices'][0]['message']['tool_calls'][0]['id'] = 'good'
        self.assertEqual(len(list(candidates(events))), 1)

    def test_template_annotation_targets_only_final_assistant(self):
        source = "{%- for message in loop_messages -%}{{ message.content }}{% for x in [] %}{{ x }}{% endfor %}{%- endfor -%}"
        marked = annotate_template(source)
        self.assertIn("loop.last and message['role'] == 'assistant'", marked)
        self.assertEqual(marked.count('{% generation %}'), 1)


class ToyTokenizer:
    """Tests label handling, not real Gemma tokenization compatibility."""
    def get_chat_template(self, **kwargs):
        return "{% for message in loop_messages %}{{ message.content }}{% endfor %}"

    def apply_chat_template(self, messages, tokenize=True, return_dict=False,
                            return_assistant_tokens_mask=False, **kwargs):
        chunks = [json.dumps(x, sort_keys=True) for x in messages]
        text = '|'.join(chunks)
        if not tokenize:
            return text
        ids = [ord(x) for x in text]
        if not return_dict:
            return ids
        start = len(text) - len(chunks[-1])
        return {'input_ids': ids, 'assistant_masks': [0] * start + [1] * len(chunks[-1])}


class MaskTests(unittest.TestCase):
    def test_transformers5_default_dict_does_not_become_field_names(self):
        class DefaultDictTokenizer(ToyTokenizer):
            def apply_chat_template(self, *args, return_dict=True, **kwargs):
                return super().apply_chat_template(*args, return_dict=return_dict, **kwargs)
        sample = loss_sample(DefaultDictTokenizer(), [{'role': 'user', 'content': 'context'}],
                             {'role': 'assistant', 'content': 'target'}, [], {}, 8192)
        self.assertTrue(all(isinstance(token, int) for token in sample['input_ids']))
        self.assertTrue(any(x != -100 for x in sample['labels']))

    def test_only_current_assistant_gets_loss(self):
        messages = [{'role': 'system', 'content': 'POLICY'}, {'role': 'user', 'content': 'TASK'},
                    {'role': 'assistant', 'content': 'earlier error action'},
                    {'role': 'tool', 'content': 'actual error result', 'tool_call_id': 'x'}]
        target = {'role': 'assistant', 'content': 'current valid action'}
        sample = loss_sample(ToyTokenizer(), messages, target, [], {}, 8192)
        prefix = sample['assistant_prefix_length']
        self.assertTrue(all(x == -100 for x in sample['labels'][:prefix]))
        self.assertEqual(sample['labels'][prefix:], sample['input_ids'][prefix:])
        self.assertIn('actual error result', sample['messages'][-2]['content'])

    def test_overlength_rejected_without_truncation(self):
        with self.assertRaises(ValueError):
            loss_sample(ToyTokenizer(), [], {'role': 'assistant', 'content': 'long'}, [], {}, 2)

    def test_nonassistant_target_rejected(self):
        with self.assertRaises(ValueError):
            loss_sample(ToyTokenizer(), [], {'role': 'tool', 'content': 'result'}, [], {}, 8192)

    def test_annotation_cannot_change_rendered_template(self):
        class WrongTemplate(ToyTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                output = super().apply_chat_template(*args, **kwargs)
                if not kwargs.get('tokenize') and '{% generation %}' in kwargs['chat_template']:
                    return 'different'
                return output
        with self.assertRaises(ValueError):
            loss_sample(WrongTemplate(), [], {'role': 'assistant', 'content': 'x'}, [], {}, 8192)

    def test_native_mask_missing_is_not_silently_all_token_loss(self):
        class MissingMask(ToyTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                output = super().apply_chat_template(*args, **kwargs)
                if isinstance(output, dict):
                    output.pop('assistant_masks')
                return output
        with self.assertRaises(ValueError):
            loss_sample(MissingMask(), [], {'role': 'assistant', 'content': 'x'}, [], {}, 8192)

    def test_embedded_tool_observation_not_supervised(self):
        with self.assertRaises(ValueError):
            loss_sample(ToyTokenizer(), [], {'role': 'assistant', 'content': 'x',
                        'tool_responses': [{'name': 'read_file', 'response': 'actual output'}]}, [], {}, 8192)


if __name__ == '__main__':
    unittest.main()
