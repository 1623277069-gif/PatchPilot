"""Negative tests for controller packaging; no inference or training fixtures."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from common import ROOT, write_json
from separation import audit_archive, prepare
from split import make_split
from test_workflow import fixtures


class BoundaryTests(unittest.TestCase):
    def archive(self, name, kind=tarfile.REGTYPE):
        f = tempfile.NamedTemporaryFile(suffix='.tgz', delete=False)
        f.close()
        self.addCleanup(Path(f.name).unlink)
        with tarfile.open(f.name, 'w:gz') as tar:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = 1 if kind == tarfile.REGTYPE else 0
            tar.addfile(info, io.BytesIO(b'x') if info.size else None)
        return Path(f.name)

    def test_archive_history_and_answers_rejected(self):
        for name in ['.git/objects/ab/cd', 'repo/.git/logs/HEAD', 'tasks.jsonl',
                     'old.ipynb', 'test_patch.diff', '../escape', '/abs']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit_archive(self.archive(name))

    def test_archive_symlinks_hardlinks_and_devices_rejected(self):
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE]:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit_archive(self.archive('link', kind))

    def test_structural_pass_does_not_authorize_generation(self):
        result = audit_archive(self.archive('src/code.py'))
        self.assertFalse(result['generation_authorized'])
        self.assertEqual(result['base_tree_provenance'], 'NOT_VERIFIED')

    def test_validation_task_rejected_before_output(self):
        tasks = fixtures()
        split = make_split(tasks)
        with tempfile.TemporaryDirectory() as folder:
            tasks_path = Path(folder) / 'synthetic.jsonl'
            split_path = Path(folder) / 'split.json'
            tasks_path.write_text(''.join(json.dumps(task) + '\n' for task in tasks))
            write_json(split_path, split)
            with self.assertRaisesRegex(ValueError, 'frozen training candidate'):
                prepare(tasks_path, split_path,
                    split['validation_ids'][0], ROOT / 'staging/generator/forbidden')

    def test_task_export_whitelist_only(self):
        # In-memory check; never export these fixtures as SFT.
        from pipeline import sanitized
        task = fixtures()[0]
        public = sanitized(task)
        self.assertEqual(set(public), {'instance_id', 'repo', 'base_commit', 'problem_statement'})
        self.assertNotIn('test_patch', public)
        self.assertNotIn('patch', public)


if __name__ == '__main__':
    unittest.main()
