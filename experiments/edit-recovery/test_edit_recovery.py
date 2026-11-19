"""Behavioral tests for the exact script shipped in the submission."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'skill/scripts/edit_lines.py'
spec = importlib.util.spec_from_file_location('edit_lines', SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'src').mkdir()
        self.file = self.root / 'src/module.py'
        self.file.write_bytes(b'def value():\n    return 1\n\nOTHER = 8\n')

    def run_helper(self, *args):
        return helper.perform(list(args), workspace=self.root)

    def digest(self):
        return hashlib.sha256(self.file.read_bytes()).hexdigest()

    def assert_rejected(self, code, *args):
        before = self.file.read_bytes()
        with self.assertRaises(helper.Rejected) as caught:
            self.run_helper(*args)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(before, self.file.read_bytes())
        self.assertFalse(list(self.root.rglob('.edit-recovery-*')))

    def test_read_numbers_hash_and_clamped_eof(self):
        result = self.run_helper('read', '/workspace/src/module.py', '2', '99')
        self.assertEqual((result['start'], result['end']), (2, 4))
        self.assertEqual(result['sha256'], self.digest())
        self.assertEqual(result['lines'][0], {'number': 2, 'text': '    return 1'})

    def test_replace_preserves_unrelated_lines_and_executable_mode(self):
        self.file.chmod(0o755)
        original = self.digest()
        result = self.run_helper('replace', 'src/module.py', '2', '2', original, '    return 2')
        self.assertEqual(self.file.read_bytes(), b'def value():\n    return 2\n\nOTHER = 8\n')
        self.assertEqual(stat.S_IMODE(self.file.stat().st_mode), 0o755)
        self.assertEqual(result['previous_sha256'], original)
        self.assertEqual(result['sha256'], self.digest())
        self.assertIn('+    return 2', result['diff'])

    def test_quotes_backslashes_unicode_are_not_unescaped_twice(self):
        replacement = '    return "中文", "\\\\", \'quoted\'\n'
        self.run_helper('replace', 'src/module.py', '2', '2', self.digest(), replacement)
        self.assertEqual(self.file.read_text().splitlines()[1], replacement.rstrip('\n'))

    def test_crlf_and_surrounding_lines(self):
        self.file.write_bytes(b'x = 1\r\ny = 2\r\n')
        self.run_helper('replace', 'src/module.py', '1', '1', self.digest(), 'x = 3')
        self.assertEqual(self.file.read_bytes(), b'x = 3\r\ny = 2\r\n')

    def test_eof_no_newline_and_deletion(self):
        self.file.write_bytes(b'x = 1\ny = 2')
        self.run_helper('replace', 'src/module.py', '2', '2', self.digest(), 'y = 3')
        self.assertEqual(self.file.read_bytes(), b'x = 1\ny = 3')
        self.run_helper('replace', 'src/module.py', '2', '2', self.digest(), '')
        self.assertEqual(self.file.read_bytes(), b'x = 1\n')

    def test_bom_python_is_accepted(self):
        self.file.write_bytes(b'\xef\xbb\xbfx = 1\ny = 2\n')
        self.run_helper('replace', 'src/module.py', '2', '2', self.digest(), 'y = 3\n')
        self.assertTrue(self.file.read_bytes().startswith(b'\xef\xbb\xbf'))

    def test_stale_hash_rejected(self):
        old = self.digest()
        self.file.write_bytes(self.file.read_bytes() + b'CHANGED = True\n')
        self.assert_rejected('STALE_FILE', 'replace', 'src/module.py', '2', '2', old, '    return 3\n')

    def test_invalid_python_rejected_before_write(self):
        self.assert_rejected('INVALID_PYTHON', 'replace', 'src/module.py', '2', '2', self.digest(), '    return (\n')

    def test_invalid_python_scope_rejected_without_execution(self):
        self.file.write_bytes(b'x = 1\n')
        for statement in ['return 2\n', 'break\n', 'await work()\n']:
            with self.subTest(statement=statement):
                self.assert_rejected('INVALID_PYTHON', 'replace', 'src/module.py', '1', '1', self.digest(), statement)
        self.run_helper('replace', 'src/module.py', '1', '1', self.digest(), 'raise RuntimeError("must not execute")\n')
        self.assertIn('raise RuntimeError', self.file.read_text())

    def test_bad_ranges_and_noop_rejected(self):
        for start, end in [('0', '1'), ('3', '2'), ('a', '2'), ('1', '121'), ('8', '9')]:
            with self.subTest(start=start, end=end):
                self.assert_rejected('INVALID_RANGE', 'read', 'src/module.py', start, end)
        self.assert_rejected('INVALID_RANGE', 'replace', 'src/module.py', '1', '8', self.digest(), '')
        self.assert_rejected('NO_CHANGE', 'replace', 'src/module.py', '2', '2', self.digest(), '    return 1\n')

    def test_paths_symlinks_and_missing_file_rejected(self):
        for name in ['../module.py', '/tmp/module.py', '.git/config']:
            self.assert_rejected('INVALID_PATH', 'read', name, '1', '1')
        self.assert_rejected('NOT_A_FILE', 'read', 'src/missing.py', '1', '1')
        (self.root / 'link.py').symlink_to(self.file)
        self.assert_rejected('SYMLINK', 'read', 'link.py', '1', '1')

    def test_protected_file_edit_rejected_but_read_allowed(self):
        for name in ['tests/a.py', 'test_any.py', 'some_test.py', 'pyproject.toml', 'custom.pth']:
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('value = 1\n')
            digest = self.run_helper('read', name, '1', '1')['sha256']
            self.assert_rejected('PROTECTED_FILE', 'replace', name, '1', '1', digest, 'value = 2\n')
            self.assertEqual(path.read_text(), 'value = 1\n')

    def test_large_replacement_and_nul_rejected(self):
        self.assert_rejected('REPLACEMENT_TOO_LARGE', 'replace', 'src/module.py', '2', '2', self.digest(), 'x' * 24001)
        self.assert_rejected('BINARY_TEXT', 'replace', 'src/module.py', '2', '2', self.digest(), '\x00')

    def test_write_rechecks_file_and_cleans_temporary_on_late_conflict(self):
        from unittest.mock import patch
        original = helper.target_path
        calls = 0
        def conflict(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.file.write_bytes(b'CHANGED = True\n')
            return original(*args, **kwargs)
        with patch.object(helper, 'target_path', conflict):
            with self.assertRaises(helper.Rejected) as caught:
                self.run_helper('replace', 'src/module.py', '2', '2', self.digest(), '    return 3\n')
        self.assertEqual(caught.exception.code, 'STALE_FILE')
        self.assertEqual(self.file.read_bytes(), b'CHANGED = True\n')
        self.assertFalse(list(self.root.rglob('.edit-recovery-*')))

    def test_cli_reports_error_as_json_and_nonzero_exit(self):
        env = dict(os.environ, GEMMA_EDIT_WORKSPACE=str(self.root))
        result = subprocess.run([sys.executable, str(SCRIPT), 'read', 'src/module.py'],
                                env=env, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)['error_code'], 'USAGE')


if __name__ == '__main__':
    unittest.main(verbosity=2)
