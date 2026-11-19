"""Small, checked fallback for existing-file edits. Python standard library only."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

MAX_FILE = 2 * 1024 * 1024
MAX_WINDOW = 24000
MAX_LINES = 120
PROTECTED = {
    'conftest.py', 'pytest.ini', 'pyproject.toml', 'tox.ini', 'setup.cfg',
    '.pytest.ini', 'sitecustomize.py', 'usercustomize.py', '_swegemma_stubs.py',
}


class Rejected(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def reject(code, message):
    raise Rejected(code, message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def target_path(root, name, writing):
    path = Path(name)
    if '..' in path.parts or '.git' in path.parts:
        reject('INVALID_PATH', 'Use a source path inside /workspace, without .. or .git.')
    if path.is_absolute():
        try:
            path = path.relative_to('/workspace')
        except ValueError:
            reject('INVALID_PATH', 'Absolute paths must start with /workspace/.')
    if not path.parts:
        reject('INVALID_PATH', 'A file path is required.')
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            reject('SYMLINK', 'Use a regular file path, not a symlink.')
    if not current.is_file() or not current.resolve().is_relative_to(root):
        reject('NOT_A_FILE', 'The target must be an existing regular workspace file.')
    if writing and (
        current.name in PROTECTED or current.suffix == '.pth'
        or (current.suffix == '.py' and (
            current.name.startswith('test_') or current.name.endswith('_test.py')
            or any(p in {'tests', 'test', 'testing'} for p in path.parts[:-1])
        ))
    ):
        reject('PROTECTED_FILE', 'Edit the implementation, not tests or test configuration.')
    return current, path.as_posix()


def perform(argv, workspace=None):
    if not argv or argv[0] not in {'read', 'replace'}:
        reject('USAGE', 'Expected read FILE START END or replace FILE START END SHA256 NEW_TEXT.')
    operation = argv[0]
    if len(argv) != (4 if operation == 'read' else 6):
        reject('USAGE', 'Pass exactly four read arguments or six replace arguments, as strings.')
    # Override only for local subprocess diagnostics; the competition default is /workspace.
    root = Path(workspace or os.environ.get('GEMMA_EDIT_WORKSPACE', '/workspace')).resolve()
    if not root.is_dir():
        reject('WORKSPACE_NOT_FOUND', 'Workspace is unavailable.')
    path, relative = target_path(root, argv[1], operation == 'replace')
    if path.stat().st_size > MAX_FILE:
        reject('FILE_TOO_LARGE', 'Use a file smaller than 2 MiB.')
    raw = path.read_bytes()
    text = raw.decode('utf-8')
    if '\x00' in text:
        reject('BINARY_FILE', 'The target must be UTF-8 text without NUL bytes.')
    lines = text.splitlines(keepends=True)
    try:
        start, end = int(argv[2]), int(argv[3])
    except (ValueError, TypeError):
        reject('INVALID_RANGE', 'START and END must be integers.')
    if not 1 <= start <= end or end - start + 1 > MAX_LINES or start > len(lines):
        reject('INVALID_RANGE', 'Use an existing 1-based inclusive range of at most 120 lines.')
    if operation == 'read':
        end = min(end, len(lines))
    elif end > len(lines):
        reject('INVALID_RANGE', 'Replacement END is past EOF; use actual line numbers from read.')
    selected = ''.join(lines[start - 1:end])
    if len(selected.encode('utf-8')) > MAX_WINDOW:
        reject('WINDOW_TOO_LARGE', 'Read or replace a smaller range (at most 24000 UTF-8 bytes).')
    result = {'status': 'ok', 'operation': operation, 'file': relative,
              'start': start, 'end': end, 'sha256': sha(raw), 'total_lines': len(lines)}
    if operation == 'read':
        result['lines'] = [{'number': i, 'text': line.rstrip('\r\n')}
                           for i, line in enumerate(lines[start - 1:end], start)]
        return result
    if argv[4] != sha(raw):
        reject('STALE_FILE', 'File changed since read. Read again before choosing the edit.')
    replacement = argv[5]
    if not isinstance(replacement, str) or len(replacement.encode('utf-8')) > MAX_WINDOW:
        reject('REPLACEMENT_TOO_LARGE', 'NEW_TEXT must be a string of at most 24000 UTF-8 bytes.')
    if '\x00' in replacement:
        reject('BINARY_TEXT', 'NEW_TEXT must not contain NUL bytes.')
    if end < len(lines) and replacement and not replacement.endswith(('\n', '\r')):
        replacement += '\r\n' if selected.endswith('\r\n') else '\n'
    updated = (''.join(lines[:start - 1]) + replacement + ''.join(lines[end:])).encode('utf-8')
    if len(updated) > MAX_FILE:
        reject('FILE_TOO_LARGE', 'Result would exceed 2 MiB.')
    if updated == raw:
        reject('NO_CHANGE', 'Replacement would not change the file.')
    if path.suffix == '.py':
        try:
            compile(updated, relative, 'exec', dont_inherit=True)
        except (SyntaxError, ValueError) as error:
            reject('INVALID_PYTHON', str(error))
    diff_lines = difflib.unified_diff(text.splitlines(keepends=True),
                                     updated.decode('utf-8').splitlines(keepends=True),
                                     fromfile='a/' + relative, tofile='b/' + relative)
    diff = ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n'
                   for line in diff_lines)
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.edit-recovery-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        checked_path, _ = target_path(root, argv[1], True)
        if checked_path != path or path.read_bytes() != raw:
            reject('STALE_FILE', 'File changed before writing. Read again.')
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    result.update(previous_sha256=sha(raw), sha256=sha(updated), diff=diff,
                  python_syntax_checked=path.suffix == '.py')
    return result


def main():
    try:
        result = perform(sys.argv[1:])
    except Rejected as error:
        result = {'status': 'error', 'error_code': error.code, 'message': str(error)}
    except (OSError, UnicodeError) as error:
        result = {'status': 'error', 'error_code': type(error).__name__, 'message': str(error)}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
