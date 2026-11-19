"""Inspect workspace changes and execute focused pytest tests with JSON outcomes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

SCRATCH_PREFIXES = ('repro', 'debug_', 'scratch_')
TEST_DIRS = {'test', 'tests', 'testing', '__snapshots__'}
SOURCE_SUFFIXES = {'.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.c', '.h', '.cpp',
                   '.hpp', '.rs', '.go', '.rb', '.php', '.cs', '.swift', '.kt',
                   '.scala', '.vue', '.svelte', '.html', '.css', '.sql'}
CONFIG_SUFFIXES = {'.toml', '.yaml', '.yml', '.json', '.ini', '.cfg'}
TEST_CONFIGS = {'pytest.ini', 'tox.ini', 'setup.cfg', 'pyproject.toml', 'conftest.py',
                'jest.config.js', 'vitest.config.ts'}
PLUGIN = '''import json
import os
from pathlib import Path
counts = {"collected": 0, "passed": 0, "failed": 0, "skipped": 0,
          "collection_errors": 0, "execution_errors": 0}
def pytest_collection_finish(session):
    counts["collected"] = len(session.items)
def pytest_collectreport(report):
    if report.failed:
        counts["collection_errors"] += 1
def pytest_runtest_logreport(report):
    if report.when == "call":
        counts[report.outcome] += 1
    elif report.failed:
        counts["execution_errors"] += 1
    elif report.when == "setup" and report.skipped:
        counts["skipped"] += 1
def pytest_sessionfinish(session, exitstatus):
    counts["exit_code"] = int(exitstatus)
    Path(os.environ["VERIFY_CHANGE_REPORT"]).write_text(json.dumps(counts))
'''


def git(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True)
    return result.stdout


def inspect(root):
    net = git(root, 'diff', '--name-only', '-z', 'HEAD').decode().split('\0')
    staged = git(root, 'diff', '--cached', '--name-only', '-z').decode().split('\0')
    unstaged = git(root, 'diff', '--name-only', '-z').decode().split('\0')
    tracked = net + staged + unstaged
    untracked = git(root, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
    changes = []
    fingerprint = hashlib.sha256(git(root, 'diff', '--binary', 'HEAD'))
    for name in sorted(set(tracked + untracked) - {''}):
        path = Path(name)
        if path.stem.startswith(SCRATCH_PREFIXES):
            kind = 'scratch'
        elif any(part in TEST_DIRS for part in path.parts) or path.name.startswith('test_') or path.stem.endswith('_test') or path.name == 'conftest.py':
            kind = 'test'
        elif path.suffix in SOURCE_SUFFIXES or path.suffix in CONFIG_SUFFIXES:
            kind = 'implementation_candidate'
        else:
            kind = 'other'
        current = root / path
        digest = None
        if name in untracked and current.is_symlink():
            digest = hashlib.sha256(os.readlink(current).encode()).hexdigest()
            fingerprint.update(name.encode() + b'\0' + digest.encode())
        elif name in untracked and current.is_file():
            digest = hashlib.sha256(current.read_bytes()).hexdigest()
            fingerprint.update(name.encode() + b'\0' + digest.encode())
        changes.append({'path': name, 'kind': kind, 'untracked': name in untracked,
                        'deleted': not current.exists(), 'staged': name in staged,
                        'unstaged': name in unstaged, 'net_changed': name in net or name in untracked})
    patch = git(root, 'diff', 'HEAD').decode(errors='replace')
    return {'fingerprint': fingerprint.hexdigest(), 'files': changes,
            'implementation_candidates': [c['path'] for c in changes if c['kind'] == 'implementation_candidate' and c['net_changed']],
            'classification_note': 'Filename heuristics; confirm intended implementation files from the issue.',
            'diff_note': 'Net working-tree change versus HEAD; staged/unstaged flags are reported separately.',
            'diff': patch[:6000], 'diff_truncated': len(patch) > 6000,
            'untracked_note': 'Untracked file contents are fingerprinted but are absent from git diff.'}


def test_path(root, filename):
    path = Path(filename)
    if path.is_absolute() and path.is_relative_to('/workspace'):
        path = root / path.relative_to('/workspace')
    elif not path.is_absolute():
        path = root / path
    path = path.resolve()
    temporary_test = path.is_relative_to(Path(tempfile.gettempdir()).resolve()) and path.name.startswith(('test_', 'repro'))
    if not (path.is_relative_to(root) or temporary_test) or not path.is_file():
        raise ValueError('Choose a focused workspace test or a /tmp/test_* reproduction file.')
    return path


def test_hashes(root, selectors):
    return {str(test_path(root, s.split('::')[0])): hashlib.sha256(test_path(root, s.split('::')[0]).read_bytes()).hexdigest() for s in selectors}


def run_tests(root, selectors, timeout):
    selected = ['::'.join([str(test_path(root, s.split('::')[0])), *s.split('::')[1:]]) for s in selectors]
    folder = Path(tempfile.mkdtemp(prefix='verify-change-'))
    (folder / '_verify_change_plugin.py').write_text(PLUGIN)
    report = folder / 'report.json'
    log = folder / 'pytest.log'
    env = dict(os.environ)
    env['VERIFY_CHANGE_REPORT'] = str(report)
    # A same-size repair within one filesystem clock tick must not reuse stale .pyc.
    env['PYTHONPYCACHEPREFIX'] = str(folder / 'pycache')
    env['PYTHONPATH'] = os.pathsep.join([str(folder), str(root / 'src'), str(root), env.get('PYTHONPATH', '')])
    command = [sys.executable, '-m', 'pytest', '-q', '-p', '_verify_change_plugin', *selected]
    started = time.monotonic()
    timed_out = False
    with log.open('wb') as handle:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            exit_code = None
    counts = json.loads(report.read_text()) if report.exists() else {}
    if timed_out:
        status = 'timeout'
    elif counts.get('collection_errors'):
        status = 'collection_error'
    elif exit_code == 5 or counts.get('collected') == 0:
        status = 'no_tests'
    elif not counts:
        status = 'runner_error'
    elif exit_code == 1:
        status = 'failed'
    elif exit_code == 0 and counts.get('collected', 0) > 0 and not counts.get('execution_errors'):
        status = 'passed' if counts.get('passed', 0) > 0 else 'all_skipped'
    else:
        status = 'runner_error'
    return {'status': status, 'exit_code': exit_code, 'counts': counts,
            'seconds': round(time.monotonic() - started, 3), 'selectors': selectors,
            'log_path': str(log), 'log_tail': log.read_text(errors='replace')[-4000:],
            'interpretation': 'Pytest outcomes do not establish that assertions cover the issue contract.'}


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('inspect')
    context = sub.add_parser('context')
    context.add_argument('symbols', nargs='+')
    tests = sub.add_parser('pytest')
    tests.add_argument('--timeout', type=int, default=30)
    tests.add_argument('selectors', nargs='+')
    args = parser.parse_args(argv)
    try:
        root = Path(os.environ.get('GEMMA_EDIT_WORKSPACE', '/workspace')).resolve()
        if not root.is_dir():
            raise ValueError('Workspace is unavailable.')
        root = Path(git(root, 'rev-parse', '--show-toplevel').decode().strip())
        identity = hashlib.sha256(str(root).encode() + git(root, 'rev-parse', 'HEAD')).hexdigest()[:24]
        storage = Path(tempfile.gettempdir()) / ('repair-acceptance-' + identity)
        storage.mkdir(mode=0o700, exist_ok=True)
        state_file = storage / 'state.json'
        state = json.loads(state_file.read_text()) if state_file.exists() else {'executions': 0}
        before = inspect(root)
        result = {'status': 'ok', 'skill': 'verify-change', 'mode': args.mode, 'changes': before}
        if args.mode == 'context':
            if not 1 <= len(args.symbols) <= 3 or any(not s or len(s) > 120 for s in args.symbols):
                raise ValueError('Choose one to three short literal symbols.')
            command = ['git', '-C', str(root), 'grep', '-n', '-I', '-F']
            for symbol in args.symbols:
                command += ['-e', symbol]
            command += ['--', '*.py', '*.js', '*.ts', '*.tsx', '*.go', '*.rs', '*.java', '*.rb', '*.cpp', '*.h']
            log = storage / 'search.log'
            with log.open('wb') as handle:
                searched = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=10)
            with log.open('rb') as handle:
                excerpt = handle.read(3501).decode(errors='replace')
            result['context'] = {'symbols': args.symbols, 'exit_code': searched.returncode,
                                 'excerpt': excerpt[:3500], 'truncated': log.stat().st_size > 3500,
                                 'full_log_path': str(log)}
        if args.mode == 'pytest':
            if not 1 <= args.timeout <= 30:
                raise ValueError('Acceptance timeout must be between 1 and 30 seconds.')
            previous = state.get('last', {})
            hashes = test_hashes(root, args.selectors)
            if previous and previous.get('test_file_hashes') != hashes:
                raise ValueError('Focused test files changed; do not weaken assertions between acceptance attempts.')
            if previous and previous.get('tested_fingerprint') == before['fingerprint'] and previous.get('test', {}).get('selectors') == args.selectors:
                result = previous
                result['reused_evidence'] = True
                print(json.dumps(result, ensure_ascii=False))
                return 0
            if state['executions'] >= 2:
                raise ValueError('Acceptance execution limit reached (2); no further retry.')
            if previous and previous.get('test', {}).get('selectors') != args.selectors:
                raise ValueError('Keep the same focused selectors; do not replace failing checks.')
            # Reserve before execution, including interrupted/invalid test attempts.
            state['executions'] += 1
            state_file.write_text(json.dumps(state))
            result['test'] = run_tests(root, args.selectors, args.timeout)
            result['test_file_hashes'] = hashes
            after = inspect(root)
            result['workspace_changed_during_test'] = before['fingerprint'] != after['fingerprint']
            result['changes'] = after
            result['tested_fingerprint'] = before['fingerprint']
            changed = [c for c in after['files'] if c['net_changed']]
            protected = [c['path'] for c in changed if not c['untracked'] and
                         (c['kind'] == 'test' or Path(c['path']).name in TEST_CONFIGS)]
            scratch = [c['path'] for c in changed if c['kind'] == 'scratch' or
                       (c['untracked'] and c['kind'] == 'test')]
            reasons = []
            if not after['implementation_candidates']:
                reasons.append('No implementation/configuration change; reproduction alone is not a repair.')
            if protected:
                reasons.append('Existing tests/test configuration changed: ' + ', '.join(protected))
            if scratch:
                reasons.append('Temporary reproduction/test files remain in patch: ' + ', '.join(scratch))
            if result['test']['status'] != 'passed':
                reasons.append('Focused tests did not pass: ' + result['test']['status'])
            if result['workspace_changed_during_test']:
                reasons.append('Workspace changed during tests; evidence is stale.')
            if test_hashes(root, args.selectors) != hashes:
                reasons.append('Test file changed during execution; evidence is stale.')
            result['acceptance'] = {'accepted': not reasons, 'reasons': reasons,
                                    'executions': state['executions'], 'remaining_executions': 2 - state['executions'],
                                    'scope': 'Mechanical acceptance only; assertion coverage and issue semantics require review.'}
            # Store the full diff separately; return only bounded evidence to the model.
            full = storage / ('acceptance-%d.json' % state['executions'])
            result['full_report_path'] = str(full)
            full.write_text(json.dumps(result, ensure_ascii=False))
            diff_length = len(after['diff'])
            result['changes']['diff'] = result['changes']['diff'][:2500]
            result['changes']['diff_truncated'] = result['changes']['diff_truncated'] or diff_length > 2500
            result['test']['log_tail'] = result['test']['log_tail'][-2200:]
            state['last'] = result
            state_file.write_text(json.dumps(state, ensure_ascii=False))
        elif args.mode == 'inspect':
            previous = state.get('last', {})
            fresh = previous.get('tested_fingerprint') == before['fingerprint']
            if fresh:
                try:
                    fresh = test_hashes(root, previous['test']['selectors']) == previous['test_file_hashes']
                except (KeyError, ValueError):
                    fresh = False
            result['acceptance'] = {'accepted': bool(fresh and previous.get('acceptance', {}).get('accepted')),
                                    'fresh': fresh, 'executions': state['executions'],
                                    'reasons': [] if fresh else ['No current matching test evidence.']}
            diff_length = len(before['diff'])
            result['changes']['diff'] = before['diff'][:2500]
            result['changes']['diff_truncated'] = before['diff_truncated'] or diff_length > 2500
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result = {'status': 'error', 'skill': 'verify-change', 'message': str(error)}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
