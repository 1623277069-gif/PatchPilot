"""Controller-only preparation for separate Generator / Validator notebooks.

This is input packaging, not a sandbox and not permission to run inference.
Never upload the experiment directory: it contains official reference answers.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from common import ROOT, read_json, rows, write_json
from pipeline import sanitized
from split import validate_split


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit_archive(path):
    """Reject unsafe paths, links, history and known controller artifacts.

    Passing is only a structural check. It does not establish base-commit
    provenance, absence of embedded answer content, or runtime isolation.
    Official snapshots with .git must first be reconstructed from their pinned
    base tree on the controller. This checker accepts source-only archives;
    fresh Git initialization in isolated setup still needs implementation and
    an official Harness compatibility test. Never merely remove Git refs.
    """
    names = set()
    banned = {'.git', '.hg', '.svn', '.ipynb_checkpoints', '__pycache__',
              'tasks.jsonl', 'split.json', 'test_patch', 'test_patch.diff',
              'reference.patch', 'gold.patch', 'final_report.json'}
    with tarfile.open(path, 'r:*') as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if (not parts or member.name.startswith('/') or '..' in parts
                    or '\\' in member.name or member.name in names):
                raise ValueError('Unsafe or duplicate archive member')
            names.add(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError('Archive links/devices are not permitted')
            if any(p.lower() in banned or p.lower().endswith(('.ipynb', '.patch', '.diff'))
                   for p in parts):
                raise ValueError('Archive includes history or controller artifact')
    if not names:
        raise ValueError('Empty archive')
    return {'sha256': sha(path), 'members': len(names),
            'structural_check': 'passed', 'base_tree_provenance': 'NOT_VERIFIED',
            'content_review': 'NOT_PERFORMED', 'generation_authorized': False}


def prepare(tasks_path, split_path, task_id, output, snapshot=None):
    tasks = rows(tasks_path)
    split = read_json(split_path)
    validate_split(split, tasks)
    if task_id not in split['train_ids'] or task_id in split['validation_ids']:
        raise ValueError('Only one frozen training candidate may be staged')
    task = next(t for t in tasks if t['instance_id'] == task_id)
    review = audit_archive(snapshot) if snapshot else None
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'staging/generator'):
        raise ValueError('Output must be inside staging/generator')
    # Do not follow existing output symlinks or mix in old notebook history.
    if output.exists():
        raise FileExistsError('Keep staged inputs immutable; use a new directory')
    output.mkdir(parents=True)
    write_json(output / 'task.json', sanitized(task))
    if snapshot:
        shutil.copyfile(snapshot, output / 'candidate_snapshot.tgz')
    manifest = {'task_id': task_id, 'split_sha256': split['freeze_sha256'],
                'files': {p.name: sha(p) for p in output.iterdir()},
                'snapshot_review': review,
                'status': 'PENDING_REAL_GENERATOR_VALIDATOR_ISOLATION',
                'generation_authorized': False,
                'missing': ['pinned_base_tree_and_content_review', 'dependency_input_review',
                            'runtime_file_and_fd_isolation', 'process_isolation',
                            'git_and_notebook_history_isolation', 'network_isolation',
                            'one_way_patch_transfer_no_validator_feedback']}
    write_json(output / 'input_manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, default=ROOT / 'data/official/tasks.jsonl')
    p.add_argument('--split', type=Path, default=ROOT / 'data/split.json')
    p.add_argument('--task-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--snapshot', type=Path)
    args = p.parse_args()
    result = prepare(args.tasks, args.split, args.task_id, args.output, args.snapshot)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
