"""Freeze 99/30, stratifying repositories and keeping related tasks together."""
import argparse
from collections import Counter, defaultdict
import re
from pathlib import Path
from common import ROOT, digest, rows, read_json, write_json


def components(tasks):
    parent = list(range(len(tasks)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    keys = {}
    for i, task in enumerate(tasks):
        repo = task['repo'].lower()
        statement = re.sub(r'\s+', ' ', task['problem_statement']).strip().lower()
        tags = [('id', task['instance_id']), ('commit', task['base_commit']),
                ('statement', digest(statement))]
        tags += [('issue', x.lower()) for x in re.findall(
            r'https?://github.com/[^\s)]+/(?:issues|pull)/\d+', task['problem_statement'])]
        # Explicit task family, when supplied by a trusted public metadata file.
        if task.get('task_group'):
            tags.append(('family', task['task_group']))
        for kind, value in tags:
            if not value:
                continue
            key = (repo, kind, value)
            if key in keys:
                parent[find(i)] = find(keys[key])
            keys[key] = i
    groups = defaultdict(list)
    for i, task in enumerate(tasks):
        groups[find(i)].append(task)
    return sorted(groups.values(), key=lambda g: min(t['instance_id'] for t in g))


def make_split(tasks, seed=20261117, expected=129, validation=30):
    tasks = sorted(tasks, key=lambda t: t['instance_id'])
    if len(tasks) != expected or len({t['instance_id'] for t in tasks}) != expected:
        raise ValueError(f'Requires {expected} unique complete tasks, found {len(tasks)}')
    for t in tasks:
        if not all(isinstance(t.get(k), str) and t[k].strip() for k in
                   ['instance_id', 'repo', 'base_commit', 'problem_statement']):
            raise ValueError('Missing task identity/statement/commit')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', t['instance_id']):
            raise ValueError('Unsafe task ID')
    grouped = defaultdict(list)
    for group in components(tasks):
        grouped[group[0]['repo']].append(group)
    # Per-repository exact-size subset sums followed by global quota optimization.
    options = {}
    counts = Counter(t['repo'] for t in tasks)
    for repo, groups in sorted(grouped.items()):
        dp = {0: []}
        ordered = sorted(groups, key=lambda g: digest([seed, sorted(t['instance_id'] for t in g)]))
        for g in ordered:
            ids = sorted(t['instance_id'] for t in g)
            for n, chosen in sorted(list(dp.items()), reverse=True):
                if n + len(g) <= validation and n + len(g) not in dp:
                    dp[n + len(g)] = chosen + ids
        options[repo] = dp
    states = {0: (0.0, [])}
    for repo, choices in options.items():
        next_states = {}
        for old, (cost, ids) in states.items():
            for n, chosen in choices.items():
                total = old + n
                if total > validation:
                    continue
                score = cost + (n - validation * counts[repo] / expected) ** 2
                if total not in next_states or score < next_states[total][0]:
                    next_states[total] = (score, ids + chosen)
        states = next_states
    if validation not in states:
        raise ValueError('Exact 99/30 impossible without splitting related task groups; no split written')
    val = set(states[validation][1])
    public = [{k: t[k] for k in ['instance_id', 'repo', 'base_commit', 'problem_statement']}
              for t in tasks]
    result = {
        'schema': 1, 'seed': seed, 'source_sha256': digest(tasks),
        'public_identity_sha256': digest(public),
        'strategy': 'repository-stratified; same base commit, normalized statement, issue URL, or explicit family kept atomic',
        'train_ids': sorted(t['instance_id'] for t in tasks if t['instance_id'] not in val),
        'validation_ids': sorted(val),
        'task_groups': [sorted(t['instance_id'] for t in g) for g in components(tasks)],
        'repositories': {repo: {'total': n, 'validation': sum(t['instance_id'] in val for t in tasks if t['repo'] == repo)}
                         for repo, n in sorted(counts.items())},
        'independence_scope': 'disjoint training tasks/groups; not unseen repositories or proven historically unseen tasks',
    }
    result['freeze_sha256'] = digest(result)
    return result


def validate_split(manifest, tasks=None):
    value = dict(manifest)
    expected = value.pop('freeze_sha256')
    if digest(value) != expected:
        raise ValueError('Frozen split checksum changed')
    train, val = set(value['train_ids']), set(value['validation_ids'])
    if len(train) != 99 or len(val) != 30 or train & val:
        raise ValueError('Split must be disjoint 99/30')
    for group in value['task_groups']:
        if set(group) & train and set(group) & val:
            raise ValueError('Task group leaked across split')
    if tasks is not None and digest(sorted(tasks, key=lambda t: t['instance_id'])) != value['source_sha256']:
        raise ValueError('Dataset differs from frozen source')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/split.json')
    args = parser.parse_args()
    tasks = rows(args.tasks)
    if args.output.exists():
        validate_split(read_json(args.output), tasks)
        print('Existing frozen split verified; unchanged')
        return
    value = make_split(tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids accidentally overwriting a frozen split.
    import json
    with args.output.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    print('Frozen 99 train candidates / 30 validation tasks', value['freeze_sha256'])


if __name__ == '__main__':
    main()
