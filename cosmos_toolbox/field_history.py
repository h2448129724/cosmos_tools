"""Read-only historical run discovery and explicit resume compatibility checks."""
from __future__ import annotations

import json
import sqlite3
import hashlib
import copy
import time
from pathlib import Path

# Only audited control/UI-only migrations may bypass a core file hash change.
# Pairs contain exact source byte hashes; arbitrary future core edits stay blocked.
COMPATIBLE_CORE_PAIRS = {
    (old, new)
    for old in (
        # 420d641 and ad2dcf0, LF and CRLF checkouts.
        '815af4ab949d4d2f8e88ea3dbc3ac7ec08494d105d1ff51939930bebbd9c6ce9',
        'f6e17b029830f8b98922c4fab9a22a26f6ec1da91cd3cbcab608862f34c8afbb',
        '04fef46d0b89965a69a4f02882b4c006c5507ff796ed85fa029a299082b968bc',
        'e79eac35831e2c0e6d74fa8d0fe282ff9339f6948aec43cebe11439bd316be47',
    )
    for new in (
        '85e8208ca9492b94dea22826904fa8d5d56ac66615e896b87a6e5ed282e8acbf',
        '8e6cb69638880bd9e2ebb1fb138fbd6768777c01fa5f4660c97f61a15b758520',
    )
}
PRESENTATION_FILES = {'field_dataset_ui.py', 'field_dataset_runtime.py',
                      'field_dataset_export.py', 'field_history.py'}


def read_run(path):
    path = Path(path).resolve(strict=True)
    if path.parent.name != 'runs' or path.suffix != '.json':
        raise ValueError('请选择 runs 目录中的任务快照')
    data = json.loads(path.read_text(encoding='utf-8'))
    if str(data.get('run_id')) != path.stem or not data.get('source') or not data.get('version'):
        raise ValueError('任务快照缺少来源、版本或运行编号')
    workspace = path.parent.parent
    if not (workspace / 'run.db').is_file():
        raise ValueError('原工作区 run.db 不存在，不能续跑')
    options = dict(data.get('options') or {})
    options.update(source=data['source'], output=str(workspace.parent if data.get('ng_only') else workspace),
                   product=data['product'], selected=data['selected'], mode=data['mode'], ng_only=data.get('ng_only', False))
    options.setdefault('product_config', data.get('models', {}).get('product_config_path'))
    missing = [key for key in ('face', 'limit') if key not in options]
    options.setdefault('face', 'all')
    options.setdefault('limit', None)
    options.setdefault('export_scope', 'current')
    return {'path': str(path), 'workspace': str(workspace), 'data': data,
            'options': options, 'missing': missing}


def discover(root):
    root = Path(root).resolve()
    paths = set()
    # Single source, NG, or batch per-source workspaces; never walk image trees.
    for pattern in ('runs/*.json', '*/runs/*.json', '*/*/runs/*.json'):
        paths.update(root.glob(pattern))
    results, errors = [], []
    for path in sorted(paths, key=lambda p: p.stem, reverse=True):
        try:
            record = read_run(path)
            db_path = Path(record['workspace']) / 'run.db'
            with sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True) as db:
                counts = dict(db.execute('SELECT status,COUNT(*) FROM items JOIN run_items ON key=item_key '
                                         'WHERE run_id=? GROUP BY status', (record['data']['run_id'],)))
            record['counts'] = counts
            results.append(record)
        except Exception as exc:
            errors.append(f'{path}: {exc}')
    return results, errors


def validate_options(record, options):
    expected = record['options']
    for key in ('source', 'output', 'product_config'):
        left, right = expected.get(key), options.get(key)
        if left and right:
            left, right = str(Path(left).resolve()).casefold(), str(Path(right).resolve()).casefold()
        if left != right:
            raise ValueError(f'续跑设置不一致：{key}；请恢复历史设置，或明确选择新任务')
    for key in ('product', 'mode', 'ng_only'):
        if expected.get(key) != options.get(key):
            raise ValueError(f'续跑设置不一致：{key}')
    if sorted(expected['selected']) != sorted(options['selected']):
        raise ValueError('续跑模型选择不一致')
    for key in ('face', 'limit'):
        if key not in record['missing'] and expected.get(key) != options.get(key):
            raise ValueError(f'续跑范围不一致：{key}；请恢复历史设置')


def validate_snapshot(previous, current):
    old, new = dict(previous), dict(current)
    old_tools, new_tools = old.pop('tool_sources', {}), new.pop('tool_sources', {})
    if not old_tools or not new_tools:
        raise ValueError('旧任务缺少源码快照，无法证明兼容，未开始处理图片')
    if old != new:
        changed = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
        raise ValueError('模型/配置/算法快照不兼容：' + ', '.join(changed))
    for name in old_tools.keys() | new_tools.keys():
        if name in PRESENTATION_FILES or old_tools.get(name) == new_tools.get(name):
            continue
        if name == 'field_dataset.py' and (old_tools.get(name), new_tools.get(name)) in COMPATIBLE_CORE_PAIRS:
            continue
        raise ValueError(f'生成源码不兼容：{name}；旧任务保留，未开始处理图片')


def workspace_for(options):
    root = Path(options['output']).resolve()
    return root / 'ng_only' if options.get('ng_only') else root


def validate_jobs(jobs):
    seen = set()
    for job in jobs:
        key = str(workspace_for(job['options'])).casefold()
        if key in seen:
            raise ValueError('同一工作区只能选择一个历史任务；请取消重复运行记录或重叠批次')
        seen.add(key)
    if not jobs:
        raise ValueError('没有可恢复的任务')


def jobs_from_records(records, environment):
    jobs = []
    for record in records:
        if record.get('kind') == 'batch':
            jobs.extend(read_batch(record['path'])['jobs'])
        else:
            jobs.append({'options': {**record['options'], 'resume_from': record['path']},
                         'environment': record['data'].get('environment') or environment,
                         'status': 'pending'})
    validate_jobs(jobs)
    return jobs


def config_hash(options):
    path = options.get('product_config')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None


def new_batch(output, jobs):
    from .field_dataset import _json
    validate_jobs(jobs)
    path = Path(output).resolve() / 'batches' / f'{time.time_ns()}.json'
    document = {'schema': 1, 'jobs': copy.deepcopy(jobs)}
    for job in document['jobs']:
        job.setdefault('config_hash', config_hash(job['options']))
        job['status'] = 'pending'
        job.pop('error', None)
        job.pop('result', None)
        job.pop('known_runs', None)
    _json(path, document)
    return path, document


def read_batch(path):
    path = Path(path).resolve(strict=True)
    document = json.loads(path.read_text(encoding='utf-8'))
    if document.get('schema') != 1 or not isinstance(document.get('jobs'), list):
        raise ValueError('无效批次清单')
    jobs = copy.deepcopy(document['jobs'])
    for job in jobs:
        options = job['options']
        if job.get('result', {}).get('run_id'):
            options['resume_from'] = str(workspace_for(options) / 'runs' / (job['result']['run_id'] + '.json'))
        elif job.get('status') in ('running', 'failed', 'stopped') and 'known_runs' in job:
            # Recover a run created before a crash/result delivery failure. Never guess among unrelated runs.
            candidates = sorted((workspace_for(options) / 'runs').glob('*.json'))
            candidates = [p for p in candidates if p.name not in job['known_runs']]
            if len(candidates) > 1:
                raise ValueError(f"{options['source']} 有多个未关联运行，请改选单独历史记录")
            if candidates:
                record = read_run(candidates[0])
                validate_options(record, options)
                options['resume_from'] = str(candidates[0])
    validate_jobs(jobs)
    return {'kind': 'batch', 'path': str(path), 'jobs': jobs,
            'statuses': [job.get('status', 'pending') for job in jobs]}


def discover_batches(root):
    records, errors = [], []
    for path in sorted((Path(root) / 'batches').glob('*.json'), reverse=True):
        try:
            records.append(read_batch(path))
        except Exception as exc:
            errors.append(f'{path}: {exc}')
    return records, errors
