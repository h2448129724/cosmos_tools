"""Persistent field batch execution, independent of Qt and GPU runtimes."""
from __future__ import annotations

import hashlib
from pathlib import Path


def configured_product(path):
    import yaml
    document = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    inspection = document.get('inspection', {}) if isinstance(document, dict) else {}
    product = inspection.get('product') if isinstance(inspection, dict) else None
    if product not in {'D01-L', 'D01-R'}:
        raise ValueError('配置文件 inspection.product 必须是 D01-L 或 D01-R')
    return product


def run_batch(request, environment, control, on_progress, run_one):
    from .field_history import new_batch, workspace_for, config_hash, read_run, validate_options
    from .field_dataset import _json
    jobs = request.get('jobs')
    if jobs is None:
        base = dict(request)
        sources = base.pop('sources')
        if base.get('product_config'):
            base['product'] = configured_product(base['product_config'])
        jobs = []
        for source in sources:
            options = {**base, 'source': source}
            if len(sources) > 1:
                suffix = hashlib.sha256(str(Path(source).resolve()).casefold().encode()).hexdigest()[:12]
                options['output'] = str(Path(base['output']) / f'{Path(source).name}_{suffix}')
            jobs.append({'options': options, 'environment': environment})
    path, batch = new_batch(request['output'], jobs)
    on_progress({'message': f'批次清单已保存（包含尚未开始的文件夹）：{path}'})
    results, errors = [], []
    for index, job in enumerate(batch['jobs']):
        while control.paused.is_set() and not control.stopped.wait(.1):
            pass
        if control.stopped.is_set():
            break
        options = dict(job['options'])
        source = options['source']
        on_progress({'message': f'文件夹 {index + 1}/{len(batch["jobs"])}：{source}'})
        job['status'] = 'running'
        job['known_runs'] = [p.name for p in (workspace_for(options) / 'runs').glob('*.json')]
        _json(path, batch)
        try:
            if config_hash(options) != job['config_hash']:
                raise ValueError('配置文件与批次保存时不同，已拒绝此任务，旧数据保留')
            if options.get('resume_from'):
                validate_options(read_run(options['resume_from']), options)
            environment = job.get('environment')
            result = {'source': source, **run_one(options, environment)}
            if result.get('stopped'):
                control.stopped.set()
            results.append(result)
            job['result'] = result
            job['status'] = 'stopped' if result.get('stopped') or control.stopped.is_set() else 'complete'
            if result.get('failed_model_jobs') or result.get('export', {}).get('errors') or result.get('export', {}).get('preview_errors'):
                job['status'] = 'failed'
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'
            job.update(status='failed', error=message)
            errors.append({'source': source, 'error': message})
            on_progress({'message': f'文件夹失败：{source}：{message}；其余任务按各自设置继续。'})
        _json(path, batch)
    return {'batch': True, 'results': results, 'errors': errors,
            'stopped': control.stopped.is_set(), 'batch_manifest': str(path)}

