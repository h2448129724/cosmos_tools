"""Read-only SQLite NG-product selection and verified, resumable image copying."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from datetime import datetime


def scan(database, project='CAB-F', product='', start='', end='', exclude_misjudged=False,
         kinds=('raw',), old_root='', new_root=''):
    database = Path(database).expanduser().resolve()
    if not database.is_file():
        raise ValueError('数据库文件不存在')
    if not kinds or set(kinds) - {'raw', 'result'}:
        raise ValueError('请选择原图或结果图')
    if bool(old_root) != bool(new_root):
        raise ValueError('路径映射需要同时填写旧根目录和新根目录')
    dates = {}
    for key, value in [('start', start), ('end', end)]:
        if value:
            parsed = datetime.fromisoformat(value.strip())
            if parsed.tzinfo is not None:
                raise ValueError('时间请使用数据库本地时间，不带时区')
            dates[key] = parsed.isoformat(sep=' ')
    if len(dates) == 2 and dates['start'] >= dates['end']:
        raise ValueError('结束时间必须晚于开始时间（结束时间不包含）')
    filters, args = ['r.ok=0'], []
    for name, value in [('project', project), ('product', product)]:
        if value:
            filters.append(f'r.{name}=?')
            args.append(value)
    for name, operator in [('start', '>='), ('end', '<')]:
        if name in dates:
            filters.append(f'r.create_time{operator}?')
            args.append(dates[name])
    if exclude_misjudged:
        filters.append('r.is_misjudged=0')
    where = ' AND '.join(filters)
    db = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=5)
    try:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        records = [dict(r) for r in db.execute(
            f'SELECT r.id,r.product,r.project,r.product_id,r.create_time,r.is_misjudged '
            f'FROM inspection_result r WHERE {where} ORDER BY r.id', args)]
        rows = [dict(r) for r in db.execute(
            f'SELECT f.* FROM image_file f JOIN inspection_result r ON r.id=f.result_id WHERE {where} ORDER BY f.id', args)]
        errors = [dict(r) for r in db.execute(
            f'SELECT e.result_id,e.file_id,e.error_code FROM ng_error e '
            f'JOIN inspection_result r ON r.id=e.result_id WHERE {where}', args)]
    finally:
        db.close()
    records_by_id = {r['id']: r for r in records}
    reasons = {}
    for error in errors:
        reasons.setdefault(error['result_id'], []).append(error)
    files = {}
    for row in rows:
        for kind in dict.fromkeys(kinds):
            original = row.get(kind + '_image') or ''
            path = Path(original) if original else None
            if path is not None:
                if old_root and path.is_relative_to(Path(old_root)):
                    path = Path(new_root) / path.relative_to(Path(old_root))
                if not path.is_absolute():
                    path = database.parent / path
                path = path.resolve()
            key = (kind, os.path.normcase(str(path))) if path else (kind, f'missing-{row["id"]}')
            association = {'record': records_by_id[row['result_id']], 'file_id': row['id'],
                           'database_path': original, 'filename': row.get('filename', ''),
                           'ng_errors': reasons.get(row['result_id'], [])}
            if key in files:
                files[key]['associations'].append(association)
                continue
            stat = path.stat() if path and path.is_file() else None
            files[key] = {'source': str(path) if path else '', 'kind': kind,
                          'status': 'ready' if stat else 'missing', 'size': stat.st_size if stat else 0,
                          'mtime_ns': stat.st_mtime_ns if stat else None, 'associations': [association]}
    files = list(files.values())
    linked = {r['result_id'] for r in rows}
    missing_records = [r for r in records if r['id'] not in linked]
    summary = {'ng_records': len(records), 'image_records': len(rows),
               'ready_files': sum(f['status'] == 'ready' for f in files),
               'missing_files': sum(f['status'] == 'missing' for f in files),
               'bytes': sum(f['size'] for f in files), 'unlinked_errors': sum(not r['file_id'] for r in errors),
               'records_without_images': len(missing_records)}
    return {'database': str(database), 'records': records, 'files': files,
            'missing_records': missing_records, 'summary': summary,
            'filters': dict(project=project, product=product, start=start, end=end,
                            exclude_misjudged=exclude_misjudged, kinds=list(kinds), old_root=old_root, new_root=new_root),
            'scope': 'NG product associated images; not individual-image NG ground truth'}


def _checkpoint(control):
    if control:
        while control.paused.is_set() and not control.stopped.wait(.1):
            pass
        return not control.stopped.is_set()
    return True


def _hash(path, control=None):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            if not _checkpoint(control):
                raise InterruptedError('已停止')
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    # Only the tool's reports are replaced; never replace an existing image.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def copy_plan(plan, output, control=None, on_progress=None):
    output = Path(output).expanduser().resolve()
    sources = [Path(f['source']) for f in plan['files'] if f['source']]
    if any(output == p or output == p.parent or output in p.parents for p in sources):
        raise ValueError('输出目录不能包含源图片；请选择独立导出目录')
    output.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (output / '.ng_copy.lock').open('a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError('此输出目录已有复制任务')
    try:
        return _copy_locked(plan, output, control, on_progress)
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


def _copy_locked(plan, output, control, on_progress):
    marker = output / '.ng_copy_job.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')).get('tool') != 'db_ng_export':
            raise ValueError('输出目录标识不匹配，请选择新的目录')
    else:
        if any(p.name != '.ng_copy.lock' for p in output.iterdir()):
            raise ValueError('输出目录不是本工具的任务目录且非空，请选择空目录以保护现有文件')
        _json(marker, {'tool': 'db_ng_export', 'version': 1})
    journal = output / 'copy_ledger.jsonl'
    completed = {}
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            try:
                item = json.loads(line)
                completed[item['key']] = item
            except (ValueError, KeyError):
                continue  # Interrupted final line is not proof of completion.
    copied = skipped = 0
    results = []
    for index, item in enumerate(plan['files']):
        if not _checkpoint(control):
            break
        result = dict(item)
        temporary = None
        try:
            if item['status'] != 'ready':
                result.update(status='missing', error='数据库路径为空或文件不存在')
            else:
                source = Path(item['source'])
                stat = source.stat()
                if (stat.st_size, stat.st_mtime_ns) != (item['size'], item['mtime_ns']):
                    raise ValueError('扫描后源文件已变化，请重新扫描')
                key = hashlib.sha256((item['kind'] + '|' + os.path.normcase(str(source))).encode()).hexdigest()
                name = f'{source.stem}__{key[:12]}{source.suffix}'
                target = output / item['kind'] / name
                if output not in target.resolve().parents:
                    raise ValueError('目标路径超出输出目录')
                result.update(key=key, target=str(target))
                prior = completed.get(key)
                if target.exists():
                    source_hash = _hash(source, control)
                    target_hash = _hash(target, control)
                    if source_hash != target_hash or (prior and
                            (prior['size'] != item['size'] or prior['mtime_ns'] != item['mtime_ns'] or prior['sha256'] != source_hash)):
                        raise ValueError('目标文件已存在且无法确认一致，保留文件，不覆盖')
                    after = source.stat()
                    if (after.st_size, after.st_mtime_ns) != (item['size'], item['mtime_ns']):
                        raise ValueError('校验期间源文件变化，请重新扫描')
                    result.update(status='skipped', sha256=source_hash)
                    if not prior:
                        # Recover a crash after the atomic rename but before ledger append.
                        with journal.open('a', encoding='utf-8') as stream:
                            stream.write('\n' + json.dumps(result, ensure_ascii=False) + '\n')
                            stream.flush()
                            os.fsync(stream.fileno())
                        completed[key] = result
                    skipped += 1
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    checksum = hashlib.sha256()
                    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.partial', delete=False) as dest:
                        temporary = Path(dest.name)
                        with source.open('rb') as stream:
                            while chunk := stream.read(1024 * 1024):
                                if not _checkpoint(control):
                                    raise InterruptedError('已停止')
                                checksum.update(chunk)
                                dest.write(chunk)
                        dest.flush()
                        os.fsync(dest.fileno())
                    after = source.stat()
                    if (after.st_size, after.st_mtime_ns) != (item['size'], item['mtime_ns']):
                        raise ValueError('复制期间源文件变化')
                    digest = checksum.hexdigest()
                    if _hash(temporary, control) != digest:
                        raise ValueError('复制校验失败')
                    temporary.rename(target)  # Windows refuses to overwrite an existing target.
                    temporary = None
                    result.update(status='copied', sha256=digest)
                    with journal.open('a', encoding='utf-8') as stream:
                        stream.write('\n' + json.dumps(result, ensure_ascii=False) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    completed[key] = result
                    copied += 1
        except InterruptedError:
            result.update(status='stopped')
            results.append(result)
            break
        except Exception as exc:
            result.update(status='failed', error=str(exc))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        results.append(result)
        if on_progress:
            on_progress({'current': index + 1, 'total': len(plan['files']), 'copied': copied,
                         'skipped': skipped, 'source': item['source'], 'status': result['status']})
    report = {'output': str(output), 'copied': copied, 'skipped': skipped,
              'failed': sum(r['status'] == 'failed' for r in results),
              'missing': sum(r['status'] == 'missing' for r in results),
              'stopped': bool(control and control.stopped.is_set()), 'scan': plan['summary']}
    _json(output / 'summary.json', report)
    _json(output / 'selection.json', {k: v for k, v in plan.items() if k != 'files'})
    _json(output / 'records_without_images.json', plan['missing_records'])
    _json(output / 'missing.json', [f for f in plan['files'] if f['status'] == 'missing'])
    _json(output / 'failures.json', [r for r in results if r['status'] == 'failed'])
    manifest = output / 'manifest.jsonl.tmp'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results), encoding='utf-8')
    os.replace(manifest, output / 'manifest.jsonl')
    return report
