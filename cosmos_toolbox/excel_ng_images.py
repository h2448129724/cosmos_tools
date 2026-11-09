#!/usr/bin/env python3
"""从 Cosmos 导出的 Excel 中提取 NG 原始图，复制到指定目录。

在可以访问 E 盘图片的电脑上执行（依赖 openpyxl）：
    conda activate onnx-gpu
    python scripts/copy_ng_images.py --excel "C:/Users/tianwei/Desktop/cosmos.xlsx" --output "D:/NG_recheck"

默认：inspection_result.ok 为 0 的记录 -> image_file.result_id -> filename 中的 E 盘原图。
filename 没有指定盘符的路径时，再使用 raw_image；绝不使用 result_image。
每条 NG 记录的全部图都会提取，包含正反面，误判标记不影响 NG 筛选。
--only-ng-faces 仅提取 ng_error.file_id 指向、且所属记录为 NG 的图。
--start-date / --end-date 按 inspection_result.create_time 筛选检测时间；可只填一端。
日期格式：2026-09-01；时间格式："2026-09-01 08:30:00"（可带微秒）。
开始和结束均包含边界；只填结束日期时包含当天全部记录。不填则不限制时间。
--source-root 可将原 E 盘根目录映射到另一个目录（保留 E 盘下的相对路径）。
--dry-run 只检查和统计，不复制、不创建目标目录；可用 --report 保存检查清单。
默认平铺到目标目录，保持原文件名。同名不同来源会按来源目录加稳定前缀，
同一目录的正反面使用相同前缀。相同来源只复制一次；已有文件内容一致则跳过，
内容不一致则报告冲突，绝不覆盖。实际执行会在目标目录生成 UTF-8 CSV 清单。
退出码：0=无缺失/冲突/失败；1=存在未取出的图或输入错误。
"""

import argparse
import csv
import filecmp
import hashlib
import ntpath
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time
from pathlib import Path, PureWindowsPath

from openpyxl import load_workbook

REPORT_FIELDS = (
    'result_id',
    'file_id',
    'project',
    'product',
    'product_id',
    'create_time',
    'path_field',
    'original_path',
    'source_path',
    'destination_path',
    'status',
    'message',
)
FAILED_STATUSES = {'missing', 'conflict', 'copy_error', 'no_source_path', 'no_image_row'}


def table_rows(book, name, required):
    if name not in book.sheetnames:
        raise ValueError(f'Excel 缺少工作表：{name}')
    rows = book[name].iter_rows(values_only=True)
    headers = next(rows, None)
    if not headers or not set(required).issubset(headers):
        raise ValueError(f'工作表 {name} 必须包含字段：{", ".join(required)}')
    if len(set(headers)) != len(headers):
        raise ValueError(f'工作表 {name} 包含重复字段名')
    for number, values in enumerate(rows, start=2):
        if any(value is not None for value in values):
            yield number, dict(zip(headers, values))


def record_id(value):
    if isinstance(value, bool):
        raise ValueError(f'记录 ID 无效：{value!r}')
    if isinstance(value, float):
        if not value.is_integer() or abs(value) > 2**53 - 1:
            raise ValueError(f'记录 ID 无效或已失去精度：{value!r}')
        value = int(value)
    text = str(value).strip()
    if not re.fullmatch(r'\d+', text):
        raise ValueError(f'记录 ID 无效：{value!r}')
    return str(int(text))


def is_ng(value):
    text = str(value).strip().lower()
    if text in {'0', '0.0', 'false', 'ng'}:
        return True
    if text in {'1', '1.0', 'true', 'ok'}:
        return False
    raise ValueError(f'ok 字段无法识别：{value!r}；应为 0/1、False/True 或 NG/OK')


def parse_datetime(value, label, end_of_day=False):
    """解析本地检测时间；结束日期扩展到当天最后一微秒。"""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.max if end_of_day else time.min)
    elif isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?)?', text):
            raise ValueError(f'{label} 无效：{value!r}；请使用 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS')
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as error:
            raise ValueError(f'{label} 无效：{value!r}') from error
        if len(text) == 10 and end_of_day:
            parsed = datetime.combine(parsed.date(), time.max)
    else:
        raise ValueError(f'{label} 缺失或不是有效的日期时间：{value!r}')
    if parsed.tzinfo is not None:
        raise ValueError(f'{label} 请使用不带时区的本地检测时间')
    return parsed


def source_path(row, source_drive, source_root):
    for field in ('filename', 'raw_image'):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = ntpath.normpath(value.strip())
        original = PureWindowsPath(normalized)
        if not original.is_absolute() or original.drive.upper() != source_drive:
            continue
        if original.name in {'', '.', '..'}:
            continue
        actual = Path(source_root).joinpath(*original.parts[1:]) if source_root else Path(normalized)
        return field, normalized, actual
    return '', '', None


def collect_images(
    excel, output, source_drive='E:', source_root=None, only_ng_faces=False, start_date=None, end_date=None
):
    start = parse_datetime(start_date, '开始日期') if start_date is not None else None
    end = parse_datetime(end_date, '结束日期', end_of_day=True) if end_date is not None else None
    if start is not None and end is not None and start > end:
        raise ValueError('开始日期/时间不能晚于结束日期/时间')
    source_drive = source_drive.upper().rstrip('\\/')
    if not re.fullmatch(r'[A-Z]:', source_drive):
        raise ValueError('--source-drive 必须是盘符，例如 E:')
    output = Path(output).expanduser().resolve()
    if source_root is not None:
        source_root = Path(source_root).expanduser().resolve()
    book = load_workbook(excel, read_only=True, data_only=False)
    try:
        ng_results, all_ids = {}, set()
        required = ('id', 'ok', 'create_time') if start is not None or end is not None else ('id', 'ok')
        for number, row in table_rows(book, 'inspection_result', required):
            result_id = record_id(row['id'])
            if result_id in all_ids:
                raise ValueError(f'inspection_result 第 {number} 行的 ID 重复：{result_id}')
            all_ids.add(result_id)
            if not is_ng(row['ok']):
                continue
            if start is not None or end is not None:
                created = parse_datetime(row['create_time'], f'inspection_result 第 {number} 行的 create_time')
                if (start is not None and created < start) or (end is not None and created > end):
                    continue
            ng_results[result_id] = row
        error_files = set()
        if only_ng_faces:
            for _, row in table_rows(book, 'ng_error', ('result_id', 'file_id')):
                error_files.add((record_id(row['result_id']), record_id(row['file_id'])))

        records, associated, file_ids = [], set(), set()
        for number, row in table_rows(book, 'image_file', ('id', 'result_id', 'filename')):
            result_id = record_id(row['result_id'])
            if result_id not in ng_results:
                continue
            file_id = record_id(row['id'])
            if file_id in file_ids:
                raise ValueError(f'image_file 第 {number} 行的 ID 重复：{file_id}')
            file_ids.add(file_id)
            if only_ng_faces and (result_id, file_id) not in error_files:
                continue
            associated.add(result_id)
            info = ng_results[result_id]
            field, original, actual = source_path(row, source_drive, source_root)
            records.append(
                {
                    'result_id': result_id,
                    'file_id': file_id,
                    **{key: info.get(key, '') for key in ('project', 'product', 'product_id', 'create_time')},
                    'path_field': field,
                    'original_path': original,
                    'source_path': str(actual) if actual else '',
                    'destination_path': '',
                    'status': 'pending' if actual else 'no_source_path',
                    'message': '' if actual else f'filename/raw_image 中没有 {source_drive} 的绝对路径',
                }
            )
        for result_id in ng_results.keys() - associated:
            info = ng_results[result_id]
            records.append(
                {
                    'result_id': result_id,
                    'file_id': '',
                    **{key: info.get(key, '') for key in ('project', 'product', 'product_id', 'create_time')},
                    'path_field': '',
                    'original_path': '',
                    'source_path': '',
                    'destination_path': '',
                    'status': 'no_image_row',
                    'message': '没有关联的 NG 错误图片行' if only_ng_faces else '没有关联的 image_file 行',
                }
            )
    finally:
        book.close()

    # 对同名异源图片所在目录统一加前缀，保证同目录的 top/bottom 仍能配对。
    name_sources = defaultdict(set)
    for row in records:
        if row['original_path']:
            name_sources[ntpath.basename(row['original_path']).lower()].add(ntpath.normcase(row['original_path']))
    prefixed_parents = {ntpath.dirname(path) for paths in name_sources.values() if len(paths) > 1 for path in paths}
    destinations = {}
    for row in records:
        if not row['original_path']:
            continue
        key = ntpath.normcase(row['original_path'])
        name = ntpath.basename(row['original_path'])
        parent = ntpath.dirname(key)
        if parent in prefixed_parents:
            name = hashlib.sha256(parent.encode('utf-8')).hexdigest()[:12] + '__' + name
        destination = output / name
        destination_key = str(destination).lower()
        if destination_key in destinations and destinations[destination_key] != key:
            raise ValueError(f'目标文件名发生冲突：{destination}')
        destinations[destination_key] = key
        row['destination_path'] = str(destination)
    return records, len(ng_results)


def copy_one(source, destination, dry_run):
    if not source.is_file():
        return 'missing', '源图片不存在或不是文件'
    if destination.exists():
        if destination.is_file() and filecmp.cmp(source, destination, shallow=False):
            return 'existing_same', '目标已有相同内容，跳过'
        return 'conflict', '目标已有不同内容，不覆盖'
    if dry_run:
        return 'ready', '可复制'
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with source.open('rb') as reader, destination.open('xb') as writer:
            created = True
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        shutil.copystat(source, destination)
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return 'copied', ''


def process_images(records, dry_run=False, *, on_progress=None, should_stop=None):
    completed = {}
    for index, row in enumerate(records, start=1):
        if should_stop and should_stop():
            break
        if on_progress and (index == 1 or index % 100 == 0 or index == len(records)):
            on_progress({'current': index - 1, 'total': len(records),
                         'message': f'已检查/处理 {index - 1}/{len(records)} 条图片记录'})
        if row['status'] != 'pending':
            continue
        key = ntpath.normcase(row['original_path'])
        if key in completed:
            status, message = completed[key]
            row['status'] = 'duplicate' if status in {'copied', 'existing_same', 'ready'} else status
            row['message'] = '相同来源已处理' if row['status'] == 'duplicate' else message
            continue
        try:
            status, message = copy_one(Path(row['source_path']), Path(row['destination_path']), dry_run)
        except OSError as error:
            status, message = 'copy_error', str(error)
        row['status'], row['message'] = status, message
        completed[key] = status, message
        if index % 100 == 0 and on_progress is None:
            print(f'已检查/处理 {index}/{len(records)} 条图片记录', flush=True)
    return Counter(row['status'] for row in records)


def write_report(records, report):
    report = Path(report).expanduser().resolve()
    if report.suffix.lower() != '.csv':
        raise ValueError('清单文件必须为 .csv')
    if any(str(report).lower() in {row['source_path'].lower(), row['destination_path'].lower()} for row in records):
        raise ValueError('清单路径不能与源图或目标图片相同')
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--excel', type=Path, default=Path.home() / 'Desktop' / 'cosmos.xlsx', help='导出的 Excel 路径')
    parser.add_argument('-o', '--output', type=Path, required=True, help='保存 NG 原图的目标文件夹')
    parser.add_argument('--source-drive', default='E:', help='原图盘符，默认 E:')
    parser.add_argument('--source-root', type=Path, help='原盘符根目录在本机的新位置，例如 F:/ 或 X:/machine_E')
    parser.add_argument('--only-ng-faces', action='store_true', help='只取 ng_error 指向的图，不取同件产品的其他面')
    parser.add_argument('--start-date', help='检测开始日期/时间（包含），如 2026-09-01 或 "2026-09-01 08:30:00"')
    parser.add_argument('--end-date', help='检测结束日期/时间（包含）；只填日期时包含当天全部记录')
    parser.add_argument('--dry-run', action='store_true', help='仅检查，不复制，不创建目标目录')
    parser.add_argument('--report', type=Path, help='自定义 CSV 清单路径（不能是已有文件）')
    args = parser.parse_args()
    try:
        records, ng_count = collect_images(
            args.excel,
            args.output,
            args.source_drive,
            args.source_root,
            args.only_ng_faces,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        report = args.report
        if not args.dry_run and report is None:
            report = args.output / ('_copy_ng_report_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
        if report and (report.suffix.lower() != '.csv' or report.exists()):
            raise ValueError('清单路径必须是尚不存在的 .csv 文件')
        if args.start_date is not None or args.end_date is not None:
            print(f'检测时间范围（含边界）：{args.start_date or "不限"} 至 {args.end_date or "不限"}', flush=True)
        print(f'NG 检测记录：{ng_count}；关联图片行：{sum(bool(row["file_id"]) for row in records)}', flush=True)
        summary = process_images(records, args.dry_run)
        if report:
            write_report(records, report)
            print(f'清单：{report.expanduser().resolve()}')
        labels = {
            'copied': '复制成功',
            'ready': '可复制',
            'existing_same': '已存在且相同',
            'duplicate': '重复来源',
            'missing': '源图缺失',
            'conflict': '同名冲突',
            'copy_error': '复制/访问失败',
            'no_source_path': '没有指定盘符原图路径',
            'no_image_row': '没有关联图片行',
        }
        print('；'.join(f'{labels[status]}：{count}' for status, count in summary.items()) or '没有需要提取的图片')
        if not args.dry_run:
            print(f'图片目录：{args.output.expanduser().resolve()}')
        return 1 if any(summary[status] for status in FAILED_STATUSES) else 0
    except (OSError, ValueError, KeyError) as error:
        print(f'提取失败：{error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
