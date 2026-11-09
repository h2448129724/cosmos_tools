#!/usr/bin/env python3
"""只读导出 SQLite 数据库，每张表对应一个 Excel 工作表。

用法（在项目根目录执行）：
    conda run --no-capture-output -n onnx-gpu python scripts/export_db_to_excel.py
    conda run --no-capture-output -n onnx-gpu python scripts/export_db_to_excel.py --output local/cosmos.xlsx
    conda run --no-capture-output -n onnx-gpu python scripts/export_db_to_excel.py --tables inspection_result ng_error

依赖：openpyxl。默认数据库为项目根目录 cosmos.db，输出为数据库旁的同名 .xlsx。
保持原始字段及关联 ID；JSON、日期和图片路径保持文本，不嵌入图片。
超长文本拆为相邻列（原字段、原字段__part2、...），按顺序拼接即可还原。
超过 15 位的整数按文本保存，BLOB 按带 0x 前缀的十六进制文本保存。
NULL 和空字符串在 Excel 中均显示为空白。遇到非法 XML 字符或超出 Excel
行列限制时明确报错，不静默丢弃数据。已有输出需用 --overwrite 显式覆盖。
"""

import argparse
import math
import os
import re
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from .paths import COSMOS_ROOT as PROJECT_ROOT
MAX_ROWS = 1_048_576
MAX_COLUMNS = 16_384
MAX_TEXT_UNITS = 32_767


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


def cell_parts(value):
    """转换为 Excel 可保存的值；按 UTF-16 单元拆文本，兼容 emoji。"""
    if isinstance(value, bytes):
        value = '0x' + value.hex()
    elif isinstance(value, int) and len(str(abs(value))) > 15:
        value = str(value)
    elif isinstance(value, float) and not math.isfinite(value):
        value = str(value)
    if not isinstance(value, str):
        return [value]
    if ILLEGAL_CHARACTERS_RE.search(value) or '\ufffe' in value or '\uffff' in value:
        raise ValueError('文本包含 Excel/XML 不支持的控制字符')
    if len(value.encode('utf-16-le')) <= MAX_TEXT_UNITS * 2:
        return [value]
    parts, start, units = [], 0, 0
    for index, char in enumerate(value):
        char_units = 2 if ord(char) > 0xFFFF else 1
        if units + char_units > MAX_TEXT_UNITS:
            parts.append(value[start:index])
            start, units = index, 0
        units += char_units
    parts.append(value[start:])
    return parts


def excel_cell(sheet, value, header=False):
    cell = WriteOnlyCell(sheet, value=value)
    if isinstance(value, str):
        # 数据库文本即使以 '=' 开头，也不能被解释为 Excel 公式。
        cell.data_type = 's'
    if header:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='245A81')
    return cell


def unique_sheet_name(table, used):
    base = re.sub(r'[\\/*?:\[\]\x00-\x1f]', '_', table).strip("'") or 'Table'
    if base.lower() == 'history':
        base = 'history_table'
    name, index = base[:31], 2
    while name.lower() in used:
        suffix = f'_{index}'
        name = base[: 31 - len(suffix)] + suffix
        index += 1
    used.add(name.lower())
    return name


def expanded_headers(columns, widths):
    headers, used = [], set(columns)
    for column, width in zip(columns, widths):
        headers.append(column)
        for part in range(2, width + 1):
            name = f'{column}__part{part}'
            while name in used:
                name += '_'
            used.add(name)
            headers.append(name)
    return headers


def export_database(database, output, tables=None, overwrite=False, *, on_progress=None):
    database = Path(database).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if not database.is_file():
        raise ValueError(f'数据库路径不是文件：{database}')
    if output.suffix.lower() != '.xlsx':
        raise ValueError('输出文件扩展名必须为 .xlsx')
    if output == database:
        raise ValueError('输出文件不能与数据库相同')
    if output.exists() and not overwrite:
        raise FileExistsError(f'输出已存在：{output}；如需覆盖，请加 --overwrite')

    workbook = Workbook(write_only=True)
    reports, used_names = [], set()
    # 不导入生产数据库模块，不建表、不修改 journal_mode；读取包含 WAL 的一致快照。
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=10)) as connection:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        available = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND substr(name, 1, 7) != 'sqlite_' ORDER BY name"
            )
        ]
        selected = list(dict.fromkeys(tables)) if tables else available
        missing = set(selected) - set(available)
        if missing:
            raise ValueError('数据库中没有这些表：' + ', '.join(sorted(missing)))
        if not selected:
            raise ValueError('数据库中没有可导出的用户表')

        # 先检查全部表，确定长文本需要的列数；此阶段失败不会留下半成品工作表。
        plans = []
        for table in selected:
            if on_progress:
                on_progress({'message': f'检查表：{table}'})
            query = f'SELECT * FROM {quote_identifier(table)}'
            cursor = connection.execute(query)
            columns = [item[0] for item in cursor.description]
            widths, count = [1] * len(columns), 0
            for count, row in enumerate(cursor, start=1):
                if count >= MAX_ROWS:
                    raise ValueError(f'表 {table} 超过 Excel 单表限制（最多 {MAX_ROWS - 1} 条数据）')
                for index, value in enumerate(row):
                    try:
                        widths[index] = max(widths[index], len(cell_parts(value)))
                    except (ValueError, UnicodeError) as error:
                        raise ValueError(f'表 {table} 第 {count} 条记录，字段 {columns[index]}：{error}') from error
            headers = expanded_headers(columns, widths)
            if len(headers) > MAX_COLUMNS:
                raise ValueError(f'表 {table} 拆分长文本后超过 Excel 的 {MAX_COLUMNS} 列限制')
            for header in headers:
                if len(cell_parts(header)) != 1:
                    raise ValueError(f'表 {table} 的字段名超过 Excel 单元格长度限制')
            plans.append((table, query, columns, widths, headers, count))

        for table, query, columns, widths, headers, count in plans:
            if on_progress:
                on_progress({'message': f'导出表：{table}（{count} 条）'})
            sheet = workbook.create_sheet(unique_sheet_name(table, used_names))
            sheet.freeze_panes = 'A2'
            for index, header in enumerate(headers, start=1):
                sheet.column_dimensions[get_column_letter(index)].width = min(48, max(16, len(header) + 3))
            sheet.append([excel_cell(sheet, header, header=True) for header in headers])
            for row in connection.execute(query):
                values = []
                for value, width in zip(row, widths):
                    parts = cell_parts(value)
                    values.extend(parts + [None] * (width - len(parts)))
                sheet.append([excel_cell(sheet, value) for value in values])
            sheet.auto_filter.ref = f'A1:{get_column_letter(len(headers))}{count + 1}'
            reports.append((table, sheet.title, count, len(headers) - len(columns)))
        connection.rollback()

    output.parent.mkdir(parents=True, exist_ok=True)
    # 先保存临时文件，成功后再替换目标，避免失败时破坏已有 Excel。
    descriptor, temporary_name = tempfile.mkstemp(prefix='.cosmos-export-', suffix='.xlsx', dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if on_progress:
            on_progress({'message': '正在保存 Excel…'})
        workbook.save(temporary)
        if overwrite:
            os.replace(temporary, output)
        else:
            # 硬链接原子创建目标；即使另一进程刚创建了同名文件，也不会覆盖。
            os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        workbook.close()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', type=Path, default=PROJECT_ROOT / 'cosmos.db', help='SQLite 数据库路径')
    parser.add_argument('-o', '--output', type=Path, help='输出 .xlsx 路径，默认与数据库同名且同目录')
    parser.add_argument('--tables', nargs='+', metavar='TABLE', help='只导出指定表；默认导出所有用户表')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已有输出文件')
    args = parser.parse_args()
    output = args.output or args.db.with_suffix('.xlsx')
    try:
        reports = export_database(args.db, output, args.tables, args.overwrite)
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f'导出失败：{error}', file=sys.stderr)
        return 1
    for table, sheet, count, extra in reports:
        note = f'（长文本增加 {extra} 列）' if extra else ''
        print(f'{table} -> {sheet}：{count} 条{note}')
    print(f'导出完成：{output.expanduser().resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
