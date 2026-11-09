"""验证导出后的数据保真、WAL 快照及输出保护。"""

import sqlite3

import pytest
from openpyxl import load_workbook

from cosmos_toolbox.database_excel import export_database


def test_export_preserves_long_text_numbers_and_literal_formulas(tmp_path):
    database, output = tmp_path / 'sample.db', tmp_path / 'sample.xlsx'
    long_text = '{"内容":"' + '中文😀' * 12_000 + '"}'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE detail (id INTEGER, payload TEXT, formula TEXT, blob BLOB, nullable TEXT)')
        db.execute(
            'INSERT INTO detail VALUES (?, ?, ?, ?, ?)', (1234567890123456789, long_text, '=1+1', b'\x00\xff', None)
        )
        db.execute('CREATE TABLE empty_table (id INTEGER, name TEXT)')
    reports = export_database(database, output)
    assert {table: count for table, _, count, _ in reports} == {'detail': 1, 'empty_table': 0}
    book = load_workbook(output)
    sheet = book['detail']
    headers = [cell.value for cell in sheet[1]]
    row = dict(zip(headers, [cell.value for cell in sheet[2]]))
    assert row['id'] == '1234567890123456789'
    assert ''.join(row[h] for h in headers if h.startswith('payload')) == long_text
    assert row['formula'] == '=1+1'
    assert sheet.cell(2, headers.index('formula') + 1).data_type == 's'
    assert row['blob'] == '0x00ff'
    assert row['nullable'] is None
    assert sheet.freeze_panes == 'A2'
    assert sheet.auto_filter.ref.endswith('2')
    assert book['empty_table'].max_row == 1
    book.close()


def test_export_reads_committed_wal_and_does_not_write_source(tmp_path):
    database, output = tmp_path / 'wal.db', tmp_path / 'wal.xlsx'
    with sqlite3.connect(database) as writer:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('CREATE TABLE records (id INTEGER)')
        writer.execute('INSERT INTO records VALUES (7)')
        writer.commit()
        before = database.read_bytes()
        wal_before = database.with_name(database.name + '-wal').read_bytes()
        export_database(database, output, tables=['records'])
        assert database.read_bytes() == before
        assert database.with_name(database.name + '-wal').read_bytes() == wal_before
        book = load_workbook(output)
        assert book['records']['A2'].value == 7
        book.close()


def test_export_protects_existing_output_and_reports_bad_input(tmp_path):
    database, output = tmp_path / 'sample.db', tmp_path / 'sample.xlsx'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE records (value TEXT)')
        db.execute('INSERT INTO records VALUES (?)', ('bad\x00text',))
    output.write_bytes(b'existing output')
    with pytest.raises(FileExistsError):
        export_database(database, output)
    with pytest.raises(ValueError, match='没有这些表'):
        export_database(database, output, ['missing'], overwrite=True)
    with pytest.raises(ValueError, match='控制字符'):
        export_database(database, output, overwrite=True)
    assert output.read_bytes() == b'existing output'
    with pytest.raises(FileNotFoundError):
        export_database(tmp_path / 'missing.db', tmp_path / 'new.xlsx')
    assert not (tmp_path / 'missing.db').exists()


def test_sheet_and_split_column_names_do_not_collide(tmp_path):
    database, output = tmp_path / 'names.db', tmp_path / 'names.xlsx'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE "a/b" (payload TEXT, payload__part2 TEXT)')
        db.execute('INSERT INTO "a/b" VALUES (?, ?)', ('x' * 40_000, 'original'))
        db.execute('CREATE TABLE "a?b" (id INTEGER)')
    export_database(database, output)
    book = load_workbook(output)
    assert book.sheetnames == ['a_b', 'a_b_2']
    sheet = book['a_b']
    assert [cell.value for cell in sheet[1]] == ['payload', 'payload__part2_', 'payload__part2']
    assert sheet['A2'].value + sheet['B2'].value == 'x' * 40_000
    assert sheet['C2'].value == 'original'
    book.close()
