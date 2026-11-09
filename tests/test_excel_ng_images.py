"""验证 NG 图片关联、真实文件复制、重名保护和只检查模式。"""

from datetime import datetime
from pathlib import Path

import pytest
from openpyxl import Workbook

from cosmos_toolbox.excel_ng_images import collect_images, process_images, write_report


def make_excel(path, results, images, errors=()):
    book = Workbook()
    book.remove(book.active)
    for name, headers, rows in (
        ('inspection_result', ['id', 'ok', 'product_id', 'create_time'], results),
        ('image_file', ['id', 'result_id', 'filename', 'raw_image', 'result_image', 'create_time'], images),
        ('ng_error', ['result_id', 'file_id'], errors),
    ):
        sheet = book.create_sheet(name)
        sheet.append(headers)
        for row in rows:
            sheet.append(row)
    book.save(path)


def create_source(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_all_ng_faces_filename_and_raw_fallback_and_missing(tmp_path):
    excel, output, source = tmp_path / 'cosmos.xlsx', tmp_path / 'images', tmp_path / 'source'
    top = create_source(source, 'saved_image/top.png', b'original top')
    bottom = create_source(source, 'saved_image/bottom.png', b'original bottom')
    make_excel(
        excel,
        [(1, 0, 'NG item'), (2, 1, 'OK item'), (3, False, 'No images')],
        [
            (11, 1, 'E:/saved_image/top.png', 'D:/raw/top.png', 'D:/result/top.png'),
            (12, 1, None, 'E:/saved_image/bottom.png', 'D:/result/bottom.png'),
            (13, 1, 'E:/saved_image/missing.png', None, None),
            (14, 2, 'E:/saved_image/ok.png', None, None),
            (15, 1, 'D:/not_an_E_source.png', None, 'E:/result/overlay.png'),
        ],
        [(1, 11), (1, 11)],
    )
    records, count = collect_images(excel, output, source_root=source)
    assert count == 2
    assert {row['file_id'] for row in records} == {'11', '12', '13', '15', ''}
    summary = process_images(records)
    assert summary == {'copied': 2, 'missing': 1, 'no_source_path': 1, 'no_image_row': 1}
    assert (output / 'top.png').read_bytes() == top.read_bytes()
    assert (output / 'bottom.png').read_bytes() == bottom.read_bytes()
    assert top.read_bytes() == b'original top'
    assert not (output / 'overlay.png').exists()
    report = output / 'report.csv'
    write_report(records, report)
    assert 'no_source_path' in report.read_text(encoding='utf-8-sig')
    with pytest.raises(FileExistsError):
        write_report(records, report)


def test_only_ng_faces_uses_error_file_ids_without_duplicating_errors(tmp_path):
    excel, output = tmp_path / 'cosmos.xlsx', tmp_path / 'images'
    make_excel(
        excel,
        [(1, 'NG', 'NG item'), (2, 'OK', 'OK item')],
        [(11, 1, 'E:/top.png', None, None), (12, 1, 'E:/bottom.png', None, None), (13, 2, 'E:/ok.png', None, None)],
        [(1, 11), (1, 11), (2, 13)],
    )
    records, count = collect_images(excel, output, only_ng_faces=True)
    assert count == 1
    assert [row['file_id'] for row in records] == ['11']


def test_dry_run_does_not_create_destination_and_duplicate_source_is_copied_once(tmp_path):
    excel, output, source = tmp_path / 'cosmos.xlsx', tmp_path / 'images', tmp_path / 'source'
    create_source(source, 'top.png', b'original')
    make_excel(
        excel, [(1, 0, 'one'), (2, 0, 'two')], [(11, 1, 'E:/top.png', None, None), (12, 2, 'e:\\TOP.png', None, None)]
    )
    records, _ = collect_images(excel, output, source_root=source)
    assert process_images(records, dry_run=True) == {'ready': 1, 'duplicate': 1}
    assert not output.exists()
    records, _ = collect_images(excel, output, source_root=source)
    assert process_images(records) == {'copied': 1, 'duplicate': 1}
    assert len(list(output.iterdir())) == 1


def test_same_basename_from_different_folders_keeps_both_and_pairs(tmp_path):
    excel, output, source = tmp_path / 'cosmos.xlsx', tmp_path / 'images', tmp_path / 'source'
    images = []
    for result_id, folder in ((1, 'day1'), (2, 'day2')):
        for face in ('top', 'bottom'):
            relative = f'{folder}/capture_{face}.png'
            create_source(source, relative, f'{folder}/{face}'.encode())
            images.append((len(images) + 1, result_id, 'E:/' + relative, None, None))
    make_excel(excel, [(1, 0, 'one'), (2, 0, 'two')], images)
    records, _ = collect_images(excel, output, source_root=source)
    assert process_images(records) == {'copied': 4}
    assert len(list(output.iterdir())) == 4
    for result_id in ('1', '2'):
        targets = [Path(row['destination_path']).name for row in records if row['result_id'] == result_id]
        assert len({name.split('__')[0] for name in targets}) == 1
    # 再次执行可以跳过相同内容；异内容的既有文件不能被覆盖。
    records, _ = collect_images(excel, output, source_root=source)
    assert process_images(records) == {'existing_same': 4}
    target = Path(records[0]['destination_path'])
    target.write_bytes(b'keep existing different content')
    records, _ = collect_images(excel, output, source_root=source)
    summary = process_images(records)
    assert summary == {'conflict': 1, 'existing_same': 3}
    assert target.read_bytes() == b'keep existing different content'


def test_unknown_ng_value_fails_before_copying(tmp_path):
    excel, output = tmp_path / 'cosmos.xlsx', tmp_path / 'images'
    make_excel(excel, [(1, None, 'unknown')], [(11, 1, 'E:/top.png', None, None)])
    with pytest.raises(ValueError, match='ok 字段无法识别'):
        collect_images(excel, output)
    assert not output.exists()


def test_date_range_includes_whole_end_day_and_uses_detection_time(tmp_path):
    excel, output = tmp_path / 'cosmos.xlsx', tmp_path / 'images'
    results = [
        (1, 0, 'before', '2026-10-08 23:59:59.999999'),
        (2, 0, 'start', '2026-10-09 00:00:00'),
        (3, 0, 'Excel datetime', datetime(2026, 10, 9, 12, 30)),
        (4, 0, 'end', '2026-10-09 23:59:59.999999'),
        (5, 0, 'after', '2026-10-10 00:00:00'),
        (6, 1, 'OK', '2026-10-09 12:30:00'),
    ]
    images = [(i, i, f'E:/{i}.png', None, None, '2020-01-01 00:00:00') for i in range(1, 7)]
    make_excel(excel, results, images)
    records, count = collect_images(excel, output, start_date='2026-10-09', end_date='2026-10-09')
    assert count == 3
    assert {row['result_id'] for row in records} == {'2', '3', '4'}
    assert not output.exists()


@pytest.mark.parametrize(
    ('bounds', 'expected'), [({'start_date': '2026-10-09'}, {'2', '3'}), ({'end_date': '2026-10-09'}, {'1', '2'})]
)
def test_date_range_can_have_one_open_end(tmp_path, bounds, expected):
    excel, output = tmp_path / 'cosmos.xlsx', tmp_path / 'images'
    results = [(i, 0, str(i), f'2026-10-{day} 23:59:59') for i, day in ((1, '08'), (2, '09'), (3, '10'))]
    make_excel(excel, results, [(i, i, f'E:/{i}.png', None, None) for i in range(1, 4)])
    records, count = collect_images(excel, output, **bounds)
    assert count == len(expected)
    assert {row['result_id'] for row in records} == expected


def test_exact_time_boundaries_and_ng_face_filter_work_together(tmp_path):
    excel, output = tmp_path / 'cosmos.xlsx', tmp_path / 'images'
    times = ['08:29:59.999999', '08:30:00', '09:00:00', '09:00:00.000001']
    make_excel(
        excel,
        [(i, 0, str(i), '2026-10-09 ' + clock) for i, clock in enumerate(times, start=1)],
        [(i, i, f'E:/{i}.png', None, None) for i in range(1, 5)] + [(22, 2, 'E:/other_face.png', None, None)],
        [(i, i) for i in range(1, 5)],
    )
    records, count = collect_images(
        excel, output, only_ng_faces=True, start_date='2026-10-09 08:30:00', end_date='2026-10-09 09:00:00'
    )
    assert count == 2
    assert {row['file_id'] for row in records} == {'2', '3'}


@pytest.mark.parametrize(
    ('bounds', 'message'),
    [
        ({'start_date': '2026-10-10', 'end_date': '2026-10-09'}, '不能晚于'),
        ({'end_date': '2026-02-30'}, '结束日期 无效'),
        ({'start_date': '2026/10/09'}, '开始日期 无效'),
    ],
)
def test_invalid_date_range_is_rejected_before_opening_excel(tmp_path, bounds, message):
    with pytest.raises(ValueError, match=message):
        collect_images(tmp_path / 'missing.xlsx', tmp_path / 'images', **bounds)


def test_date_filter_rejects_missing_detection_time_before_copying(tmp_path):
    excel, output = tmp_path / 'cosmos.xlsx', tmp_path / 'images'
    make_excel(excel, [(1, 0, 'no time')], [(1, 1, 'E:/top.png', None, None)])
    with pytest.raises(ValueError, match='create_time'):
        collect_images(excel, output, start_date='2026-10-09')
    assert not output.exists()
