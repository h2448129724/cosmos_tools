"""SQLite inspection adapter. Never import the production database session here."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import time


@contextmanager
def open_readonly(path):
    source = Path(path).expanduser().resolve(strict=True)
    connection = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=2)
    try:
        connection.execute("PRAGMA query_only=ON")
        deadline = time.monotonic() + 5
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        yield connection
    finally:
        connection.close()


def quote_identifier(value):
    return '"' + value.replace('"', '""') + '"'


def read_inspection_details(path, result_id):
    """Fetch every saved face for one result without production initialization."""
    with open_readonly(path) as connection:
        rows = connection.execute(
            "SELECT id, result_detail FROM inspection_metadata WHERE result_id = ? ORDER BY id LIMIT 101",
            (result_id,),
        ).fetchall()
        if len(rows) > 100:
            raise ValueError("该记录明细超过 100 条，请在 inspection_metadata 表按 result_id 筛选查看")
        return rows


def inspect_database(path, table=None, column=None, keyword="", order=None, descending=False, offset=0, limit=200):
    """Return a bounded page and schema from one short-lived read transaction."""
    if offset < 0 or not 1 <= limit <= 500:
        raise ValueError("分页参数超出范围")
    with open_readonly(path) as connection:
        connection.execute("BEGIN")
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        if table is None:
            table = "inspection_result" if "inspection_result" in tables else next(iter(tables), None)
        if table is None:
            return dict(tables=[], table=None, schema=[], columns=[], rows=[], total=0, offset=0)
        if table not in tables:
            raise ValueError("数据表不存在，请重新打开数据库")
        quoted = quote_identifier(table)
        schema = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
        columns = [row[1] for row in schema]
        for name in (column, order):
            if name is not None and name not in columns:
                raise ValueError("字段不存在")
        params = []
        where = ""
        if keyword:
            fields = [column] if column else columns
            where = (
                " WHERE ("
                + " OR ".join(f"instr(CAST({quote_identifier(name)} AS TEXT), ?) > 0" for name in fields)
                + ")"
            )
            params = [keyword] * len(fields)
        keys = [row[1] for row in sorted(schema, key=lambda row: row[5]) if row[5]]
        sort_fields = ([order] if order else []) + [key for key in keys if key != order]
        sorting = (
            (
                " ORDER BY "
                + ", ".join(quote_identifier(name) + (" DESC" if descending else " ASC") for name in sort_fields)
            )
            if sort_fields
            else ""
        )
        total = connection.execute(f"SELECT COUNT(*) FROM {quoted}{where}", params).fetchone()[0]
        offset = min(offset, max(0, ((total - 1) // limit) * limit))
        rows = connection.execute(
            f"SELECT * FROM {quoted}{where}{sorting} LIMIT ? OFFSET ?", [*params, limit, offset]
        ).fetchall()
        return dict(tables=tables, table=table, schema=schema, columns=columns, rows=rows, total=total, offset=offset)
