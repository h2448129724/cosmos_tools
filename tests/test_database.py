import hashlib
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication, QAbstractItemView
from cosmos_toolbox.database_reader import inspect_database, open_readonly
from cosmos_toolbox.database_ui import DatabasePage
from cosmos_toolbox.capability_catalog import plan_default_capabilities


class DatabaseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "数据 #1.db"
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, value TEXT, payload BLOB)")
            connection.executemany(
                "INSERT INTO records VALUES (?, ?, ?)", [(i, f"item {i}", b"abc") for i in range(405)]
            )
            connection.execute('CREATE TABLE "odd""table" ("a""b" TEXT)')
            connection.execute('INSERT INTO "odd""table" VALUES (?)', ("100%_'",))
        connection.close()

    def test_readonly_and_no_creation(self):
        before = hashlib.sha256(self.path.read_bytes()).digest()
        with open_readonly(self.path) as connection:
            for sql in ("DELETE FROM records", "CREATE TABLE forbidden (id)", "PRAGMA user_version=99"):
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute(sql)
        inspect_database(self.path)
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).digest())
        missing = self.path.parent / "missing.db"
        with self.assertRaises(FileNotFoundError):
            inspect_database(missing)
        self.assertFalse(missing.exists())

    def test_pagination_filter_and_identifiers(self):
        page = inspect_database(self.path, "records", order="id", descending=True, offset=200)
        self.assertEqual(page["total"], 405)
        self.assertEqual(page["rows"][0][0], 204)
        result = inspect_database(self.path, 'odd"table', column='a"b', keyword="%_'")
        self.assertEqual(result["total"], 1)
        self.assertEqual(inspect_database(self.path, "records", keyword="' OR 1=1 --")["total"], 0)
        with self.assertRaises(ValueError):
            inspect_database(self.path, "records; DROP TABLE records")

    def test_live_wal_reads_committed_records(self):
        with sqlite3.connect(self.path) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute('INSERT INTO records VALUES (500, "live", NULL)')
            writer.commit()
            result = inspect_database(self.path, "records", column="value", keyword="live")
            self.assertEqual(result["rows"], [(500, "live", None)])
        writer.close()

    def test_ui_load_and_navigation(self):
        app = QApplication.instance() or QApplication([])
        page = DatabasePage()
        page.path.setText(str(self.path))
        page.resize(960, 640)
        page.show()

        def wait():
            deadline = time.monotonic() + 10
            while page.worker is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            self.assertIsNone(page.worker)

        page.load(reset=True)
        wait()
        page.table.setCurrentText("records")
        wait()
        self.assertEqual(page.data.rowCount(), 200)
        self.assertEqual(page.data.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        page.next.click()
        wait()
        self.assertEqual(page.offset, 200)
        page.show_detail(0, 1)
        self.assertEqual(page.detail.toPlainText(), "item 200")
        page.close()
        item = next(
            p for p in plan_default_capabilities([], include_project_capabilities=True) if p.spec.key == "data.database"
        )
        self.assertEqual(item.page_factory_key, "database")


if __name__ == "__main__":
    unittest.main()
