from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtGui import QFont, QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from cosmos_toolbox.paths import TOOLBOX_ROOT, ensure_import_paths  # noqa: E402

ensure_import_paths()

from cosmos_toolbox.app import ToolboxWindow  # noqa: E402
from cosmos_toolbox.project_context import ProjectContext  # noqa: E402


@dataclass(frozen=True, slots=True)
class _IsolatedRuntimeState:
    config_path: Path
    settings_path: Path


@contextmanager
def _isolated_runtime_state(root: Path):
    """Route legacy workspace persistence to a disposable QA directory."""
    from apps.cabf_flow.flow import CONFIG_PATH
    from apps.labeling_ui.app import main_window as labeling_main_window
    from trainer_gui import main_window as trainer_main_window

    root = Path(root).resolve()
    config_path = root / "cabf_flow" / "default_paths.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG_PATH, config_path)

    settings_root = root / "trainer"
    settings_path = settings_root / "artifacts" / "gui_settings.json"
    real_settings_path = TOOLBOX_ROOT / "artifacts" / "gui_settings.json"
    if real_settings_path.exists():
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(real_settings_path, settings_path)

    original_config_path = labeling_main_window.CONFIG_PATH
    original_settings_manager = trainer_main_window.SettingsManager

    class _QaSettingsManager(original_settings_manager):
        def __init__(self, _project_root):
            super().__init__(settings_root)

    labeling_main_window.CONFIG_PATH = config_path
    trainer_main_window.SettingsManager = _QaSettingsManager
    try:
        yield _IsolatedRuntimeState(config_path=config_path, settings_path=settings_path)
    finally:
        labeling_main_window.CONFIG_PATH = original_config_path
        trainer_main_window.SettingsManager = original_settings_manager


def _persistence_snapshot() -> dict[Path, bytes | None]:
    from apps.cabf_flow.flow import CONFIG_PATH

    watched = (Path(CONFIG_PATH), TOOLBOX_ROOT / "artifacts" / "gui_settings.json")
    return {path: path.read_bytes() if path.exists() else None for path in watched}


def _assert_persistence_unchanged(before: dict[Path, bytes | None]) -> None:
    changed = []
    for path, expected in before.items():
        actual = path.read_bytes() if path.exists() else None
        if actual != expected:
            changed.append(str(path))
    if changed:
        raise AssertionError("视觉 QA 修改了真实持久化文件：" + ", ".join(changed))


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "page"


def _font(size: int):
    candidates = (
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "seguisym.ttf",
    )
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _install_qt_qa_font(app: QApplication) -> None:
    """Load a concrete CJK font so offscreen screenshots can verify labels."""
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for path in (font_dir / "msyh.ttc", font_dir / "msyhbd.ttc", font_dir / "simsun.ttc"):
        if not path.exists():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 10))
            return


def _make_gallery(items: list[tuple[str, str, Path]], output: Path) -> None:
    thumb_width, thumb_height = 640, 400
    label_height = 46
    columns = 2
    rows = (len(items) + columns - 1) // columns
    gallery = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "#d9dcdf")
    draw = ImageDraw.Draw(gallery)
    font = _font(17)
    for index, (key, title, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        cell_x = (index % columns) * thumb_width
        cell_y = (index // columns) * (thumb_height + label_height)
        x = cell_x + (thumb_width - image.width) // 2
        y = cell_y + (thumb_height - image.height) // 2
        gallery.paste(image, (x, y))
        draw.rectangle((cell_x, cell_y + thumb_height, cell_x + thumb_width, cell_y + thumb_height + label_height), fill="#ffffff")
        draw.text((cell_x + 10, cell_y + thumb_height + 11), f"{title}  [{key}]", fill="#202428", font=font)
    gallery.save(output)


def render(output_dir: Path, width: int = 1600, height: int = 1000) -> Path:
    persistence_before = _persistence_snapshot()
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / "workspace"
    images = workspace / "images"
    annotations = workspace / "annotations"
    outputs = workspace / "outputs"
    for directory in (images, annotations, outputs):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="cosmos-toolbox-ui-qa-") as state_dir:
            with _isolated_runtime_state(Path(state_dir)):
                app = QApplication.instance() or QApplication(sys.argv[:1])
                _install_qt_qa_font(app)
                context = ProjectContext(Path(state_dir) / "workspace_state.json")
                context.set_dataset_root(workspace)
                context.update(project_name="UI 逐页检查", image_dir=images, annotation_dir=annotations, output_root=outputs)
                window = ToolboxWindow(context=context)
                window.resize(width, height)
                window.show()
                app.processEvents()

                rendered: list[tuple[str, str, Path]] = []
                errors: list[str] = []
                try:
                    for index, capability in enumerate(window.catalog.visible(), start=1):
                        try:
                            window.navigate(capability.key)
                            for _ in range(4):
                                QCoreApplication.processEvents()
                            path = output_dir / f"{index:02d}_{_safe_name(capability.key)}.png"
                            if not window.grab().save(str(path)):
                                raise RuntimeError("window.grab().save returned false")
                            rendered.append((capability.key, capability.title, path))
                            print(f"[OK] {capability.key}: {path}")
                        except Exception as exc:
                            errors.append(f"{capability.key}: {exc}")
                            print(f"[ERROR] {capability.key}: {exc}", file=sys.stderr)
                finally:
                    window.close()
                    app.processEvents()

                gallery = output_dir / "capability_gallery.png"
                _make_gallery(rendered, gallery)
                if errors:
                    raise RuntimeError("页面渲染失败：" + "; ".join(errors))
                print(f"[GALLERY] {gallery}")
                return gallery
    finally:
        _assert_persistence_unchanged(persistence_before)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render every Toolbox capability into a visual QA gallery.")
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "ui_capability_gallery")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()
    render(args.output.resolve(), args.width, args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
