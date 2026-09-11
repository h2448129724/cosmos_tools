"""Pure planning rules shared by the YOLO Python and CLI shells.

This module deliberately contains no Ultralytics, Torch, process, environment, or
filesystem effects.  The command-line wrappers and trainer GUI adapt the plan to
their respective imperative runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping


ActionName = Literal["train", "predict", "export_onnx"]
PathStyle = Literal["native", "posix"]


@dataclass(frozen=True, slots=True)
class YoloMaterializationIntent:
    """Describe output routing without creating directories or moving files."""

    arg_name: str
    target: Path | None
    project: Path | None
    name: str | None
    is_directory: bool
    move_result: bool = False


@dataclass(frozen=True, slots=True)
class YoloActionPlan:
    """Normalized YOLO action inputs and deterministic projections.

    ``from_params`` accepts GUI/schema mappings while the explicit fields make
    direct Python callers straightforward and typed.  Empty GUI values are
    treated as omitted, except booleans where ``False`` is meaningful.
    """

    action_name: ActionName
    task: str = "detect"
    model_series: str = "yolo11"
    model_size: str = "n"
    custom_model: str = ""
    data: str | Path | None = None
    model: str | Path | None = None
    source: str | Path | None = None
    epochs: int = 100
    imgsz: int = 640
    batch: int = 4
    conf: float = 0.25
    iou: float = 0.7
    device: str | int | None = None
    workers: int = 0
    patience: int = 30
    seed: int = 42
    cache: bool = False
    amp: bool = False
    save: bool = True
    save_txt: bool = False
    save_conf: bool = False
    save_crop: bool = False
    opset: int = 11
    simplify: bool = False
    dynamic: bool = False
    half: bool = False
    save_dir: str | Path | None = None
    run_name: str = ""
    output_dir: str | Path | None = None
    output: str | Path | None = None

    @classmethod
    def from_params(cls, action_name: str, params: Mapping[str, object]) -> "YoloActionPlan":
        """Build a plan from schema/GUI values, applying one set of defaults."""

        if action_name not in {"train", "predict", "export_onnx"}:
            raise ValueError(f"unsupported YOLO action: {action_name}")

        def value(name: str, default: object = None) -> object:
            raw = params.get(name, default)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                return default
            return raw

        def text(name: str, default: str = "") -> str:
            raw = value(name, default)
            return str(raw).strip() if raw is not None else default

        def integer(name: str, default: int) -> int:
            raw = value(name, default)
            return int(raw)  # type: ignore[arg-type]

        def number(name: str, default: float) -> float:
            raw = value(name, default)
            return float(raw)  # type: ignore[arg-type]

        def boolean(name: str, default: bool) -> bool:
            raw = params.get(name)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                return default
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str):
                normalized = raw.strip().lower()
                if normalized in {"true", "1", "yes", "on"}:
                    return True
                if normalized in {"false", "0", "no", "off"}:
                    return False
            raise ValueError(f"invalid boolean value for {name}: {raw!r}")

        return cls(
            action_name=action_name,  # type: ignore[arg-type]
            task=text("task", "detect"),
            model_series=text("model_series", "yolo11"),
            model_size=text("model_size", "n"),
            custom_model=text("custom_model"),
            data=value("data"),
            model=value("model"),
            source=value("source"),
            epochs=integer("epochs", 100),
            imgsz=integer("imgsz", 1024 if action_name == "train" else 640),
            batch=integer("batch", 4),
            conf=number("conf", 0.25),
            iou=number("iou", 0.7),
            device=value("device", "0" if action_name == "train" else None),
            workers=integer("workers", 0),
            patience=integer("patience", 30),
            seed=integer("seed", 42),
            cache=boolean("cache", False),
            amp=boolean("amp", False),
            save=boolean("save", True),
            save_txt=boolean("save_txt", False),
            save_conf=boolean("save_conf", False),
            save_crop=boolean("save_crop", False),
            opset=integer("opset", 11),
            simplify=boolean("simplify", False),
            dynamic=boolean("dynamic", False),
            half=boolean("half", False),
            save_dir=value("save_dir"),
            run_name=text("run_name"),
            output_dir=value("output_dir"),
            output=value("output"),
        )

    from_mapping = from_params

    @property
    def model_name(self) -> str:
        """Resolve custom model precedence and task-specific official suffixes."""

        if self.custom_model.strip():
            return self.custom_model.strip()
        base = f"{self.model_series}{self.model_size}"
        suffix = {"segment": "-seg", "classify": "-cls", "pose": "-pose"}.get(self.task, "")
        return f"{base}{suffix}.pt"

    def train_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "data": _text_value(self.data),
            "task": self.task,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "project": _path_text(self._train_project(), "native"),
            "name": self._train_name(),
            "workers": self.workers,
            "patience": self.patience,
            "seed": self.seed,
            "cache": self.cache,
            "amp": self.amp,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        return kwargs

    def predict_kwargs(self) -> dict[str, object]:
        output_dir = self._predict_output_dir()
        kwargs: dict[str, object] = {
            "source": _text_value(self.source),
            "task": self.task,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "project": _path_text(output_dir.parent, "native"),
            "name": output_dir.name,
            "save": self.save,
            "save_txt": self.save_txt,
            "save_conf": self.save_conf,
            "save_crop": self.save_crop,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        return kwargs

    def export_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "format": "onnx",
            "imgsz": self.imgsz,
            "opset": self.opset,
            "simplify": self.simplify,
            "dynamic": self.dynamic,
            "half": self.half,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        return kwargs

    def python_kwargs(self) -> dict[str, object]:
        """Project to the direct Ultralytics Python call for this action."""

        if self.action_name == "train":
            return self.train_kwargs()
        if self.action_name == "predict":
            return self.predict_kwargs()
        return self.export_kwargs()

    ultralytics_kwargs = python_kwargs

    def cli_argv(self, *, path_style: PathStyle = "native") -> tuple[str, ...]:
        """Project this plan to Ultralytics ``yolo`` CLI arguments."""

        if self.action_name == "train":
            command = [
                "yolo",
                self.task,
                "train",
                f"data={_text_value(self.data)}",
                f"model={self.model_name}",
            ]
            _append(command, self.epochs, "epochs")
            _append(command, self.imgsz, "imgsz")
            _append(command, self.batch, "batch")
            _append(command, self.workers, "workers")
            _append(command, self.patience, "patience")
            _append(command, self.seed, "seed")
            _append_optional(command, self.device, "device")
            command.extend((f"cache={self.cache}", f"amp={self.amp}"))
            project, name = self._train_project_name()
            command.extend([f"project={_path_text(project, path_style)}", f"name={name}"])
            return tuple(command)

        if self.action_name == "predict":
            command = [
                "yolo",
                self.task,
                "predict",
                f"model={_text_value(self.model)}",
                f"source={_text_value(self.source)}",
            ]
            _append(command, self.imgsz, "imgsz")
            _append(command, self.conf, "conf")
            _append(command, self.iou, "iou")
            _append_optional(command, self.device, "device")
            command.append(f"save={self.save}")
            command.extend(
                (
                    f"save_txt={self.save_txt}",
                    f"save_conf={self.save_conf}",
                    f"save_crop={self.save_crop}",
                )
            )
            output_dir = self._predict_output_dir()
            command.extend([f"project={_path_text(output_dir.parent, path_style)}", f"name={output_dir.name}"])
            return tuple(command)

        command = ["yolo", "export", f"model={_text_value(self.model)}", "format=onnx"]
        _append(command, self.imgsz, "imgsz")
        _append(command, self.opset, "opset")
        command.extend(
            (
                f"simplify={self.simplify}",
                f"dynamic={self.dynamic}",
                f"half={self.half}",
            )
        )
        _append_optional(command, self.device, "device")
        # Ultralytics export does not materialize ONNX files through
        # project/name.  Exact output routing is represented by the separate
        # intent and executed by ``module_argv``'s Python wrapper.
        return tuple(command)

    def module_argv(
        self,
        *,
        python_executable: str = "python",
        module_name: str = "yolo.scripts.export_onnx",
    ) -> tuple[str, ...]:
        """Project ONNX export to the Python wrapper's argparse contract."""

        if self.action_name != "export_onnx":
            raise ValueError("module_argv is only defined for export_onnx")
        command = [python_executable, "-m", module_name, "--model", _text_value(self.model)]
        _append_flag_value(command, "--imgsz", self.imgsz)
        _append_flag_value(command, "--opset", self.opset)
        for name, enabled in (
            ("--simplify", self.simplify),
            ("--dynamic", self.dynamic),
            ("--half", self.half),
        ):
            if enabled:
                command.append(name)
        _append_optional_flag_value(command, "--device", self.device)
        if self.output not in (None, ""):
            _append_flag_value(command, "--output", _text_value(self.output))
        return tuple(command)

    to_cli_argv = cli_argv

    def materialization_intent(self) -> YoloMaterializationIntent:
        if self.action_name == "train":
            project, name = self._train_project_name()
            target = project / name
            return YoloMaterializationIntent("save_dir", target, project, name, True)
        if self.action_name == "predict":
            target = self._predict_output_dir()
            project, name = target.parent, target.name
            return YoloMaterializationIntent("output_dir", target, project, name, True)
        target = Path(self.output) if self.output not in (None, "") else None  # type: ignore[arg-type]
        project, name = (target.parent, target.stem) if target is not None else (None, None)
        return YoloMaterializationIntent("output", target, project, name, False, move_result=target is not None)

    output_intent = materialization_intent

    def _train_project_name(self) -> tuple[Path, str]:
        target = (
            Path(self.save_dir)
            if self.save_dir not in (None, "")
            else Path("checkpoints/yolo/runs")
        )  # type: ignore[arg-type]
        if self.run_name.strip():
            return target.parent, self.run_name.strip()
        return target.parent, target.name

    def _train_project(self) -> Path:
        return self._train_project_name()[0]

    def _train_name(self) -> str:
        return self._train_project_name()[1]

    def _predict_output_dir(self) -> Path:
        return Path(self.output_dir) if self.output_dir not in (None, "") else Path("predict_output")


def _text_value(value: object) -> str:
    return "" if value is None else str(value)


def _append(command: list[str], value: object, name: str) -> None:
    if value not in (None, ""):
        command.append(f"{name}={value}")


def _append_optional(command: list[str], value: object, name: str) -> None:
    if value not in (None, ""):
        command.append(f"{name}={value}")


def _append_flag_value(command: list[str], flag: str, value: object) -> None:
    command.extend((flag, str(value)))


def _append_optional_flag_value(command: list[str], flag: str, value: object) -> None:
    if value not in (None, ""):
        _append_flag_value(command, flag, value)


def _path_text(path: Path, style: PathStyle) -> str:
    return path.as_posix() if style == "posix" else str(path)


def resolve_model_name(task: str, model_series: str, model_size: str, custom_model: str) -> str:
    """Compatibility helper for the former train.py public function."""

    return YoloActionPlan("train", task, model_series, model_size, custom_model).model_name
