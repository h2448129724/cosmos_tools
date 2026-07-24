from __future__ import annotations

import argparse
import json
import os
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from cosmos_toolbox.paths import COSMOS_ROOT


DEFAULT_BACKEND_CONFIG = COSMOS_ROOT / "assets" / "config" / "backend_config.yaml"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "cosmos_pipeline"


@dataclass(frozen=True, slots=True)
class InputSlot:
    name: str
    min_files: int = 1
    max_files: int = 1


@dataclass(frozen=True, slots=True)
class PipelineDescriptor:
    project: str
    product: str
    backend_key: str
    engine_default: str
    slots: tuple[InputSlot, ...]


@dataclass(frozen=True, slots=True)
class InputCase:
    case_id: str
    inputs: dict[str, tuple[Path, ...]]
    color_space: str = "rgb"


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    config_path: Path
    backend_config_path: Path
    input_manifest_path: Path
    output_dir: Path
    engine: str = "auto"
    dry_run: bool = False
    persist_db: bool = False
    fail_on_ng: bool = False
    save_annotated_images: bool = True
    onnx_session_report: Path | None = None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 解析失败：{path}：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"YAML 根节点必须是对象：{path}")
    return data


def backend_key_for_project(project: str) -> str:
    normalized = project.lower().replace("-", "_")
    return "dab" if normalized.startswith("dab") else normalized


def describe_config(path: str | Path) -> PipelineDescriptor:
    data = _read_yaml(Path(path).expanduser().resolve())
    inspection = data.get("inspection")
    if not isinstance(inspection, dict):
        raise ValueError("产品配置缺少 inspection")
    project = str(inspection.get("project") or "").strip()
    product = str(inspection.get("product") or "").strip()
    conf = inspection.get("conf")
    if not project or not isinstance(conf, dict):
        raise ValueError("产品配置缺少 inspection.project 或 inspection.conf")

    if project == "CAB":
        loader = data.get("image_loader") or {}
        nested = loader.get("loaders") if isinstance(loader, dict) else None
        file_count = max(1, len(nested)) if isinstance(nested, list) else 1
        slots = (InputSlot("combined", file_count, file_count),)
        engine = "service"
    else:
        preferred = {"top", "bottom"} if project == "CAB-F" else {"front", "back"}
        face_names = tuple(str(name) for name, value in conf.items() if name in preferred and isinstance(value, dict))
        if not face_names:
            face_names = tuple(str(key) for key, value in conf.items() if isinstance(value, dict))
        if not face_names:
            raise ValueError("inspection.conf 没有输入侧别")
        slots = tuple(InputSlot(name) for name in face_names)
        engine = "service" if project.startswith("DAB") else "local"
    return PipelineDescriptor(project, product, backend_key_for_project(project), engine, slots)


def load_input_manifest(path: str | Path, descriptor: PipelineDescriptor) -> list[InputCase]:
    manifest_path = Path(path).expanduser().resolve()
    data = _read_yaml(manifest_path)
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("输入清单必须包含非空 cases 列表")
    base = manifest_path.parent
    slot_map = {slot.name: slot for slot in descriptor.slots}
    cases: list[InputCase] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, dict) or not isinstance(raw.get("inputs"), dict):
            raise ValueError(f"cases[{index - 1}] 缺少 inputs")
        case_id = str(raw.get("id") or f"case-{index:03d}").strip()
        if case_id in seen:
            raise ValueError(f"输入清单 case id 重复：{case_id}")
        seen.add(case_id)
        unknown = set(raw["inputs"]) - set(slot_map)
        if unknown:
            raise ValueError(f"{case_id} 包含未知输入槽：{', '.join(sorted(unknown))}")
        resolved: dict[str, tuple[Path, ...]] = {}
        for name, slot in slot_map.items():
            values = raw["inputs"].get(name)
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                raise ValueError(f"{case_id}.{name} 必须是文件列表")
            files = tuple((base / str(value)).resolve() if not Path(str(value)).is_absolute() else Path(str(value)).resolve() for value in values)
            if not slot.min_files <= len(files) <= slot.max_files:
                raise ValueError(f"{case_id}.{name} 需要 {slot.min_files} 个文件，当前 {len(files)} 个")
            missing = [str(file) for file in files if not file.is_file()]
            if missing:
                raise FileNotFoundError(f"输入文件不存在：{', '.join(missing)}")
            resolved[name] = files
        color_space = str(raw.get("color_space") or data.get("color_space") or "rgb").lower()
        if color_space not in {"rgb", "bgr", "gray"}:
            raise ValueError(f"{case_id}.color_space 仅支持 rgb/bgr/gray")
        cases.append(InputCase(case_id, resolved, color_space))
    return cases


def validate_request(request: PipelineRequest) -> tuple[PipelineDescriptor, list[InputCase]]:
    for label, path in (
        ("产品配置", request.config_path),
        ("后端配置", request.backend_config_path),
        ("输入清单", request.input_manifest_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在：{path}")
    if request.engine not in {"auto", "local", "service"}:
        raise ValueError(f"未知执行引擎：{request.engine}")
    descriptor = describe_config(request.config_path)
    backend = _read_yaml(request.backend_config_path)
    if descriptor.backend_key not in backend:
        raise ValueError(f"后端配置缺少项目节点：{descriptor.backend_key}")
    cases = load_input_manifest(request.input_manifest_path, descriptor)
    return descriptor, cases


def _serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(item) for item in value]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _serializable(to_list())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _serializable(item())
        except ValueError:
            pass
    if hasattr(value, "__dict__"):
        return _serializable(vars(value))
    return repr(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _event(event_type: str, **payload: Any) -> None:
    print("COSMOS_EVENT " + json.dumps({"type": event_type, **_serializable(payload)}, ensure_ascii=False), flush=True)


def _load_image(path: Path, color_space: str):
    import cv2
    import numpy as np
    from PIL import Image

    if color_space == "rgb":
        # Cosmos line-scan frames legitimately exceed Pillow's generic web
        # image safety threshold.  Match biz.image_loader for trusted local
        # production files, while restoring the process-wide setting after IO.
        original_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            with Image.open(path) as source:
                return np.asarray(source.convert("RGB"))
        finally:
            Image.MAX_IMAGE_PIXELS = original_limit
    flag = cv2.IMREAD_GRAYSCALE if color_space == "gray" else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return image


def _save_image(path: Path, image: Any, color_space: str) -> None:
    import cv2
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if color_space == "rgb" and getattr(image, "ndim", 0) == 3:
        Image.fromarray(image).save(path)
        return
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"无法编码结果图：{path}")
    encoded.tofile(path)


def _configure_cosmos(request: PipelineRequest, descriptor: PipelineDescriptor) -> None:
    from biz import config_loader

    config_loader.config.load_config(str(request.config_path))
    backend = config_loader.load_config(str(request.backend_config_path))
    subtree = backend.get(descriptor.backend_key)
    if not isinstance(subtree, dict):
        raise ValueError(f"后端配置缺少项目节点：{descriptor.backend_key}")
    config_loader.reload_backend_config({descriptor.backend_key: subtree}, project=descriptor.backend_key)


def _engine(request: PipelineRequest, descriptor: PipelineDescriptor) -> str:
    return descriptor.engine_default if request.engine == "auto" else request.engine


def _detect_local(descriptor: PipelineDescriptor, slot: str, images: list[Any], index: int):
    from biz.config_loader import config

    conf = deepcopy(config.inspection.get("conf") or {})
    project = descriptor.project
    if project == "CAB":
        from algo.cab_eval import cab_evaluation

        return images[-1], cab_evaluation(images, conf)
    if project == "CAB-F":
        from algo.cab_f_eval import cab_f_evaluation

        conf["face"] = index
        return cab_f_evaluation(images[-1], conf)
    if project.startswith("DAB"):
        from algo.dab_eval import dab_evaluation
        from utils import rotate_image

        image = rotate_image(images[0], config.inspection.get("rotate", 0))
        conf["face"] = index
        return image, dab_evaluation(image, conf)
    if project == "OS-DAB":
        from algo.os_dab_eval import os_dab_evaluation
        from utils import rotate_image

        image = rotate_image(images[0], config.inspection.get("rotate", 0))
        conf["face"] = index
        result, prepared = os_dab_evaluation(image, conf)
        return prepared, result
    raise ValueError(f"不支持的 Cosmos 项目：{project}")


def _detect_service(descriptor: PipelineDescriptor, images: list[Any], index: int):
    from biz.services.algo_service import AlgoService

    if descriptor.project == "CAB":
        return images[-1], AlgoService.inspect_images(images, index)
    if descriptor.project.startswith("DAB"):
        return images[0], AlgoService.inspect_np_image(images[0], index)
    raise ValueError(f"{descriptor.project} 没有生产算法服务入口，请使用 local/auto")


def _evaluate_and_draw(descriptor: PipelineDescriptor, result: dict[str, Any], image: Any, index: int):
    import numpy as np

    from algo import cab_drawing, cab_f_drawing, dab_drawing, os_dab_drawing
    from algo.utils import adjust
    from biz.checker import CABFChecker, Checker, OSDABChecker
    from biz.config_loader import config

    project = descriptor.project
    definitions = (config.inspection.get("ok_checkers") or [])[index] or []
    checks: dict[str, Any] = {}
    messages: list[str] = []
    passed = True
    face = descriptor.slots[index].name
    calibration = result.get("calibration") if project == "OS-DAB" else None
    if isinstance(calibration, dict) and not calibration.get("ok", True):
        passed = False
        messages.append("图像校正失败")
    for definition in definitions:
        method = str(definition.get("check_method") or "")
        if project == "CAB-F":
            outcome = CABFChecker.evaluate(method, result, config.inspection.get("conf") or {})
            ok, message = outcome.as_tuple()
            checks[method] = outcome
        elif project == "OS-DAB":
            ok, message = getattr(OSDABChecker, method)(result, (config.inspection.get("conf") or {}).get(face, {}))
            checks[method] = ok
        elif project.startswith("DAB"):
            ok, message = getattr(Checker, method)(result, (config.inspection.get("conf") or {}).get(face, {}))
            checks[method] = ok
        else:
            ok, message = getattr(Checker, method)(result, config.inspection.get("conf") or {})
            checks[method] = ok
        if not ok:
            passed = False
            if message:
                messages.append(str(message))
    for path, error in _collect_failed_results(result):
        passed = False
        messages.append(f"{path} 执行失败：{error}")

    if project == "CAB-F":
        drawn = cab_f_drawing.draw(image, result, config.inspection.get("conf") or {}, checks)
    elif project == "OS-DAB":
        drawn = os_dab_drawing.draw(image, result, (config.inspection.get("conf") or {}).get(face, {}), checks)
    elif project.startswith("DAB"):
        if result.get("face") == "back" and "origin_anchors" in result:
            image, _ = adjust(image, np.array(result["origin_anchors"]))
            drawn = dab_drawing.draw(image, result, (config.inspection.get("conf") or {}).get(face, {}), checks)
        elif result.get("face") == "back":
            target = np.stack((image.copy(),) * 3, axis=-1) if getattr(image, "ndim", 0) == 2 else image
            drawn = {"output": target}
        else:
            drawn = dab_drawing.draw(image, result, (config.inspection.get("conf") or {}).get(face, {}), checks)
    else:
        drawn = cab_drawing.draw(image, result, config.inspection.get("conf") or {}, checks)
    annotated = drawn.get("output") if isinstance(drawn, dict) else drawn
    return passed, checks, messages, annotated


def _collect_failed_results(node: Any, path: str = "result") -> list[tuple[str, str]]:
    if isinstance(node, dict):
        failures: list[tuple[str, str]] = []
        if node.get("failed") is True:
            failures.append((path, str(node.get("error") or "未知错误")))
        for key, value in node.items():
            if key not in {"failed", "error"}:
                failures.extend(_collect_failed_results(value, f"{path}.{key}"))
        return failures
    if isinstance(node, (list, tuple)):
        failures = []
        for index, value in enumerate(node):
            failures.extend(_collect_failed_results(value, f"{path}[{index}]"))
        return failures
    return []


def _run_case(case: InputCase, descriptor: PipelineDescriptor, request: PipelineRequest, output_dir: Path) -> dict[str, Any]:
    from biz.checker import Extractor
    from biz.config_loader import backend_config, config
    from biz.result import InspectionResultDto

    dto = InspectionResultDto(descriptor.product, descriptor.project)
    product_id: str | None = None
    reports: list[dict[str, Any]] = []
    overall = True
    engine = _engine(request, descriptor)
    started = perf_counter()
    for index, slot in enumerate(descriptor.slots):
        paths = case.inputs[slot.name]
        images = [_load_image(path, case.color_space) for path in paths]
        _event("stage", case=case.case_id, slot=slot.name, stage="detect", current=index, total=len(descriptor.slots))
        image, result = (
            _detect_service(descriptor, images, index)
            if engine == "service"
            else _detect_local(descriptor, slot.name, images, index)
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"{slot.name} 检测未返回结果对象")
        passed, checks, messages, annotated = _evaluate_and_draw(descriptor, result, image, index)
        overall = overall and passed
        face_conf = (config.inspection.get("conf") or {}).get(slot.name, config.inspection.get("conf") or {})
        extracted, update_required = Extractor.extract_product_id(result, face_conf)
        if extracted and (product_id is None or str(product_id).startswith("unknown_") or update_required):
            product_id = str(extracted)

        annotated_path = output_dir / f"{slot.name}_annotated.png"
        result_path = output_dir / f"{slot.name}_result.json"
        if request.save_annotated_images:
            _save_image(annotated_path, annotated, case.color_space)
        _write_json(result_path, {"passed": passed, "messages": messages, "checks": checks, "result": result})
        dto.filenames.extend(str(path) for path in paths)
        if request.save_annotated_images:
            dto.images["result_image"].append(str(annotated_path))
        dto.result_details.append(json.dumps(_serializable(result), ensure_ascii=False))
        dto.conf.append(json.dumps(_serializable(config.inspection), ensure_ascii=False))
        dto.model_info.append(json.dumps(_serializable(backend_config), ensure_ascii=False))
        for message in messages:
            if message not in dto.err_msgs:
                dto.err_msgs.append(message)
                dto.err_file_map[message] = str(paths[-1])
        reports.append({
            "slot": slot.name,
            "inputs": [str(path) for path in paths],
            "passed": passed,
            "messages": messages,
            "result_json": str(result_path),
            "annotated_image": str(annotated_path) if request.save_annotated_images else None,
        })
        _event("business_result", case=case.case_id, slot=slot.name, outcome="OK" if passed else "NG")

    dto.update_result(product_id, overall, True)
    persisted = False
    if request.persist_db:
        from biz.result_writer import ResultWriter

        persisted = ResultWriter(project=descriptor.project).write(dto)
        if not persisted:
            raise RuntimeError("写入 Cosmos 数据库失败")
    report = {
        "case_id": case.case_id,
        "project": descriptor.project,
        "product": descriptor.product,
        "product_id": product_id,
        "engine": engine,
        "input_color_space": case.color_space,
        "passed": overall,
        "persisted": persisted,
        "execution_time": perf_counter() - started,
        "slots": reports,
        "inspection_result": dto.to_dict(),
    }
    _write_json(output_dir / "case_result.json", report)
    return report


def _execute(request: PipelineRequest) -> int:
    descriptor, cases = validate_request(request)
    _event("describe", descriptor=asdict(descriptor), case_count=len(cases))
    if request.dry_run:
        print("[DRY-RUN] Cosmos 配置、后端节点和输入清单检查通过；未加载模型、未执行外部接口。", flush=True)
        return 0

    request.output_dir.mkdir(parents=True, exist_ok=True)
    session_recorder = None
    if request.onnx_session_report is not None:
        from .onnx_sessions import install_onnx_session_recorder

        session_recorder = install_onnx_session_recorder(request.onnx_session_report)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "request": asdict(request),
        "descriptor": asdict(descriptor),
        "cases": [],
    }
    overall = True
    try:
        _configure_cosmos(request, descriptor)
        if session_recorder is not None:
            session_recorder.snapshot("after_configure")
        for current, case in enumerate(cases, start=1):
            _event("progress", current=current - 1, total=len(cases), case=case.case_id)
            case_dir = request.output_dir / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            report = _run_case(case, descriptor, request, case_dir)
            summary["cases"].append(report)
            overall = overall and bool(report["passed"])
            if session_recorder is not None:
                session_recorder.snapshot(f"after_case:{case.case_id}")
            _event("progress", current=current, total=len(cases), case=case.case_id)
    except Exception as exc:
        overall = False
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        summary["passed"] = overall
        _write_json(request.output_dir / "run_manifest.json", summary)
        if session_recorder is not None:
            session_recorder.snapshot("pipeline_finished")
            session_recorder.write()
    _event("artifact", path=request.output_dir)
    _event("complete", outcome="OK" if overall else "NG")
    return 2 if request.fail_on_ng and not overall else 0


def execute(request: PipelineRequest) -> int:
    """Execute with Cosmos as cwd so every production-relative asset resolves."""
    original_cwd = Path.cwd()
    os.chdir(COSMOS_ROOT)
    try:
        return _execute(request)
    finally:
        os.chdir(original_cwd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete Cosmos inspection flow without production UI/hardware.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend-config", default=str(DEFAULT_BACKEND_CONFIG))
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--engine", choices=("auto", "local", "service"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--persist-db", action="store_true")
    parser.add_argument("--fail-on-ng", action="store_true")
    parser.add_argument("--skip-annotated-images", action="store_true")
    parser.add_argument("--onnx-session-report", type=Path)
    return parser


def request_from_args(args: argparse.Namespace) -> PipelineRequest:
    return PipelineRequest(
        Path(args.config).expanduser().resolve(),
        Path(args.backend_config).expanduser().resolve(),
        Path(args.input_manifest).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        args.engine,
        bool(args.dry_run),
        bool(args.persist_db),
        bool(args.fail_on_ng),
        not bool(args.skip_annotated_images),
        args.onnx_session_report.expanduser().resolve() if args.onnx_session_report else None,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(request_from_args(args))
    except Exception as exc:
        traceback.print_exc()
        _event("error", error_type=type(exc).__name__, message=str(exc))
        return 1
