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
from cosmos_toolbox.training.cab_f_project import project_entry

from .case_decision import (
    CheckFact,
    SlotDecision,
    aggregate_case,
    collect_failed_results,
    decide_slot,
    select_product_id,
)
from .outcome import business_outcome, cli_exit_code
from .request_plan import (
    InputSlot as _InputSlot,
    PipelineDescriptor as _PipelineDescriptor,
    ProjectFamily,
    backend_key_for_project as _backend_key_for_project,
    describe_product_config,
    plan_input_manifest,
    route_project,
    select_engine,
)
from .runtime_session import CosmosRuntimeSession, build_runtime_session


DEFAULT_BACKEND_CONFIG = COSMOS_ROOT / "assets" / "config" / "backend_config.yaml"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "cosmos_pipeline"

# Private aliases make the moved core types/functions available through the
# historical runner module interface without duplicating their rules.
InputSlot = _InputSlot
PipelineDescriptor = _PipelineDescriptor
backend_key_for_project = _backend_key_for_project


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
    fail_on_ng: bool = True
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


def describe_config(path: str | Path) -> PipelineDescriptor:
    data = _read_yaml(Path(path).expanduser().resolve())
    return describe_product_config(data)


def load_input_manifest(path: str | Path, descriptor: PipelineDescriptor) -> list[InputCase]:
    manifest_path = Path(path).expanduser().resolve()
    data = _read_yaml(manifest_path)
    base = manifest_path.parent
    cases: list[InputCase] = []
    for planned in plan_input_manifest(data, descriptor):
        resolved: dict[str, tuple[Path, ...]] = {}
        for name, values in planned.inputs.items():
            files = tuple(
                (base / value).resolve() if not Path(value).is_absolute() else Path(value).resolve() for value in values
            )
            missing = [str(file) for file in files if not file.is_file()]
            if missing:
                raise FileNotFoundError(f"输入文件不存在：{', '.join(missing)}")
            resolved[name] = files
        cases.append(InputCase(planned.case_id, resolved, planned.color_space))
    return cases


def validate_request(request: PipelineRequest) -> tuple[PipelineDescriptor, list[InputCase]]:
    for label, path in (
        ("产品配置", request.config_path),
        ("后端配置", request.backend_config_path),
        ("输入清单", request.input_manifest_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在：{path}")
    descriptor = describe_config(request.config_path)
    select_engine(request.engine, descriptor)
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


def _configure_cosmos(
    request: PipelineRequest,
    descriptor: PipelineDescriptor,
) -> CosmosRuntimeSession:
    from biz import config_loader

    config_loader.config.load_config(str(request.config_path))
    declared_backend = config_loader.load_config(str(request.backend_config_path))
    # Registry resolution depends on the just-loaded product config, so it is
    # an imperative adapter concern.  Resolve the request's own backend data;
    # never consult the process-global default-backend cache here.
    resolved_backend = config_loader._resolve_registry_paths(deepcopy(declared_backend))
    runtime = build_runtime_session(
        {"inspection": config_loader.config.inspection},
        resolved_backend,
        descriptor.backend_key,
    )
    _validate_backend_model_paths(dict(runtime.backend_subtree))
    # Parent Cosmos algorithms remain compatibility adapters for now and read
    # this global at import time.  Full replacement also initializes an empty
    # lazy cache, fixing first-load custom backend selection.
    config_loader.reload_backend_config(deepcopy(dict(runtime.backend)))
    return runtime


def _validate_backend_model_paths(subtree: dict[str, Any]) -> None:
    """Fail before first inference when model references remain unresolved."""

    def walk(node: Any, location: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child = f"{location}.{key}" if location else str(key)
                is_path_field = isinstance(key, str) and (key == "path" or key.endswith("_path"))
                if is_path_field and isinstance(value, str):
                    if value.startswith("@"):
                        raise ValueError(f"后端模型路径未解析：{child}={value}")
                    path = Path(value).expanduser()
                    if not path.is_absolute():
                        path = COSMOS_ROOT / path
                    if not path.is_file():
                        raise FileNotFoundError(f"后端模型文件不存在：{child}={path}")
                else:
                    walk(value, child)
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{location}[{index}]")

    walk(subtree, "")


def _engine(request: PipelineRequest, descriptor: PipelineDescriptor) -> str:
    return select_engine(request.engine, descriptor)


def _detect_local(
    descriptor: PipelineDescriptor,
    slot: str,
    images: list[Any],
    index: int,
    runtime: CosmosRuntimeSession | None = None,
):
    if runtime is None:
        from biz.config_loader import config

        inspection = config.inspection
    else:
        inspection = runtime.inspection

    conf = deepcopy(inspection.get("conf") or {})
    route = route_project(descriptor.project)
    if route.family is ProjectFamily.CAB:
        from algo.cab_eval import cab_evaluation

        return images[-1], cab_evaluation(images, conf)
    if route.family is ProjectFamily.CAB_F:
        conf["face"] = index
        return project_entry().cab_f_evaluation(images[-1], conf)
    if route.family is ProjectFamily.DAB:
        from algo.dab_eval import dab_evaluation
        from utils import rotate_image

        image = rotate_image(images[0], inspection.get("rotate", 0))
        conf["face"] = index
        return image, dab_evaluation(image, conf)
    if route.family is ProjectFamily.OS_DAB:
        from algo.os_dab_eval import os_dab_evaluation
        from utils import rotate_image

        image = rotate_image(images[0], inspection.get("rotate", 0))
        conf["face"] = index
        result, prepared = os_dab_evaluation(image, conf)
        return prepared, result
    raise AssertionError(f"unhandled project route: {route.family}")


def _detect_service(descriptor: PipelineDescriptor, images: list[Any], index: int):
    from biz.services.algo_service import AlgoService

    route = route_project(descriptor.project)
    if route.service_mode == "images":
        return images[-1], AlgoService.inspect_images(images, index)
    if route.service_mode == "image":
        return images[0], AlgoService.inspect_np_image(images[0], index)
    raise ValueError(f"{descriptor.project} 没有生产算法服务入口，请使用 local/auto")


def _evaluate_and_draw(
    descriptor: PipelineDescriptor,
    result: dict[str, Any],
    image: Any,
    index: int,
    *,
    draw_output: bool = True,
    runtime: CosmosRuntimeSession | None = None,
):
    from biz.checker import Checker, OSDABChecker

    if runtime is None:
        from biz.config_loader import config

        inspection = config.inspection
    else:
        inspection = runtime.inspection

    project = descriptor.project
    route = route_project(project)
    definitions = (inspection.get("ok_checkers") or [])[index] or []
    checks: dict[str, Any] = {}
    check_facts: list[CheckFact] = []
    face = descriptor.slots[index].name
    calibration = result.get("calibration") if route.family is ProjectFamily.OS_DAB else None
    calibration_ok = not (isinstance(calibration, dict) and not calibration.get("ok", True))
    for definition in definitions:
        method = str(definition.get("check_method") or "")
        if route.family is ProjectFamily.CAB_F:
            outcome = project_entry().CABFChecker.evaluate(method, result, inspection.get("conf") or {})
            ok, message = outcome.as_tuple()
            checks[method] = outcome
        elif route.family is ProjectFamily.OS_DAB:
            ok, message = getattr(OSDABChecker, method)(result, (inspection.get("conf") or {}).get(face, {}))
            checks[method] = ok
        elif route.family is ProjectFamily.DAB:
            ok, message = getattr(Checker, method)(result, (inspection.get("conf") or {}).get(face, {}))
            checks[method] = ok
        else:
            ok, message = getattr(Checker, method)(result, inspection.get("conf") or {})
            checks[method] = ok
        check_facts.append(CheckFact(method, bool(ok), str(message or "")))

    failure_facts = collect_failed_results(result)
    slot_decision = decide_slot(
        face,
        (),
        checks=check_facts,
        failures=failure_facts,
        calibration_ok=calibration_ok,
    )
    passed = slot_decision.passed
    messages = list(slot_decision.messages)

    if not draw_output:
        return passed, checks, messages, None

    import numpy as np

    from algo import cab_drawing, dab_drawing, os_dab_drawing
    from algo.utils import adjust

    if route.family is ProjectFamily.CAB_F:
        drawn = project_entry().draw_inspection_result(image, result, inspection.get("conf") or {}, checks)
    elif route.family is ProjectFamily.OS_DAB:
        drawn = os_dab_drawing.draw(image, result, (inspection.get("conf") or {}).get(face, {}), checks)
    elif route.family is ProjectFamily.DAB:
        if result.get("face") == "back" and "origin_anchors" in result:
            image, _ = adjust(image, np.array(result["origin_anchors"]))
            drawn = dab_drawing.draw(image, result, (inspection.get("conf") or {}).get(face, {}), checks)
        elif result.get("face") == "back":
            target = np.stack((image.copy(),) * 3, axis=-1) if getattr(image, "ndim", 0) == 2 else image
            drawn = {"output": target}
        else:
            drawn = dab_drawing.draw(image, result, (inspection.get("conf") or {}).get(face, {}), checks)
    else:
        drawn = cab_drawing.draw(image, result, inspection.get("conf") or {}, checks)
    annotated = drawn.get("output") if isinstance(drawn, dict) else drawn
    return passed, checks, messages, annotated


def _collect_failed_results(node: Any, path: str = "result") -> list[tuple[str, str]]:
    """Compatibility wrapper retaining the legacy tuple-returning helper."""

    return [(failure.path, failure.error) for failure in collect_failed_results(node, path)]


def _legacy_runtime_session(descriptor: PipelineDescriptor) -> CosmosRuntimeSession:
    """Snapshot legacy globals for private-call compatibility tests/callers."""

    from biz.config_loader import backend_config, config

    return CosmosRuntimeSession(
        inspection=config.inspection,
        backend_key=descriptor.backend_key,
        backend={descriptor.backend_key: backend_config.get(descriptor.backend_key, {})},
    )


def _run_case(
    case: InputCase,
    descriptor: PipelineDescriptor,
    request: PipelineRequest,
    output_dir: Path,
    runtime: CosmosRuntimeSession | None = None,
) -> dict[str, Any]:
    from biz.checker import Extractor
    from biz.result import InspectionResultDto

    runtime = runtime or _legacy_runtime_session(descriptor)
    dto = InspectionResultDto(descriptor.product, descriptor.project)
    product_id: str | None = None
    reports: list[dict[str, Any]] = []
    slot_decisions: list[SlotDecision] = []
    engine = _engine(request, descriptor)
    started = perf_counter()
    for index, slot in enumerate(descriptor.slots):
        paths = case.inputs[slot.name]
        images = [_load_image(path, case.color_space) for path in paths]
        _event("stage", case=case.case_id, slot=slot.name, stage="detect", current=index, total=len(descriptor.slots))
        image, result = (
            _detect_service(descriptor, images, index)
            if engine == "service"
            else _detect_local(descriptor, slot.name, images, index, runtime)
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"{slot.name} 检测未返回结果对象")
        passed, checks, messages, annotated = _evaluate_and_draw(
            descriptor,
            result,
            image,
            index,
            draw_output=request.save_annotated_images,
            runtime=runtime,
        )
        face_conf = runtime.face_config(slot.name)
        extracted, update_required = Extractor.extract_product_id(result, face_conf)
        product_id = select_product_id(product_id, extracted, update_required)

        annotated_path = output_dir / f"{slot.name}_annotated.png"
        result_path = output_dir / f"{slot.name}_result.json"
        if request.save_annotated_images:
            _save_image(annotated_path, annotated, case.color_space)
        _write_json(result_path, {"passed": passed, "messages": messages, "checks": checks, "result": result})
        dto.filenames.extend(str(path) for path in paths)
        if request.save_annotated_images:
            dto.images["result_image"].append(str(annotated_path))
        dto.result_details.append(json.dumps(_serializable(result), ensure_ascii=False))
        dto.conf.append(json.dumps(_serializable(runtime.inspection), ensure_ascii=False))
        dto.model_info.append(json.dumps(_serializable(runtime.backend), ensure_ascii=False))
        slot_decisions.append(
            SlotDecision(
                slot=slot.name,
                inputs=tuple(str(path) for path in paths),
                passed=passed,
                messages=tuple(messages),
            )
        )
        reports.append(
            {
                "slot": slot.name,
                "inputs": [str(path) for path in paths],
                "passed": passed,
                "messages": messages,
                "result_json": str(result_path),
                "annotated_image": str(annotated_path) if request.save_annotated_images else None,
            }
        )
        _event(
            "business_result",
            case=case.case_id,
            slot=slot.name,
            outcome=business_outcome(passed).business_label,
        )
        # Full-size line-scan images are hundreds of MiB each.  Release the
        # completed slot before the next slot image is loaded; otherwise the
        # previous source/result/annotation stay live while the next RHS is
        # evaluated and can push a two-face CAB-F run over its memory limit.
        del images, image, result, annotated

    case_decision = aggregate_case(case.case_id, slot_decisions, product_id=product_id)
    overall = case_decision.passed
    error_file_map = dict(case_decision.error_file_map)
    for message in case_decision.error_messages:
        if message not in dto.err_msgs:
            dto.err_msgs.append(message)
            dto.err_file_map[message] = error_file_map.get(message, "")
    dto.update_result(case_decision.product_id, case_decision.passed, True)
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
        "product_id": case_decision.product_id,
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
        runtime = _configure_cosmos(request, descriptor)
        if session_recorder is not None:
            session_recorder.snapshot("after_configure")
        for current, case in enumerate(cases, start=1):
            _event("progress", current=current - 1, total=len(cases), case=case.case_id)
            case_dir = request.output_dir / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            report = _run_case(case, descriptor, request, case_dir, runtime)
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
    outcome = business_outcome(overall)
    _event("artifact", path=request.output_dir)
    _event("complete", outcome=outcome.business_label)
    return cli_exit_code(outcome, fail_on_ng=request.fail_on_ng)


def execute(request: PipelineRequest) -> int:
    """Execute with Cosmos as cwd so every production-relative asset resolves."""
    original_cwd = Path.cwd()
    os.chdir(COSMOS_ROOT)
    try:
        return _execute(request)
    finally:
        os.chdir(original_cwd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete Cosmos inspection flow without production UI/hardware."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend-config", default=str(DEFAULT_BACKEND_CONFIG))
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--engine", choices=("auto", "local", "service"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--persist-db", action="store_true")
    ng_group = parser.add_mutually_exclusive_group()
    ng_group.add_argument("--fail-on-ng", dest="fail_on_ng", action="store_true", help="业务 NG 返回退出码 2（默认）")
    ng_group.add_argument(
        "--allow-ng",
        dest="fail_on_ng",
        action="store_false",
        help="兼容模式：业务 NG 仍返回退出码 0",
    )
    parser.set_defaults(fail_on_ng=True)
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
