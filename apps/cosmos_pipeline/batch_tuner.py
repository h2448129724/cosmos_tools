"""Hardware-aware CAB-F batch-size tuner built on the Headless Pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from cosmos_toolbox.paths import COSMOS_ROOT


TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = TOOLBOX_ROOT / "scripts" / "cosmos_pipeline_test.py"
RUNTIME_RESULT_KEYS = {"execution_time", "finish_time", "stage_timings"}
CUDA_MEMORY_OPTION_KEYS = {
    "gpu_mem_limit",
    "arena_extend_strategy",
    "cudnn_conv_algo_search",
    "cudnn_conv_use_max_workspace",
    "do_copy_in_default_stream",
}


@dataclass(frozen=True, slots=True)
class TuningRequest:
    config_path: Path
    backend_config_path: Path
    input_manifest_path: Path
    output_dir: Path
    point_batches: tuple[int, ...]
    patch_batches: tuple[int, ...]
    warmup_cases: int = 1
    repeats: int = 3
    target_vram_mib: int = 12288
    reserve_vram_mib: int = 1536
    gpu_index: int = 0
    sample_interval: float = 0.25
    timeout_per_candidate: float = 1200.0
    semantic_float_digits: int = 6
    semantic_abs_tolerance: float = 0.002
    steady_state_tolerance: float = 0.20
    ort_cuda_profile: str = "default"
    quality_reference_report: Path | None = None
    resume: bool = False


def parse_positive_ints(value: str | Iterable[int]) -> tuple[int, ...]:
    raw_values = value.split(",") if isinstance(value, str) else value
    values = tuple(dict.fromkeys(int(item) for item in raw_values))
    if not values or any(item <= 0 for item in values):
        raise ValueError("batch sizes must contain positive integers")
    return values


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def build_candidate_config(base: dict[str, Any], point_batch: int, patch_batch: int) -> dict[str, Any]:
    candidate = copy.deepcopy(base)
    try:
        inspection = candidate["inspection"]
    except (KeyError, TypeError) as exc:
        raise ValueError("product config is missing inspection") from exc
    if not isinstance(inspection, dict):
        raise ValueError("inspection must be a mapping")
    model_params = inspection.setdefault("model_params", {})
    if not isinstance(model_params, dict):
        raise ValueError("inspection.model_params must be a mapping")
    density = model_params.setdefault("sew_point_density", {})
    if not isinstance(density, dict):
        raise ValueError("inspection.model_params.sew_point_density must be a mapping")
    density["batch_size"] = int(point_batch)
    density["patch_batch_size"] = int(patch_batch)
    return candidate


def build_repeat_manifest(source: dict[str, Any], warmup_cases: int, repeats: int) -> dict[str, Any]:
    raw_cases = source.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("input manifest must contain at least one case")
    if warmup_cases < 0 or repeats < 1:
        raise ValueError("warmup_cases must be >= 0 and repeats must be >= 1")
    template = raw_cases[0]
    cases = []
    for index in range(warmup_cases):
        item = copy.deepcopy(template)
        item["id"] = f"warmup-{index + 1:03d}"
        cases.append(item)
    for index in range(repeats):
        item = copy.deepcopy(template)
        item["id"] = f"measure-{index + 1:03d}"
        cases.append(item)
    result = {key: copy.deepcopy(value) for key, value in source.items() if key != "cases"}
    result["cases"] = cases
    return result


def normalize_result(value: Any, float_digits: int | None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): normalize_result(item, float_digits)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if key not in RUNTIME_RESULT_KEYS
        }
    if isinstance(value, list):
        return [normalize_result(item, float_digits) for item in value]
    if isinstance(value, float) and float_digits is not None:
        return round(value, int(float_digits))
    return value


def payload_digest(value: Any, float_digits: int | None) -> str:
    normalized = normalize_result(value, float_digits)
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compare_payloads(reference: Any, candidate: Any, abs_tolerance: float) -> dict[str, Any]:
    """Compare result structures exactly except for bounded floating-point drift."""

    mismatch_examples: list[dict[str, Any]] = []
    numeric_difference_count = 0
    max_abs_numeric_difference = 0.0

    def mismatch(path: str, reason: str, left: Any, right: Any) -> None:
        if len(mismatch_examples) < 20:
            mismatch_examples.append(
                {"path": path or "$", "reason": reason, "reference": left, "candidate": right}
            )

    def walk(left: Any, right: Any, path: str = "") -> None:
        nonlocal max_abs_numeric_difference, numeric_difference_count
        if isinstance(left, dict) and isinstance(right, dict):
            left_keys = {key for key in left if key not in RUNTIME_RESULT_KEYS}
            right_keys = {key for key in right if key not in RUNTIME_RESULT_KEYS}
            if left_keys != right_keys:
                mismatch(path, "mapping_keys", sorted(map(str, left_keys)), sorted(map(str, right_keys)))
                return
            for key in sorted(left_keys, key=str):
                walk(left[key], right[key], f"{path}.{key}" if path else str(key))
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatch(path, "list_length", len(left), len(right))
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                walk(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(left, bool) or isinstance(right, bool):
            if left != right or type(left) is not type(right):
                mismatch(path, "value", left, right)
            return
        if isinstance(left, int) and isinstance(right, int):
            if left != right:
                mismatch(path, "integer", left, right)
            return
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            difference = abs(float(left) - float(right))
            if difference > 0:
                numeric_difference_count += 1
                max_abs_numeric_difference = max(max_abs_numeric_difference, difference)
            if not math.isfinite(difference) or difference > abs_tolerance:
                mismatch(path, "float_tolerance", left, right)
            return
        if left != right or type(left) is not type(right):
            mismatch(path, "value", left, right)

    walk(reference, candidate)
    return {
        "matched": not mismatch_examples,
        "abs_tolerance": abs_tolerance,
        "numeric_difference_count": numeric_difference_count,
        "max_abs_numeric_difference": max_abs_numeric_difference,
        "mismatch_examples": mismatch_examples,
    }


def compare_case_outputs(
    reference_case: dict[str, Any], candidate_case: dict[str, Any], abs_tolerance: float
) -> dict[str, Any]:
    reference_paths = reference_case.get("result_jsons") or {}
    candidate_paths = candidate_case.get("result_jsons") or {}
    if set(reference_paths) != set(candidate_paths):
        return {
            "matched": False,
            "abs_tolerance": abs_tolerance,
            "numeric_difference_count": 0,
            "max_abs_numeric_difference": 0.0,
            "mismatch_examples": [
                {
                    "path": "$slots",
                    "reason": "slot_names",
                    "reference": sorted(reference_paths),
                    "candidate": sorted(candidate_paths),
                }
            ],
        }
    comparisons = []
    for slot in sorted(reference_paths):
        reference_payload = json.loads(Path(reference_paths[slot]).read_text(encoding="utf-8"))
        candidate_payload = json.loads(Path(candidate_paths[slot]).read_text(encoding="utf-8"))
        comparison = compare_payloads(reference_payload, candidate_payload, abs_tolerance)
        comparison["slot"] = slot
        comparisons.append(comparison)
    return {
        "matched": all(item["matched"] for item in comparisons),
        "abs_tolerance": abs_tolerance,
        "numeric_difference_count": sum(item["numeric_difference_count"] for item in comparisons),
        "max_abs_numeric_difference": max(
            (item["max_abs_numeric_difference"] for item in comparisons), default=0.0
        ),
        "mismatch_examples": [
            {**example, "slot": item["slot"]}
            for item in comparisons
            for example in item["mismatch_examples"]
        ][:20],
    }


def _shape_from_value_info(value_info: Any) -> list[int | str]:
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(int(dim.dim_value))
        else:
            shape.append(str(dim.dim_param or "?"))
    return shape


def _walk_onnx_paths(node: Any, prefix: tuple[str, ...] = ()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_onnx_paths(value, (*prefix, str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_onnx_paths(value, (*prefix, str(index)))
    elif isinstance(node, str) and node.lower().split("#", 1)[0].endswith(".onnx"):
        yield prefix, Path(node)


def _batch_role(path: Path, usages: list[str], inputs: list[dict[str, Any]]) -> str:
    name = path.name.lower()
    if name == "sew_point_detector.onnx" or any(
        usage.endswith(("sew_point_density.point_model_path", "sew_point_detector.path")) for usage in usages
    ):
        return "density.point_detection"
    if name == "sew_point_connector_patch.onnx" or any(
        usage.endswith("sew_point_density.patch_model_path") for usage in usages
    ):
        return "density.patch_encode"
    if ("glue" in name or any(usage.endswith("glue_segment.path") for usage in usages)) and any(
        item.get("shape", [None])[0] == "batch" for item in inputs
    ):
        return "calibration.glue_segmentation"
    if name == "error_detector.onnx" or any(usage.endswith("error_detector.path") for usage in usages):
        return "detector.error_detector (static batch=1; re-export required)"
    return "not exposed by this tuner"


def _nested_value(node: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def inventory_onnx_models(
    resolved_backend: dict[str, Any], declared_backend: dict[str, Any] | None = None, backend_key: str = "cab_f"
) -> list[dict[str, Any]]:
    try:
        resolved_backend[backend_key]
    except KeyError as exc:
        raise ValueError(f"resolved backend is missing {backend_key}") from exc
    inventory_source = declared_backend if declared_backend is not None else resolved_backend
    source_subtree = inventory_source.get(backend_key)
    if not isinstance(source_subtree, dict):
        raise ValueError(f"declared backend is missing {backend_key}")
    grouped: dict[Path, dict[str, Any]] = {}
    for path_parts, declared_path in _walk_onnx_paths(source_subtree, (backend_key,)):
        resolved_path = Path(_nested_value(resolved_backend, path_parts)).resolve()
        entry = grouped.setdefault(resolved_path, {"usages": [], "declared_paths": []})
        entry["usages"].append(".".join(path_parts))
        entry["declared_paths"].append(str(declared_path))

    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - target runtime includes onnx
        raise RuntimeError("onnx is required for session inventory") from exc

    inventory = []
    for model_path, model_entry in sorted(grouped.items(), key=lambda item: str(item[0])):
        usages = model_entry["usages"]
        inputs: list[dict[str, Any]] = []
        error = None
        if model_path.is_file():
            try:
                model = onnx.load(str(model_path), load_external_data=False)
                inputs = [{"name": item.name, "shape": _shape_from_value_info(item)} for item in model.graph.input]
            except Exception as exc:  # noqa: BLE001 - inventory records model-specific failures
                error = f"{type(exc).__name__}: {exc}"
        else:
            error = "model file does not exist"

        first_dimension = inputs[0]["shape"][0] if inputs and inputs[0]["shape"] else None
        dynamic_first_dimension = isinstance(first_dimension, str)
        inventory.append(
            {
                "path": str(model_path),
                "artifact_id": model_path.name,
                "declared_paths": sorted(model_entry["declared_paths"]),
                "usages": sorted(usages),
                "size_mib": round(model_path.stat().st_size / 1024 / 1024, 3) if model_path.is_file() else None,
                "inputs": inputs,
                "dynamic_first_dimension": dynamic_first_dimension,
                "batch_role": _batch_role(model_path, usages, inputs),
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "provider_options": {},
                "memory_policy": "CUDA EP default arena (no explicit aggregate budget)",
                "lifecycle": "lazy process singleton; retained until process exit",
                "error": error,
            }
        )
    return inventory


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def query_gpu(gpu_index: int = 0) -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        f"--id={int(gpu_index)}",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
            creationflags=_creation_flags(),
        )
        values = [item.strip() for item in completed.stdout.strip().split(",")]
        if len(values) != 8:
            return None
        return {
            "name": values[0],
            "memory_total_mib": float(values[1]),
            "memory_used_mib": float(values[2]),
            "memory_free_mib": float(values[3]),
            "gpu_utilization_pct": float(values[4]),
            "memory_utilization_pct": float(values[5]),
            "power_w": float(values[6]),
            "driver_version": values[7],
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class GpuSampler:
    def __init__(self, gpu_index: int, interval: float) -> None:
        self.gpu_index = int(gpu_index)
        self.interval = max(0.1, float(interval))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = query_gpu(self.gpu_index)
            if sample is not None:
                sample["monotonic"] = time.monotonic()
                self.samples.append(sample)
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="cabf-batch-tuner-gpu", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if not self.samples:
            return {"sample_count": 0}
        return {
            "sample_count": len(self.samples),
            "peak_used_mib": max(item["memory_used_mib"] for item in self.samples),
            "minimum_free_mib": min(item["memory_free_mib"] for item in self.samples),
            "mean_gpu_utilization_pct": statistics.fmean(item["gpu_utilization_pct"] for item in self.samples),
            "max_gpu_utilization_pct": max(item["gpu_utilization_pct"] for item in self.samples),
            "max_power_w": max(item["power_w"] for item in self.samples),
        }


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def _read_case_metrics(case: dict[str, Any], float_digits: int) -> dict[str, Any]:
    slot_metrics = {}
    strict_payload = []
    semantic_payload = []
    result_jsons = {}
    for slot in case.get("slots", []):
        result_path = Path(slot["result_json"])
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = payload.get("result") or {}
        timings = result.get("stage_timings") or {}
        slot_name = str(slot["slot"])
        result_jsons[slot_name] = str(result_path)
        slot_metrics[slot_name] = {
            "passed": bool(slot.get("passed")),
            "messages": slot.get("messages") or [],
            "execution_time": result.get("execution_time"),
            "stage_timings": timings,
        }
        strict_payload.append((slot_name, payload_digest(payload, None)))
        semantic_payload.append((slot_name, payload_digest(payload, float_digits)))
    business = {
        "passed": bool(case.get("passed")),
        "product_id": case.get("product_id"),
        "slots": [
            (str(slot.get("slot")), bool(slot.get("passed")), slot.get("messages") or [])
            for slot in case.get("slots", [])
        ],
    }
    return {
        "case_id": case.get("case_id"),
        "execution_time": float(case.get("execution_time", 0.0)),
        "slots": slot_metrics,
        "business_digest": payload_digest(business, None),
        "strict_digest": payload_digest(strict_payload, None),
        "semantic_digest": payload_digest(semantic_payload, None),
        "result_jsons": result_jsons,
    }


def summarize_pipeline_output(
    output_dir: Path, warmup_cases: int, repeats: int, float_digits: int, abs_tolerance: float
) -> dict[str, Any]:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"pipeline manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [_read_case_metrics(case, float_digits) for case in manifest.get("cases", [])]
    measured = cases[warmup_cases : warmup_cases + repeats]
    if len(measured) != repeats:
        raise RuntimeError(f"expected {repeats} measured cases, got {len(measured)}")

    def slot_value(case: dict[str, Any], slot: str, key: str) -> float | None:
        value = (case.get("slots", {}).get(slot) or {}).get(key)
        return float(value) if isinstance(value, (int, float)) else None

    def stage_value(case: dict[str, Any], slot: str, stage: str) -> float | None:
        value = ((case.get("slots", {}).get(slot) or {}).get("stage_timings") or {}).get(stage)
        return float(value) if isinstance(value, (int, float)) else None

    totals = [case["execution_time"] for case in measured]
    first_total = totals[0]
    last_total = totals[-1]
    within_candidate_comparisons = [
        compare_case_outputs(measured[0], case, abs_tolerance) for case in measured[1:]
    ]
    return {
        "warmup_case_count": warmup_cases,
        "measured_case_count": len(measured),
        "case_metrics": measured,
        "median_total_seconds": _median(totals),
        "median_top_seconds": _median(
            value for case in measured if (value := slot_value(case, "top", "execution_time")) is not None
        ),
        "median_bottom_seconds": _median(
            value for case in measured if (value := slot_value(case, "bottom", "execution_time")) is not None
        ),
        "median_density_seconds": _median(
            value for case in measured if (value := stage_value(case, "bottom", "detector.density")) is not None
        ),
        "median_point_detection_seconds": _median(
            value
            for case in measured
            if (value := stage_value(case, "bottom", "density.point_detection")) is not None
        ),
        "median_patch_encode_seconds": _median(
            value for case in measured if (value := stage_value(case, "bottom", "density.patch_encode")) is not None
        ),
        "steady_state_slowdown": (last_total / first_total - 1.0) if first_total > 0 else math.inf,
        "within_candidate_business_stable": len({case["business_digest"] for case in measured}) == 1,
        "within_candidate_semantic_stable": all(item["matched"] for item in within_candidate_comparisons),
        "within_candidate_semantic_comparisons": within_candidate_comparisons,
        "within_candidate_strict_stable": len({case["strict_digest"] for case in measured}) == 1,
        "reference_business_digest": measured[0]["business_digest"],
        "reference_semantic_digest": measured[0]["semantic_digest"],
        "reference_strict_digest": measured[0]["strict_digest"],
    }


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _enrich_result_json_paths(candidate: dict[str, Any]) -> None:
    """Backfill artifact paths for reports produced before tolerant comparison was added."""

    pipeline_output = candidate.get("pipeline_output")
    for case in (candidate.get("pipeline") or {}).get("case_metrics") or []:
        if case.get("result_jsons") or not pipeline_output or not case.get("case_id"):
            continue
        case_root = Path(pipeline_output) / str(case["case_id"])
        case["result_jsons"] = {
            slot: str(case_root / f"{slot}_result.json") for slot in (case.get("slots") or {})
        }


def run_candidate(
    request: TuningRequest,
    base_config: dict[str, Any],
    repeat_manifest: dict[str, Any],
    point_batch: int,
    patch_batch: int,
) -> dict[str, Any]:
    candidate_id = f"point-{point_batch}_patch-{patch_batch}"
    candidate_root = request.output_dir / "candidates" / candidate_id
    summary_path = candidate_root / "candidate_summary.json"
    if request.resume and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stored_profile = summary.get("ort_cuda_profile", "default")
        if stored_profile != request.ort_cuda_profile:
            raise ValueError(
                f"cannot resume {candidate_id}: stored ORT profile is {stored_profile!r}, "
                f"requested {request.ort_cuda_profile!r}"
            )
        _enrich_result_json_paths(summary)
        return summary

    candidate_root.mkdir(parents=True, exist_ok=True)
    config_path = candidate_root / "product_config.yaml"
    manifest_path = candidate_root / "repeat_manifest.yaml"
    pipeline_output = candidate_root / "pipeline"
    session_report_path = candidate_root / "onnx_sessions.json"
    _write_yaml(config_path, build_candidate_config(base_config, point_batch, patch_batch))
    _write_yaml(manifest_path, repeat_manifest)

    command = [
        sys.executable,
        "-u",
        str(PIPELINE_SCRIPT),
        "--config",
        str(config_path),
        "--backend-config",
        str(request.backend_config_path),
        "--input-manifest",
        str(manifest_path),
        "--output-dir",
        str(pipeline_output),
        "--engine",
        "local",
        "--skip-annotated-images",
        "--onnx-session-report",
        str(session_report_path),
    ]
    gpu_before = query_gpu(request.gpu_index)
    sampler = GpuSampler(request.gpu_index, request.sample_interval)
    started_at = time.perf_counter()
    timed_out = False
    returncode = 1
    error = None
    log_path = candidate_root / "pipeline.log"
    child_env = {**os.environ, "PYTHONUTF8": "1"}
    if request.ort_cuda_profile == "default":
        child_env.pop("COSMOS_ORT_CUDA_PROFILE", None)
    else:
        child_env["COSMOS_ORT_CUDA_PROFILE"] = request.ort_cuda_profile
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            cwd=TOOLBOX_ROOT,
            env=child_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            creationflags=_creation_flags(),
        )
        sampler.start()
        try:
            returncode = process.wait(timeout=request.timeout_per_candidate)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        finally:
            gpu_metrics = sampler.stop()

    pipeline_summary = None
    if not timed_out and returncode == 0:
        try:
            pipeline_summary = summarize_pipeline_output(
                pipeline_output,
                warmup_cases=request.warmup_cases,
                repeats=request.repeats,
                float_digits=request.semantic_float_digits,
                abs_tolerance=request.semantic_abs_tolerance,
            )
        except Exception as exc:  # noqa: BLE001 - candidate failure belongs in the report
            error = f"{type(exc).__name__}: {exc}"
    elif timed_out:
        error = f"candidate timed out after {request.timeout_per_candidate:.0f}s"
    else:
        error = f"pipeline exited with code {returncode}; see {log_path}"

    summary = {
        "candidate_id": candidate_id,
        "point_batch_size": int(point_batch),
        "patch_batch_size": int(patch_batch),
        "command": command,
        "duration_seconds": time.perf_counter() - started_at,
        "returncode": returncode,
        "timed_out": timed_out,
        "technical_success": pipeline_summary is not None,
        "error": error,
        "gpu_before": gpu_before,
        "gpu": gpu_metrics,
        "pipeline": pipeline_summary,
        "log_path": str(log_path),
        "pipeline_output": str(pipeline_output),
        "onnx_session_report": str(session_report_path) if session_report_path.is_file() else None,
        "ort_cuda_profile": request.ort_cuda_profile,
    }
    _write_json(summary_path, summary)
    return summary


def annotate_quality_comparisons(
    candidates: list[dict[str, Any]], request: TuningRequest, external_reference: dict[str, Any] | None = None
) -> None:
    reference = external_reference or next((item for item in candidates if item.get("technical_success")), None)
    if reference is None:
        return
    reference_cases = (reference.get("pipeline") or {}).get("case_metrics") or []
    if not reference_cases:
        return
    reference_case = reference_cases[0]
    for candidate in candidates:
        candidate_cases = (candidate.get("pipeline") or {}).get("case_metrics") or []
        comparisons = [
            compare_case_outputs(reference_case, case, request.semantic_abs_tolerance) for case in candidate_cases
        ]
        candidate["quality_comparison"] = {
            "matched": bool(comparisons) and all(item["matched"] for item in comparisons),
            "abs_tolerance": request.semantic_abs_tolerance,
            "numeric_difference_count": sum(item["numeric_difference_count"] for item in comparisons),
            "max_abs_numeric_difference": max(
                (item["max_abs_numeric_difference"] for item in comparisons), default=0.0
            ),
            "mismatch_examples": [
                example for item in comparisons for example in item["mismatch_examples"]
            ][:20],
        }


def apply_quality_and_resource_gates(
    candidates: list[dict[str, Any]], request: TuningRequest, external_reference: dict[str, Any] | None = None
) -> None:
    reference = external_reference or next((item for item in candidates if item.get("technical_success")), None)
    reference_pipeline = reference.get("pipeline") if reference else None
    memory_limit = request.target_vram_mib - request.reserve_vram_mib
    for candidate in candidates:
        pipeline = candidate.get("pipeline") or {}
        tolerant_comparison = candidate.get("quality_comparison")
        semantic_match = (
            bool(tolerant_comparison.get("matched"))
            if isinstance(tolerant_comparison, dict)
            else pipeline.get("reference_semantic_digest") == reference_pipeline.get("reference_semantic_digest")
            if reference_pipeline
            else False
        )
        quality_match = bool(
            reference_pipeline
            and pipeline.get("reference_business_digest") == reference_pipeline.get("reference_business_digest")
            and semantic_match
        )
        strict_match = bool(
            reference_pipeline
            and pipeline.get("reference_strict_digest") == reference_pipeline.get("reference_strict_digest")
        )
        peak_used = (candidate.get("gpu") or {}).get("peak_used_mib")
        memory_safe = isinstance(peak_used, (int, float)) and peak_used <= memory_limit
        stable = bool(
            pipeline
            and pipeline.get("within_candidate_business_stable")
            and pipeline.get("within_candidate_semantic_stable")
            and float(pipeline.get("steady_state_slowdown", math.inf)) <= request.steady_state_tolerance
        )
        candidate["gates"] = {
            "quality_match": quality_match,
            "strict_result_match": strict_match,
            "memory_safe": memory_safe,
            "steady_state_safe": stable,
            "memory_limit_mib": memory_limit,
            "accepted": bool(candidate.get("technical_success") and quality_match and memory_safe and stable),
        }


def recommend_candidate(candidates: list[dict[str, Any]], ort_cuda_profile: str = "default") -> dict[str, Any]:
    accepted = [item for item in candidates if (item.get("gates") or {}).get("accepted")]
    if not accepted:
        return {
            "status": "no_safe_candidate",
            "reason": "No candidate passed technical, quality, steady-state, and VRAM gates.",
            "product_config_override": None,
        }
    best = min(accepted, key=lambda item: float((item.get("pipeline") or {})["median_total_seconds"]))
    return {
        "status": "recommended",
        "candidate_id": best["candidate_id"],
        "median_total_seconds": best["pipeline"]["median_total_seconds"],
        "peak_used_mib": best["gpu"]["peak_used_mib"],
        "ort_cuda_profile": ort_cuda_profile,
        "product_config_override": {
            "inspection": {
                "model_params": {
                    "sew_point_density": {
                        "batch_size": best["point_batch_size"],
                        "patch_batch_size": best["patch_batch_size"],
                    }
                }
            }
        },
    }


def _load_resolved_backend(path: Path) -> dict[str, Any]:
    from biz import config_loader

    original_cwd = Path.cwd()
    os.chdir(COSMOS_ROOT)
    try:
        return config_loader._resolve_registry_paths(config_loader.load_config(str(path)))
    finally:
        os.chdir(original_cwd)


def _load_quality_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    reference = next(
        (item for item in report.get("candidates") or [] if item.get("technical_success")),
        None,
    )
    if reference is None:
        raise ValueError(f"quality reference report has no successful candidate: {path}")
    _enrich_result_json_paths(reference)
    return reference


def run_tuning(request: TuningRequest) -> dict[str, Any]:
    import onnxruntime as ort

    request.output_dir.mkdir(parents=True, exist_ok=True)
    base_config = _read_yaml(request.config_path)
    base_backend = _read_yaml(request.backend_config_path)
    source_manifest = _read_yaml(request.input_manifest_path)
    repeat_manifest = build_repeat_manifest(source_manifest, request.warmup_cases, request.repeats)
    resolved_backend = _load_resolved_backend(request.backend_config_path)
    inventory = inventory_onnx_models(resolved_backend, base_backend)
    hardware = query_gpu(request.gpu_index)
    quality_reference = _load_quality_reference(request.quality_reference_report)

    report = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "request": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(request).items()},
        "hardware": hardware,
        "runtime": {"python": sys.version, "onnxruntime": ort.__version__},
        "session_inventory": inventory,
        "runtime_session_inventory": None,
        "session_findings": {
            "explicit_cuda_memory_policy_count": sum(
                1
                for item in inventory
                if any(key in (item.get("provider_options") or {}) for key in CUDA_MEMORY_OPTION_KEYS)
            ),
            "default_cuda_arena_count": sum(
                1 for item in inventory if not item.get("provider_options") and item.get("error") is None
            ),
            "lifecycle": "CAB-F model instances are retained for the process lifetime; candidate isolation is required.",
        },
        "candidates": [],
        "recommendation": None,
        "quality_reference": (
            {
                "report": str(request.quality_reference_report),
                "candidate_id": quality_reference.get("candidate_id"),
            }
            if quality_reference is not None
            else {"report": None, "candidate_id": None, "mode": "first successful candidate"}
        ),
    }
    _write_json(request.output_dir / "tuning_report.partial.json", report)

    for point_batch in request.point_batches:
        for patch_batch in request.patch_batches:
            candidate = run_candidate(
                request,
                base_config=base_config,
                repeat_manifest=repeat_manifest,
                point_batch=point_batch,
                patch_batch=patch_batch,
            )
            report["candidates"].append(candidate)
            _write_json(request.output_dir / "tuning_report.partial.json", report)

    runtime_report_path = next(
        (
            Path(item["onnx_session_report"])
            for item in report["candidates"]
            if item.get("onnx_session_report") and Path(item["onnx_session_report"]).is_file()
        ),
        None,
    )
    if runtime_report_path is not None:
        report["runtime_session_inventory"] = json.loads(runtime_report_path.read_text(encoding="utf-8"))
        runtime_inventory = report["runtime_session_inventory"]
        usages_by_path = {str(Path(item["path"]).resolve()): item["usages"] for item in inventory}
        for session in runtime_inventory.get("sessions") or []:
            model_path = session.get("model_path")
            session["usages"] = usages_by_path.get(str(Path(model_path).resolve()), []) if model_path else []
        report["session_findings"].update(
            {
                "runtime_session_count": runtime_inventory.get("session_count"),
                "runtime_cuda_session_count": runtime_inventory.get("cuda_session_count"),
                "runtime_retained_session_count": runtime_inventory.get("retained_session_count"),
                "runtime_explicit_cuda_memory_policy_count": runtime_inventory.get(
                    "explicit_cuda_memory_policy_count"
                ),
            }
        )

    annotate_quality_comparisons(report["candidates"], request, quality_reference)
    apply_quality_and_resource_gates(report["candidates"], request, quality_reference)
    report["recommendation"] = recommend_candidate(report["candidates"], request.ort_cuda_profile)
    report_path = request.output_dir / "tuning_report.json"
    _write_json(report_path, report)
    if report["recommendation"].get("product_config_override") is not None:
        _write_yaml(
            request.output_dir / "recommended_product_override.yaml",
            report["recommendation"]["product_config_override"],
        )
        _write_yaml(
            request.output_dir / "recommended_hardware_profile.yaml",
            {
                "schema_version": 1,
                "hardware": report["hardware"],
                "runtime": report["runtime"],
                "models": [
                    {
                        "declared_paths": item["declared_paths"],
                        "artifact_id": item["artifact_id"],
                        "size_mib": item["size_mib"],
                    }
                    for item in inventory
                ],
                "ort_cuda_profile": request.ort_cuda_profile,
                "product_config_override": report["recommendation"]["product_config_override"],
            },
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-batches", default="4,8,12,16")
    parser.add_argument("--patch-batches", default="32,64,128")
    parser.add_argument("--warmup-cases", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target-vram-mib", type=int, default=12288)
    parser.add_argument("--reserve-vram-mib", type=int, default=1536)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--timeout-per-candidate", type=float, default=1200.0)
    parser.add_argument("--semantic-float-digits", type=int, default=6)
    parser.add_argument("--semantic-abs-tolerance", type=float, default=0.002)
    parser.add_argument("--steady-state-tolerance", type=float, default=0.20)
    parser.add_argument("--ort-cuda-profile", choices=("default", "memory_12gb"), default="default")
    parser.add_argument("--quality-reference-report", type=Path)
    parser.add_argument("--fail-if-no-safe-candidate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def request_from_args(args: argparse.Namespace) -> TuningRequest:
    if args.target_vram_mib <= args.reserve_vram_mib:
        raise ValueError("target VRAM must be greater than reserved VRAM")
    if args.semantic_abs_tolerance < 0:
        raise ValueError("semantic absolute tolerance must be >= 0")
    return TuningRequest(
        config_path=args.config.expanduser().resolve(),
        backend_config_path=args.backend_config.expanduser().resolve(),
        input_manifest_path=args.input_manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        point_batches=parse_positive_ints(args.point_batches),
        patch_batches=parse_positive_ints(args.patch_batches),
        warmup_cases=args.warmup_cases,
        repeats=args.repeats,
        target_vram_mib=args.target_vram_mib,
        reserve_vram_mib=args.reserve_vram_mib,
        gpu_index=args.gpu_index,
        sample_interval=args.sample_interval,
        timeout_per_candidate=args.timeout_per_candidate,
        semantic_float_digits=args.semantic_float_digits,
        semantic_abs_tolerance=args.semantic_abs_tolerance,
        steady_state_tolerance=args.steady_state_tolerance,
        ort_cuda_profile=args.ort_cuda_profile,
        quality_reference_report=(
            args.quality_reference_report.expanduser().resolve() if args.quality_reference_report else None
        ),
        resume=bool(args.resume),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = request_from_args(args)
    paths = [request.config_path, request.backend_config_path, request.input_manifest_path]
    if request.quality_reference_report is not None:
        paths.append(request.quality_reference_report)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    report = run_tuning(request)
    recommendation = report["recommendation"]
    print(json.dumps(recommendation, ensure_ascii=False, indent=2))
    print(f"Report: {request.output_dir / 'tuning_report.json'}")
    if recommendation.get("status") != "recommended" and args.fail_if_no_safe_candidate:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
