"""Opt-in runtime inventory for ONNX Runtime sessions created by a pipeline run."""

from __future__ import annotations

import json
import threading
import time
import weakref
from datetime import datetime
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _model_source(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    value = args[0] if args else kwargs.get("path_or_bytes")
    if isinstance(value, (str, Path)):
        return str(Path(value).expanduser().resolve())
    if isinstance(value, bytes):
        return f"<serialized model: {len(value)} bytes>"
    return repr(value)


class OnnxSessionRecorder:
    """Record actual providers/options without retaining sessions or changing lifecycle."""

    def __init__(self, report_path: Path) -> None:
        self.report_path = report_path
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._references: dict[int, weakref.ReferenceType[Any] | None] = {}
        self._snapshots: list[dict[str, Any]] = []

    def record(self, session: Any, args: tuple[Any, ...], kwargs: dict[str, Any], creation_seconds: float) -> None:
        sequence = len(self._records) + 1
        try:
            reference: weakref.ReferenceType[Any] | None = weakref.ref(session)
        except TypeError:
            reference = None
        try:
            active_providers = session.get_providers()
        except Exception:  # noqa: BLE001 - diagnostics must not affect inference
            active_providers = []
        try:
            active_options = session.get_provider_options()
        except Exception:  # noqa: BLE001 - diagnostics must not affect inference
            active_options = {}
        requested_providers = kwargs.get("providers")
        if requested_providers is None and len(args) >= 3:
            requested_providers = args[2]
        requested_provider_options = kwargs.get("provider_options")
        if requested_provider_options is None and len(args) >= 4:
            requested_provider_options = args[3]
        record = {
            "session_id": sequence,
            "model_path": _model_source(args, kwargs),
            "created_on_thread": threading.current_thread().name,
            "creation_seconds": creation_seconds,
            "requested_providers": _json_safe(requested_providers),
            "requested_provider_options": _json_safe(requested_provider_options),
            "active_providers": _json_safe(active_providers),
            "active_provider_options": _json_safe(active_options),
            "weakref_supported": reference is not None,
        }
        with self._lock:
            self._records.append(record)
            self._references[sequence] = reference

    def snapshot(self, label: str) -> None:
        with self._lock:
            states = {
                str(session_id): reference() is not None if reference is not None else None
                for session_id, reference in self._references.items()
            }
            self._snapshots.append(
                {
                    "label": label,
                    "monotonic": time.monotonic(),
                    "session_alive": states,
                }
            )

    def report(self) -> dict[str, Any]:
        with self._lock:
            records = [dict(item) for item in self._records]
            snapshots = [dict(item) for item in self._snapshots]
        final_states = snapshots[-1]["session_alive"] if snapshots else {}
        for record in records:
            final_alive = final_states.get(str(record["session_id"]))
            record["alive_at_pipeline_end"] = final_alive
            record["lifecycle"] = (
                "retained_through_pipeline"
                if final_alive is True
                else "released_before_pipeline_end"
                if final_alive is False
                else "unknown"
            )
        cuda_sessions = [item for item in records if "CUDAExecutionProvider" in item["active_providers"]]
        explicit_memory_sessions = []
        memory_keys = {
            "gpu_mem_limit",
            "arena_extend_strategy",
            "cudnn_conv_algo_search",
            "cudnn_conv_use_max_workspace",
        }
        for item in cuda_sessions:
            requested = item.get("requested_providers") or []
            inline_options = [entry[1] for entry in requested if isinstance(entry, list) and len(entry) == 2]
            separate_options = item.get("requested_provider_options") or []
            requested_options = [*inline_options, *separate_options]
            if any(
                isinstance(options, dict) and any(key in options for key in memory_keys)
                for options in requested_options
            ):
                explicit_memory_sessions.append(item)
        return {
            "schema_version": 1,
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "session_count": len(records),
            "cuda_session_count": len(cuda_sessions),
            "retained_session_count": sum(item["alive_at_pipeline_end"] is True for item in records),
            "explicit_cuda_memory_policy_count": len(explicit_memory_sessions),
            "sessions": records,
            "snapshots": snapshots,
        }

    def write(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(self.report(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def install_onnx_session_recorder(report_path: Path) -> OnnxSessionRecorder:
    """Wrap ``onnxruntime.InferenceSession`` for this process only."""

    import onnxruntime as ort

    recorder = OnnxSessionRecorder(report_path)
    original = ort.InferenceSession

    def recording_inference_session(*args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        session = original(*args, **kwargs)
        recorder.record(session, args, kwargs, time.perf_counter() - started_at)
        return session

    ort.InferenceSession = recording_inference_session
    return recorder
