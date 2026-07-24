from __future__ import annotations

import copy
from pathlib import Path

import pytest

from apps.cosmos_pipeline.batch_tuner import (
    TuningRequest,
    apply_quality_and_resource_gates,
    build_candidate_config,
    build_repeat_manifest,
    compare_payloads,
    normalize_result,
    parse_positive_ints,
    payload_digest,
    recommend_candidate,
)
from apps.cosmos_pipeline.runner import build_parser as build_pipeline_parser
from apps.cosmos_pipeline.runner import request_from_args as pipeline_request_from_args


def _request(tmp_path: Path, **overrides) -> TuningRequest:
    values = {
        "config_path": tmp_path / "product.yaml",
        "backend_config_path": tmp_path / "backend.yaml",
        "input_manifest_path": tmp_path / "inputs.yaml",
        "output_dir": tmp_path / "output",
        "point_batches": (4, 8),
        "patch_batches": (32, 64),
    }
    values.update(overrides)
    return TuningRequest(**values)


def _candidate(
    candidate_id: str,
    total_seconds: float,
    peak_used_mib: float,
    business_digest: str = "business-a",
    semantic_digest: str = "semantic-a",
    strict_digest: str = "strict-a",
    slowdown: float = 0.0,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "point_batch_size": 8,
        "patch_batch_size": 64,
        "technical_success": True,
        "gpu": {"peak_used_mib": peak_used_mib},
        "pipeline": {
            "median_total_seconds": total_seconds,
            "reference_business_digest": business_digest,
            "reference_semantic_digest": semantic_digest,
            "reference_strict_digest": strict_digest,
            "within_candidate_business_stable": True,
            "within_candidate_semantic_stable": True,
            "steady_state_slowdown": slowdown,
        },
    }


def test_parse_positive_ints_deduplicates_and_preserves_order():
    assert parse_positive_ints("16, 4,16,8") == (16, 4, 8)
    with pytest.raises(ValueError, match="positive"):
        parse_positive_ints("8,0")


def test_build_candidate_config_changes_only_product_hardware_batch_values():
    base = {
        "inspection": {
            "model_params": {
                "sew_point_density": {"batch_size": 16, "patch_batch_size": 64},
            },
            "conf": {"bottom": {"density": {"single_line": True}}},
        }
    }
    original = copy.deepcopy(base)

    candidate = build_candidate_config(base, point_batch=8, patch_batch=128)

    assert base == original
    assert candidate["inspection"]["model_params"]["sew_point_density"] == {
        "batch_size": 8,
        "patch_batch_size": 128,
    }
    assert candidate["inspection"]["conf"]["bottom"]["density"] == {"single_line": True}


def test_build_repeat_manifest_uses_one_fixed_case_without_mutating_source():
    source = {"color_space": "rgb", "cases": [{"id": "real", "inputs": {"top": ["top.png"]}}]}
    original = copy.deepcopy(source)

    result = build_repeat_manifest(source, warmup_cases=1, repeats=2)

    assert source == original
    assert [item["id"] for item in result["cases"]] == ["warmup-001", "measure-001", "measure-002"]
    assert all(item["inputs"] == source["cases"][0]["inputs"] for item in result["cases"])


def test_result_digest_removes_runtime_fields_and_supports_semantic_float_tolerance():
    left = {"execution_time": 2.0, "result": {"score": 0.12345641, "stage_timings": {"x": 1.0}}}
    right = {"execution_time": 9.0, "result": {"score": 0.12345649, "stage_timings": {"x": 8.0}}}

    assert normalize_result(left, 6) == {"result": {"score": 0.123456}}
    assert payload_digest(left, 6) == payload_digest(right, 6)
    assert payload_digest(left, None) != payload_digest(right, None)


def test_tolerant_comparison_requires_identical_structure_and_bounds_float_drift():
    reference = {"result": {"points": [{"x": 10, "score": 0.5331}], "status": True}}
    acceptable = {"result": {"points": [{"x": 10, "score": 0.5344}], "status": True}}
    changed_structure = {"result": {"points": [], "status": True}}
    changed_score = {"result": {"points": [{"x": 10, "score": 0.54}], "status": True}}

    comparison = compare_payloads(reference, acceptable, abs_tolerance=0.002)

    assert comparison["matched"] is True
    assert comparison["max_abs_numeric_difference"] == pytest.approx(0.0013)
    assert compare_payloads(reference, changed_structure, 0.002)["matched"] is False
    assert compare_payloads(reference, changed_score, 0.002)["matched"] is False


def test_12gb_gate_rejects_quality_memory_and_steady_state_failures(tmp_path: Path):
    request = _request(tmp_path, target_vram_mib=12288, reserve_vram_mib=1536, steady_state_tolerance=0.2)
    reference = _candidate("reference", 80.0, 10000)
    quality_bad = _candidate("quality-bad", 70.0, 9000, semantic_digest="semantic-b")
    memory_bad = _candidate("memory-bad", 60.0, 10753)
    steady_bad = _candidate("steady-bad", 50.0, 9000, slowdown=0.21)
    candidates = [reference, quality_bad, memory_bad, steady_bad]

    apply_quality_and_resource_gates(candidates, request)

    assert reference["gates"]["accepted"] is True
    assert quality_bad["gates"]["quality_match"] is False
    assert memory_bad["gates"]["memory_safe"] is False
    assert memory_bad["gates"]["memory_limit_mib"] == 10752
    assert steady_bad["gates"]["steady_state_safe"] is False


def test_recommendation_uses_fastest_accepted_candidate():
    slow = _candidate("slow", 80.0, 9000)
    fast = _candidate("fast", 65.0, 10000)
    rejected = _candidate("rejected", 50.0, 8000)
    slow["gates"] = {"accepted": True}
    fast["gates"] = {"accepted": True}
    rejected["gates"] = {"accepted": False}

    recommendation = recommend_candidate([slow, fast, rejected])

    assert recommendation["status"] == "recommended"
    assert recommendation["candidate_id"] == "fast"
    assert recommendation["product_config_override"]["inspection"]["model_params"]["sew_point_density"] == {
        "batch_size": 8,
        "patch_batch_size": 64,
    }


def test_pipeline_cli_can_skip_images_and_write_runtime_session_report(tmp_path: Path):
    args = build_pipeline_parser().parse_args(
        [
            "--config",
            str(tmp_path / "product.yaml"),
            "--input-manifest",
            str(tmp_path / "inputs.yaml"),
            "--skip-annotated-images",
            "--onnx-session-report",
            str(tmp_path / "sessions.json"),
        ]
    )

    request = pipeline_request_from_args(args)

    assert request.save_annotated_images is False
    assert request.onnx_session_report == (tmp_path / "sessions.json").resolve()
