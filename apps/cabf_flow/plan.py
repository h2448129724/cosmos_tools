"""Pure planning primitives for the CAB-F workflow.

The objects in this module deliberately describe commands instead of running
them.  ``flow.py`` is the imperative adapter which supplies configuration,
performs the annotation-directory fallback and executes the resulting argv.
Keeping path values as strings is intentional: constructing a plan must not
touch the filesystem (or even require that paths exist).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePath


class CabfStepKind(str, Enum):
    VALIDATE = "validate"
    EXPORT_MODEL_A = "export-model-a"
    EXPORT_MODEL_B = "export-model-b"


class CabfPathRole(str, Enum):
    IMAGES = "images"
    MASTER_ANNOTATIONS = "master-annotations"
    REPORT = "report"
    MODEL_A_EXPORT = "model-a-export"
    MODEL_B_EXPORT = "model-b-export"


@dataclass(frozen=True, slots=True)
class BoundPath:
    """A path value and its domain role, without resolving the path."""

    role: CabfPathRole
    value: str


@dataclass(frozen=True, slots=True)
class CabfStepPlan:
    kind: CabfStepKind
    argv: tuple[str, ...]
    cwd: str
    inputs: tuple[BoundPath, ...] = ()
    outputs: tuple[BoundPath, ...] = ()


@dataclass(frozen=True, slots=True)
class CabfWorkflowPlan:
    """Ordered commands for one CAB-F run."""

    steps: tuple[CabfStepPlan, ...]

    def __iter__(self):
        return iter(self.steps)


def _text(value: object) -> str:
    return str(value)


def _script_path(root: object, name: str) -> str:
    """Join a command path without resolving or touching the filesystem."""
    return str(PurePath(_text(root)) / "scripts" / name)


def _paths(*items: tuple[CabfPathRole, object]) -> tuple[BoundPath, ...]:
    return tuple(BoundPath(role, _text(value)) for role, value in items if _text(value))


def build_validate_plan(
    *,
    python_executable: str,
    repo_root: str,
    images_dir: str,
    annotation_dir: str,
    report_path: str = "",
    show_samples: bool = False,
    annotation_role: CabfPathRole = CabfPathRole.MASTER_ANNOTATIONS,
) -> CabfStepPlan:
    argv = [
        _text(python_executable),
        _script_path(repo_root, "cabf_validate.py"),
        "--image_dir",
        _text(images_dir),
        "--annotation_dir",
        _text(annotation_dir),
    ]
    if _text(report_path).strip():
        argv.extend(("--report_json", _text(report_path)))
    if show_samples:
        argv.append("--details")
    return CabfStepPlan(
        CabfStepKind.VALIDATE,
        tuple(argv),
        _text(repo_root),
        inputs=_paths((CabfPathRole.IMAGES, images_dir), (annotation_role, annotation_dir)),
        outputs=_paths((CabfPathRole.REPORT, report_path)),
    )


def _build_export_plan(
    kind: CabfStepKind,
    script_name: str,
    *,
    python_executable: str,
    repo_root: str,
    images_dir: str,
    annotation_dir: str,
    output_dir: str,
    output_role: CabfPathRole,
    annotation_role: CabfPathRole,
) -> CabfStepPlan:
    argv = (
        _text(python_executable),
        _script_path(repo_root, script_name),
        "--image_dir",
        _text(images_dir),
        "--annotation_dir",
        _text(annotation_dir),
        "--output_dir",
        _text(output_dir),
    )
    return CabfStepPlan(
        kind,
        argv,
        _text(repo_root),
        inputs=_paths((CabfPathRole.IMAGES, images_dir), (annotation_role, annotation_dir)),
        outputs=_paths((output_role, output_dir)),
    )


def build_export_model_a_plan(
    *,
    python_executable: str,
    repo_root: str,
    images_dir: str,
    annotation_dir: str,
    output_dir: str,
    annotation_role: CabfPathRole = CabfPathRole.MASTER_ANNOTATIONS,
) -> CabfStepPlan:
    return _build_export_plan(
        CabfStepKind.EXPORT_MODEL_A,
        "cabf_export_model_a.py",
        output_role=CabfPathRole.MODEL_A_EXPORT,
        annotation_role=annotation_role,
        python_executable=python_executable, repo_root=repo_root, images_dir=images_dir,
        annotation_dir=annotation_dir, output_dir=output_dir,
    )


def build_export_model_b_plan(
    *,
    python_executable: str,
    repo_root: str,
    images_dir: str,
    annotation_dir: str,
    output_dir: str,
    annotation_role: CabfPathRole = CabfPathRole.MASTER_ANNOTATIONS,
) -> CabfStepPlan:
    return _build_export_plan(
        CabfStepKind.EXPORT_MODEL_B,
        "cabf_export_model_b.py",
        output_role=CabfPathRole.MODEL_B_EXPORT,
        annotation_role=annotation_role,
        python_executable=python_executable, repo_root=repo_root, images_dir=images_dir,
        annotation_dir=annotation_dir, output_dir=output_dir,
    )


def build_workflow_plan(
    *,
    python_executable: str,
    repo_root: str,
    images_dir: str,
    annotation_dir: str,
    model_a_output: str,
    model_b_output: str,
    report_path: str = "",
    show_samples: bool = False,
) -> CabfWorkflowPlan:
    """Validate hand-reviewed annotations, then export both datasets."""
    common = dict(python_executable=python_executable, repo_root=repo_root, images_dir=images_dir,
                  annotation_dir=annotation_dir)
    return CabfWorkflowPlan((
        build_validate_plan(report_path=report_path, show_samples=show_samples, **common),
        build_export_model_a_plan(output_dir=model_a_output, **common),
        build_export_model_b_plan(output_dir=model_b_output, **common),
    ))


__all__ = [
    "BoundPath",
    "CabfPathRole",
    "CabfStepKind",
    "CabfStepPlan",
    "CabfWorkflowPlan",
    "build_validate_plan",
    "build_export_model_a_plan",
    "build_export_model_b_plan",
    "build_workflow_plan",
]
