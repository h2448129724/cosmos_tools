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
    PREDICT_POINTS = "predict-points"
    PREDICT_EDGES = "predict-edges"
    VALIDATE = "validate"
    EXPORT_MODEL_A = "export-model-a"
    EXPORT_MODEL_B = "export-model-b"
    TRAIN_MODEL_A = "train-model-a"
    TRAIN_MODEL_B = "train-model-b"


class CabfPathRole(str, Enum):
    IMAGES = "images"
    MASTER_ANNOTATIONS = "master-annotations"
    POINT_ANNOTATIONS = "point-annotations"
    EDGE_ANNOTATIONS = "edge-annotations"
    MODEL = "model"
    REPORT = "report"
    OUTPUT = "output"
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


def build_predict_points_plan(
    *,
    python_executable: str,
    modules_root: str,
    images_dir: str,
    output_dir: str,
    model: str = "",
    threshold: float = 0.3,
    distance_threshold: float = 0.0,
) -> CabfStepPlan:
    argv = [
        _text(python_executable),
        "-m",
        "sew_point.tools.batch_infer",
        "--input_dir",
        _text(images_dir),
        "--output_dir",
        _text(output_dir),
        "--threshold",
        _text(threshold),
        "--output_format",
        "master",
    ]
    if float(distance_threshold) > 0:
        argv.extend(("--cluster_dist", _text(distance_threshold)))
    if _text(model).strip():
        argv.extend(("--model", _text(model)))
    return CabfStepPlan(
        CabfStepKind.PREDICT_POINTS,
        tuple(argv),
        _text(modules_root),
        inputs=_paths((CabfPathRole.IMAGES, images_dir), (CabfPathRole.MODEL, model)),
        outputs=_paths((CabfPathRole.POINT_ANNOTATIONS, output_dir)),
    )


def build_predict_edges_plan(
    *,
    python_executable: str,
    modules_root: str,
    images_dir: str,
    annotation_dir: str,
    output_dir: str,
    model: str = "",
    postprocess_preset: str = "balanced",
    no_compare_gt: bool = False,
) -> CabfStepPlan:
    argv = [
        _text(python_executable),
        "-m",
        "sew_point_conntect.batch_predict",
        "--image_dir",
        _text(images_dir),
        "--annotation_dir",
        _text(annotation_dir),
        "--output_annotation_dir",
        _text(output_dir),
        "--postprocess_preset",
        _text(postprocess_preset),
    ]
    if _text(model).strip():
        argv.extend(("--model_path", _text(model)))
    if no_compare_gt:
        argv.append("--no_compare_gt")
    return CabfStepPlan(
        CabfStepKind.PREDICT_EDGES,
        tuple(argv),
        _text(modules_root),
        inputs=_paths(
            (CabfPathRole.IMAGES, images_dir),
            (CabfPathRole.POINT_ANNOTATIONS, annotation_dir),
            (CabfPathRole.MODEL, model),
        ),
        outputs=_paths((CabfPathRole.EDGE_ANNOTATIONS, output_dir)),
    )


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


def build_train_model_a_plan(
    *,
    python_executable: str,
    modules_root: str,
    images_dir: str,
    annotation_dir: str,
    output_dir: str,
) -> CabfStepPlan:
    argv = (
        _text(python_executable),
        "-m",
        "sew_point.train",
        "--img_dir",
        _text(images_dir),
        "--ann_dir",
        _text(annotation_dir),
        "--save_dir",
        _text(output_dir),
    )
    return CabfStepPlan(
        CabfStepKind.TRAIN_MODEL_A,
        argv,
        _text(modules_root),
        inputs=_paths(
            (CabfPathRole.IMAGES, images_dir),
            (CabfPathRole.MASTER_ANNOTATIONS, annotation_dir),
        ),
        outputs=_paths((CabfPathRole.OUTPUT, output_dir)),
    )


def build_train_model_b_plan(
    *,
    python_executable: str,
    modules_root: str,
    images_dir: str,
    annotation_dir: str,
    output_dir: str,
) -> CabfStepPlan:
    argv = (
        _text(python_executable),
        "-m",
        "sew_point_conntect.train",
        "--image_dir",
        _text(images_dir),
        "--annotation_dir",
        _text(annotation_dir),
        "--save_dir",
        _text(output_dir),
    )
    return CabfStepPlan(
        CabfStepKind.TRAIN_MODEL_B,
        argv,
        _text(modules_root),
        inputs=_paths((CabfPathRole.IMAGES, images_dir), (CabfPathRole.MASTER_ANNOTATIONS, annotation_dir)),
        outputs=_paths((CabfPathRole.OUTPUT, output_dir)),
    )


def build_post_point_workflow_plan(
    *,
    python_executable: str,
    repo_root: str,
    modules_root: str,
    images_dir: str,
    point_annotation_dir: str,
    edge_output_dir: str,
    edge_model: str = "",
    postprocess_preset: str = "balanced",
    no_compare_gt: bool = False,
    validate_annotation_dir: str | None = None,
    export_annotation_dir: str | None = None,
    validate_annotation_role: CabfPathRole = CabfPathRole.EDGE_ANNOTATIONS,
    export_annotation_role: CabfPathRole = CabfPathRole.EDGE_ANNOTATIONS,
    report_path: str = "",
    show_samples: bool = False,
    model_a_output: str = "",
    model_b_output: str = "",
    include_train: bool = False,
    model_a_images: str = "",
    model_a_annotations: str = "",
    model_a_out: str = "",
    model_b_images: str = "",
    model_b_annotations: str = "",
    model_b_out: str = "",
) -> CabfWorkflowPlan:
    """Build the tail after point inference and its shell-side resolution."""
    edge = build_predict_edges_plan(
        python_executable=python_executable,
        modules_root=modules_root,
        images_dir=images_dir,
        annotation_dir=point_annotation_dir,
        output_dir=edge_output_dir,
        model=edge_model,
        postprocess_preset=postprocess_preset,
        no_compare_gt=no_compare_gt,
    )
    validate_annotations = edge_output_dir if validate_annotation_dir is None else validate_annotation_dir
    export_annotations = edge_output_dir if export_annotation_dir is None else export_annotation_dir
    steps: list[CabfStepPlan] = [
        edge,
        build_validate_plan(
            python_executable=python_executable,
            repo_root=repo_root,
            images_dir=images_dir,
            annotation_dir=validate_annotations,
            report_path=report_path,
            show_samples=show_samples,
            annotation_role=validate_annotation_role,
        ),
        build_export_model_a_plan(
            python_executable=python_executable,
            repo_root=repo_root,
            images_dir=images_dir,
            annotation_dir=export_annotations,
            output_dir=model_a_output,
            annotation_role=export_annotation_role,
        ),
        build_export_model_b_plan(
            python_executable=python_executable,
            repo_root=repo_root,
            images_dir=images_dir,
            annotation_dir=export_annotations,
            output_dir=model_b_output,
            annotation_role=export_annotation_role,
        ),
    ]
    if include_train:
        steps.extend((
            build_train_model_a_plan(
                python_executable=python_executable,
                modules_root=modules_root,
                images_dir=model_a_images,
                annotation_dir=model_a_annotations,
                output_dir=model_a_out,
            ),
            build_train_model_b_plan(
                python_executable=python_executable,
                modules_root=modules_root,
                images_dir=model_b_images,
                annotation_dir=model_b_annotations,
                output_dir=model_b_out,
            ),
        ))
    return CabfWorkflowPlan(tuple(steps))


def build_workflow_plan(
    *,
    python_executable: str,
    repo_root: str,
    modules_root: str,
    images_dir: str,
    point_output_dir: str,
    edge_output_dir: str,
    point_model: str = "",
    edge_model: str = "",
    point_threshold: float = 0.3,
    point_distance_threshold: float = 0.0,
    postprocess_preset: str = "balanced",
    no_compare_gt: bool = False,
    edge_annotation_dir: str | None = None,
    validate_annotation_dir: str | None = None,
    export_annotation_dir: str | None = None,
    validate_annotation_role: CabfPathRole = CabfPathRole.EDGE_ANNOTATIONS,
    export_annotation_role: CabfPathRole = CabfPathRole.EDGE_ANNOTATIONS,
    report_path: str = "",
    show_samples: bool = False,
    model_a_output: str = "",
    model_b_output: str = "",
    include_train: bool = False,
    model_a_images: str = "",
    model_a_annotations: str = "",
    model_a_out: str = "",
    model_b_images: str = "",
    model_b_annotations: str = "",
    model_b_out: str = "",
) -> CabfWorkflowPlan:
    """Build the standard point → edge → validate → export workflow.

    ``None`` annotation bindings are automatic: point output feeds edge,
    edge output feeds validation and both exports.  Passing a string makes the
    binding explicit, which is useful when the shell selected a fallback or a
    caller intentionally uses a hand-labelled directory.
    """
    point = build_predict_points_plan(
        python_executable=python_executable, modules_root=modules_root,
        images_dir=images_dir, output_dir=point_output_dir, model=point_model,
        threshold=point_threshold, distance_threshold=point_distance_threshold,
    )
    edge_annotations = point_output_dir if edge_annotation_dir is None else edge_annotation_dir
    tail = build_post_point_workflow_plan(
        python_executable=python_executable,
        repo_root=repo_root,
        modules_root=modules_root,
        images_dir=images_dir,
        point_annotation_dir=edge_annotations,
        edge_output_dir=edge_output_dir,
        edge_model=edge_model,
        postprocess_preset=postprocess_preset,
        no_compare_gt=no_compare_gt,
        validate_annotation_dir=validate_annotation_dir,
        export_annotation_dir=export_annotation_dir,
        validate_annotation_role=validate_annotation_role,
        export_annotation_role=export_annotation_role,
        report_path=report_path,
        show_samples=show_samples,
        model_a_output=model_a_output,
        model_b_output=model_b_output,
        include_train=include_train,
        model_a_images=model_a_images,
        model_a_annotations=model_a_annotations,
        model_a_out=model_a_out,
        model_b_images=model_b_images,
        model_b_annotations=model_b_annotations,
        model_b_out=model_b_out,
    )
    return CabfWorkflowPlan((point, *tail.steps))


__all__ = [
    "BoundPath",
    "CabfPathRole",
    "CabfStepKind",
    "CabfStepPlan",
    "CabfWorkflowPlan",
    "build_predict_points_plan",
    "build_predict_edges_plan",
    "build_validate_plan",
    "build_export_model_a_plan",
    "build_export_model_b_plan",
    "build_train_model_a_plan",
    "build_train_model_b_plan",
    "build_post_point_workflow_plan",
    "build_workflow_plan",
]
