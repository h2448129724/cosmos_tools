"""CAB-F project tool registration."""
from __future__ import annotations

from .models import RegisteredProject, RegisteredProjectTool


PROJECT = RegisteredProject(
    key="cabf",
    menu_title="CAB-F",
    display_name="CAB-F 缝纫点与连边标注数据处理",
    tools=(
        RegisteredProjectTool(
            key="cabf_workflow",
            title="9步流程",
            description="按 CAB-F 项目预设执行数据初始化、筛选、预测、修正、校验与训练导出流程。",
            launcher_name="_show_cabf_workflow_dialog",
            category="主流程",
            featured=True,
        ),
        RegisteredProjectTool(
            key="cabf_stitch_editor",
            title="连边标注器",
            description="编辑 CAB-F 图像中的缝纫点与连边关系。若当前主界面已经加载图片，会自动把当前图片带入编辑器。",
            launcher_name="_show_graph_annotation_dialog",
            category="标注修正",
        ),
        RegisteredProjectTool(
            key="cabf_point_editor",
            title="缝纫点编辑器",
            description="编辑和修正 CAB-F 缝纫点标注，支持模型辅助预标注。",
            launcher_name="_show_point_annotation_dialog",
            category="标注修正",
        ),
        RegisteredProjectTool(
            key="cabf_point_filter",
            title="缝纫点数据筛选",
            description="筛选和复核 CAB-F 缝纫点数据，适合批量检查异常样本、脏数据和候选修复对象。",
            launcher_name="_show_point_filter_dialog",
            category="数据复核",
        ),
        RegisteredProjectTool(
            key="dataset_export_dialog",
            title="数据集校验与导出",
            description="校验 CAB-F 母数据，按样本汇总问题，并导出训练数据到 images / annotations / error 目录结构。",
            launcher_name="_show_dataset_export_dialog",
            category="数据复核",
        ),
    ),
)
