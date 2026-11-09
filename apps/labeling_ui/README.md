# Labeling UI

工具箱中标注与数据整理相关的页面代码，全部由项目工作台挂载，不再提供独立启动的窗口。

- `app/tools/workflow_page.py`：CAB-F 四步流程（数据集准备 → 数据筛选 → 点边标注 → 校验与导出）。
- `app/graph_annotation_dialog.py`：点边一体标注器。
- `app/data_review/`：样本审阅（保留 / 移除 / 撤销），CAB-F 流程的数据筛选也复用它。
- `app/dataset_export_panel.py`：校验与点 / 连边数据集导出面板。
- `app/tools/`：裁剪、分块、关键字划分、内侧 Mask、通用 YAML ROI、标注可视化、点位标注导出等页面。

图片读写统一使用 `img_tools.core.image_io`。
