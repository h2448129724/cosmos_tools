# CABF Flow

`cabf_flow` 是 monorepo 内的流程编排层，负责：

- 读取和保存工作流配置
- 校验关键路径与模型权重
- 调用 `modules/` 下的预测和训练 CLI
- 调用 `scripts/` 下的校验与导出入口

入口：

- Python: `python D:/project/tianwei/train_model/scripts/cabf_flow.py`
- Python module: `python -m apps.cabf_flow`
- PowerShell:
  - `cabf_predict_points.ps1`
  - `cabf_predict_edges.ps1`
  - `cabf_validate.ps1`
  - `cabf_export.ps1`
  - `cabf_train.ps1`
