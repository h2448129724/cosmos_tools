# CABF Flow

`cabf_flow` 是 CAB-F 标注数据的流程编排层，负责：

- 读取和保存工作流配置（`config/default_paths.json`，本机文件，不进 Git）
- 调用 `scripts/` 下的母标注校验与点 / 连边数据集导出入口

工具箱不再包含模型预测和训练；点和连边标注在“点边标注”页面中人工完成。

入口：

- Python: `python scripts/cabf_flow.py <command>`
- Python module: `python -m apps.cabf_flow <command>`
- PowerShell: `cabf_validate.ps1`、`cabf_export.ps1`

命令：`validate`、`export`、`pipeline`（校验后导出）、`doctor`、`show-config`、`init-config`。
