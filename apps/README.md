# Apps Layout

当前 `apps/` 下分为三类：

- `cabf_flow`
  - 单仓流程配置、路径模板、命令编排
- `labeling_ui`
  - 标注 UI、数据筛选 UI、流程控制台页面
- `data_tools`
  - 图像裁剪、关键词拆分、CABF 数据集辅助逻辑

设计约束：

- `modules/` 不依赖 `labeling_ui`
- `shared/cabf_common` 是 CABF 协议真源
- `labeling_ui` 与 `data_tools` 可以依赖 `shared/cabf_common` 和 `cabf_flow`
- `labeling_ui` 与 `data_tools` 已整理为 monorepo 包导入；仅保留 `shared/cabf_common` 的路径注入以便直接脚本启动

命名约定：

- `apps/labeling_ui/app`：应用层 UI 代码
- `apps/data_tools/processing`：处理逻辑
- `apps/data_tools/common`：通用辅助模块
