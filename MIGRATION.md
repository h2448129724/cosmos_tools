# 迁移记录

迁移日期：2026-07-13

## 来源

- `D:\project\changrui\img_tools\img_tools` → `img_tools`
- `D:\project\tianwei\train_model\apps` → `apps`
- `D:\project\tianwei\train_model\modules` → `modules`
- `D:\project\tianwei\train_model\shared` → `shared`
- `D:\project\tianwei\train_model\trainer_gui` → `trainer_gui`
- `D:\project\tianwei\train_model\scripts` → `scripts`
- 两边测试合并到 `tests`

旧目录没有删除；迁移后的工具箱是新的运行入口，不再依赖旧 `train_model` 绝对路径。

## 单窗口融合

原先的入口只负责启动三个独立 `QMainWindow`。现在统一为项目工作台：

- `ProjectContext` 集中数据集、图片、标注、模型和产物路径。
- 图像、标注与训练均已抽取为原生 `QWidget` 工作区页面，通过 `WorkspaceAdapter` 嵌入唯一主窗口；原 `QMainWindow` 只保留独立启动兼容入口。
- 第一阶段曾使用顶部水平工作区标签；第二阶段已替换为由能力目录生成的左侧项目流程导航。
- `WorkspaceAdapter` 对原生页面不修改 window flags 或伪造菜单栏；只有显式接入真正 legacy `QMainWindow` 时才清除顶层标记并隐藏其菜单栏。
- 项目上下文持久化到 `artifacts/project_workspace.json`，不会进入 Cosmos Git。
- 当前图片会从图像工作区传递给标注工作区；训练与 CAB-F 流程复用共享路径。

业务功能通过 `PageRouter` 进入主窗口的统一内容栈。目前已嵌入的二级功能包括 CAB-F 工作流、点边标注、数据筛选、数据集导出、项目工具入口、图像任务历史、训练设置和训练日志/文本详情。命令生成结果直接显示在训练工作区已有的“命令”页签中。

页面栈采用后进先出返回：从 CAB-F 工作流进入筛选或点/边修正时，工作流页面继续保留；点击“返回”先回到工作流，再返回发起能力。切换左侧项目流程能力会清空当前页面栈，防止返回到另一个活动的旧页面。

### 任务导向 Shell（2026-07-15）

第二阶段不再把旧工具名称作为唯一导航结构：

- `CapabilityCatalog` 成为导航和定制功能发现的唯一来源。
- 左侧按项目、数据准备、标注复核、训练推理和产物组织能力。
- `ActivityHost` 一次只突出一个当前活动，项目路径编辑移入按需展开的检查器。
- 训练 `module.json` 自动投影为独立训练能力；进入后隐藏训练窗口自己的重复模块导航。
- `ProjectSession` 保存产物、来源能力、来源任务与输入关系。
- `TaskCenter` 适配训练 `RunManager`，统一保存任务状态和日志，并通过“任务与日志”活动呈现。
- Sew Point Connect 新增原生项目路径页，把素材、CAB-F 流程、训练和产物串成一条入口。
- CAB-F 配置与模板新增原生能力页：手动选择 TOP/BOTTOM 初始原图，同步生成全尺寸校准基准图与半尺寸匹配模板，并在校准坐标系完成 ROI 编辑、尺寸与越界检查；生成模型延迟读取生产 `backend_config.yaml`。

图像、标注和训练默认 adapters 现已全部返回 native workspace page。三个旧 `QMainWindow` 仅作为 standalone chrome adapters 存在；新增定制功能不得继续扩展其顶层导航职责。

单窗口模式的界面边界如下：

- 业务表单、编辑器、历史记录、日志和详情页必须嵌入主窗口。
- 只有操作系统文件/目录选择、危险操作确认和错误提示允许保留为弹窗。
- 旧业务窗口仍保留独立启动 fallback；只有独立启动时才沿用原来的 `exec()` 行为。

## 模型解析

注册中心先按任务和模式过滤，再按优先级选择可用实现：CAB-F 根入口的生产推理 factory（300）→ CAB-F 根入口的训练模型 factory（200）→ 工具箱内随迁移代码保存的本地模型（100）→ 外部 Ultralytics（50）。CAB-F 生产 ONNX 推理封装只参与推理模式；训练/导出通过同一个根入口惰性取得项目内训练模型，不存在兼容实现时再落到工具箱本地模型。

| 任务 | 训练模型 | 生产消费者 |
|---|---|---|
| segmentation | `cosmos/train/models/microunet.py` | `algo.models.unet_segmenter` |
| sew_point | 工具箱 `modules/sew_point/model.py` | CAB-F 根入口 `SewPointDetector` |
| sew_point_connect | CAB-F 根入口 `EdgeGraphNet` | CAB-F 根入口 `SewPointConnector` |
| yolo | `ultralytics.YOLO` | CAB-F 根入口公开的生产 wrappers |

模型注册中心位于 `cosmos_toolbox/training/model_registry.py`，其 `cab_f_project` adapter 只访问 `load_project_package('CAB-F')` 返回的公开属性。训练模式会跳过仅支持 ONNX 推理的 CAB-F factory。

## Functional Core / Imperative Shell（2026-08-14）

当前业务规则已迁入可直接测试、惰性导入的 functional cores；Qt、目录扫描、YAML/JSON、进程/线程、Conda、Torch/ONNX、GPU、时钟和文件移动保留在 imperative shells。兼容入口的函数和 CLI 保持不变。

| 领域 | Functional core | Imperative shell / adapter |
|---|---|---|
| Cosmos 完整检测 | `apps/cosmos_pipeline/{outcome,case_decision,request_plan,runtime_session}.py` | `runner.py`、项目工作台 activity |
| CAB-F ROI / 数据集 / 工作流 | `shared/cabf_common/cabf/{roi,dataset_core}.py`、`apps/cabf_flow/plan.py` | YAML/图片 I/O、CLI、工作流页面 |
| CAB-F 标注源 / 模板校准 | `apps/cabf_flow/annotation_source_core.py`、`cosmos_toolbox/cabf_calibration.py` | JSON/目录扫描、OpenCV、分割模型、模板 matcher |
| 能力目录 / 任务中心 | `cosmos_toolbox/capability_catalog.py`、`task_ledger.py`、`task_presentation.py` | feature 扫描、Qt factories、task_center signal 与取消 callback |
| 模型选择与运行 | `cosmos_toolbox/training/model_resolution.py`、`trainer_gui/run_plan.py` | registry 动态 import、RunManager、Conda/进程 |
| 缝纫点 / 连边 | `modules/sew_point/inference_core.py`、`modules/sew_point_conntect/{inference_core,evaluation_core}.py` | Torch/ONNX adapters、数据集/批次/可视化 wrappers 与显式 task-scoped runtime |
| YOLO | `modules/yolo/action_plan.py` | train/predict/export wrappers 与产物移动 |
| 点边标注 | `apps/labeling_ui/app/annotation/commands.py` | Qt canvases/dialogs |
| 图像裁剪 / 数据审阅 | `img_tools/core/crop_plan.py`、`img_tools/core/review_session.py` | 两套裁剪 adapter、审阅页面与文件移动 adapter |

生命周期规则：完整检测在模型加载前冻结 request facts；Sew Point Connect runtime 在单次任务内复用模型但不进入全局 cache；CAB-F 标注源按 requested → master → prediction 消费显式 JSON facts，校准计划保持 `(row, column)` half→full offset 与全尺寸 ROI 语义；CAB-F 数据集遵循 `read facts → assess/decide → persist`；图片裁剪遵循 `decode → plan → slice/write`，workspace 与 legacy adapters 显式保留各自既有缩放、目录和重名策略；数据审阅遵循 `plan → preflight/execute → commit`，复合移动中途失败则逆序 rollback，core session 不提交。

架构防回退由 `tests/test_functional_core_architecture.py`（source effects）和 `tests/test_core_import_purity.py`（惰性导入）共同执行。

## UI foundation 与工作台状态（2026-08-14）

- `cosmos_toolbox/ui` 统一主题 token、语义 roles、页面 scaffold、section、path field、换行 action bar、status banner、metrics、empty state、collapsible log 与 viewport density。
- `workbench_state.py`、`route_state.py` 和 `task_presentation.py` 分别以 pure reducer/projection 管理导航与 header ownership、嵌套页面生命周期、任务词汇与进度；Qt shells 只执行 intents。
- 工作台在 `<1180px` 使用 compact navigation，最低支持 960×640；项目检查器、页面 margin/spacing 与业务 activity 共享同一 responsive projection。
- 当前/最近任务通过常驻 task strip 投影，切换页面后仍可见并可进入任务中心。
- Labeling 的 Point/Graph editor、数据审阅、数据导出、批量裁剪、筛选、标签可视化和轻量工具页均使用共享 primitives；长日志独立折叠，不与结构化结果混在同一 label。
- 工具页后台 worker 的业务结果与 QThread lifecycle 已分离，页面 shutdown 会在释放 worker 前完成线程 teardown。
- CAB-F Config 在中等宽度投影侧栏 toggles、宽屏检查栏独立滚动；Cosmos Pipeline 的路径字段保持最小高度，页面整体纵向滚动，不再因 viewport 变矮而相互压叠。
- `scripts/render_capability_gallery.py` 隔离运行时持久化、显式加载 CJK QA 字体并逐一渲染所有可见能力；compact 与 wide 两种尺寸均用于原尺寸视觉验收。

## 验证基线

- `onnx-gpu` 环境预检通过，PyTorch CUDA 和 ONNX Runtime CUDA/TensorRT 可用。
- 全量回归：513 passed、3 skipped（2026-08-14，`onnx-gpu`）；三个 skip 是当前 Cosmos checkout 未提供的外部产品配置 fixtures。
- Functional-core source-effect 与 inert-import gates 全部通过。
- 双尺寸视觉 QA：compact 1280×760 与 wide 1540×920 均完成 22/22 可见能力渲染，中文字体、滚动、换行、无重叠和持久化隔离通过。
- 单窗口 offscreen 冒烟验证：逐页打开全部可见能力后 teardown 的可见顶层窗口为 0，`PageRouter` 返回深度正确。
- 全量 Ruff、`compileall` 与 `git diff --check` 通过。
- 桌面快捷方式、批处理与 PowerShell 启动链已核查，实际参数冒烟通过且不再依赖 `ll`。
- Segmentation、Sew Point、Sew Point Connect 反向传播 smoke 通过。
- MicroUNet、Sew Point、split EdgeGraphNet ONNX 契约检查通过。
- Sew Point ONNX 动态输入 `64x64` 与 `80x96` 均通过。
- 工具箱位于 Cosmos 忽略的 `/tools/` 下，不会把个人工具代码带入 Cosmos Git。
- 本次重构未安装、升级或降级任何环境依赖。
