# Cosmos 个人工具箱

放置在 Cosmos 的 `tools/` 下，但不进入 Cosmos Git。包含通用图像处理、CAB-F 标注数据工具和完整训练/推理入口。

当前界面采用任务导向的统一项目工作台：紧凑项目栏显示当前项目，左侧流程导航由能力目录生成，中央活动宿主一次承载一项业务活动，右侧项目检查器按需编辑路径，“任务与日志”能力集中运行状态与日志。图像、标注和训练不再打开三个独立主窗口。

视觉上采用朴素桌面工程工具风格：浅灰导航和工作区、细边框、2–3px 小圆角及紧凑控件；蓝色仅用于当前选择和主要操作。避免深色品牌侧栏、胶囊徽章、大面积装饰色和多层卡片嵌套。

## 启动

```powershell
.\启动Cosmos工具箱.ps1
```

默认使用 conda 环境 `onnx-gpu`。环境检查：

```powershell
.\启动Cosmos工具箱.ps1 --preflight
```

也可以让工作台启动后直接定位到某个工作区：

```powershell
.\启动Cosmos工具箱.ps1 --workspace image
.\启动Cosmos工具箱.ps1 --workspace labeling
.\启动Cosmos工具箱.ps1 --workspace training
```

## 项目工作台

顶部项目栏共享以下信息：

- 项目名称与数据集根目录
- 图片目录与标注目录
- 当前模型与产物目录
- 固定运行环境 `onnx-gpu`

选择数据集根目录后，会优先识别 `master_images` / `images`、`master_annotations` / `annotations` / `labels` 和 `outputs` / `runs` / `artifacts`。共享状态保存在被忽略的 `artifacts/project_workspace.json`。

工作区之间的首批联动：

- 图像工作区可以直接载入当前项目图片，并把当前选中图片写回项目上下文。
- CAB-F 标注、筛选、数据导出和 9 步流程会接收当前图片、标注、模型与产物路径。
- 训练表单会自动填入当前图片目录、标注目录和兼容的模型路径。

### 单窗口交互约定

- 工作区主界面和业务功能页都嵌入同一个主窗口；CAB-F 流程、点边标注、数据筛选、数据集导出、图像任务历史、训练设置和完整日志等不会再创建独立业务窗口。
- 进入业务功能时，主内容区显示统一的页面标题、说明和“返回”按钮。业务页可以继续打开下一级页面；返回时按后进先出顺序逐层回到上一级，最后回到发起该流程的工作区，已有流程状态不会因打开下一级页面而丢失。
- 切换左侧项目流程能力会结束当前业务页面栈，并直接进入所选活动。
- 只有操作系统文件/目录选择、危险操作确认以及错误提示允许使用弹窗；业务表单、编辑器、历史记录和日志详情应保持在主窗口内。
- 三个业务入口仍可独立运行，用于开发和兼容；从 Cosmos 工具箱启动时统一使用原生 `QWidget` workspace page，不创建或伪造嵌套菜单栏。

### 扩展定制功能

顶层导航不再写死在主窗口中。通用或定制功能先进入纯能力目录计划，再由 Qt factories / workspace activators 绑定到左侧项目流程并交给活动宿主打开。新增能力应提供原生业务页面；旧 `QMainWindow` 只通过兼容 adapter 继续工作。训练 feature 扫描失败或没有启用项时，原生 Training 工作区会作为可见 fallback，入口不会静默消失。

训练模块会从 `module.json` 自动生成训练能力。例如 `sew_point_conntect` 会显示为独立训练入口，打开时自动选择对应训练模块，而不再显示训练工具自己的重复模块导航。

Segmentation 的“导出 ONNX”动作支持 `microunet` / `microunet_gn`、二分类或多标签输出通道，并生成高宽动态（尺寸需能被 4 整除）的模型。导出器会先在临时文件上执行 ONNX 结构检查及静态/动态尺寸 PyTorch 数值对照，全部通过后才替换目标产物。

运行中的训练会同步到统一任务中心，完成后在项目会话中登记产物。项目设置中的路径和项目会话中的产物血缘是两类状态：前者描述当前输入，后者描述运行产生了什么。

首个原生定制能力是“CAB-F 配置与模板”：手动选择产品 YAML 和 TOP/BOTTOM 初始原图，一次生成全尺寸彩色校准基准图与半尺寸二值匹配模板。YAML ROI 只在校准基准图坐标系中查看、创建和调整；匹配模板仅用于生产标定匹配，不承载 ROI。生成任务在后台读取 Cosmos 的 `assets/config/backend_config.yaml → cab_f.glue_segment.path`，沿用现有权重注册解析，不再使用旧脚本写死的 `.pth`。大尺寸基准图在画布中降采样到最长边 4096，ROI 写回时仍保持全尺寸校准坐标。

实现位置：

- `cosmos_toolbox/project_context.py`：共享项目状态与持久化。
- `cosmos_toolbox/project_session.py`：产物与来源任务的持久化。
- `cosmos_toolbox/capability_catalog.py` / `capabilities.py`：纯能力 specification/catalog projection 与 shell binding interface。
- `cosmos_toolbox/default_capabilities.py`：feature 扫描、Qt factories 和 workspace activators adapter。
- `cosmos_toolbox/cabf_calibration.py` / `cabf_config.py`：CAB-F 校准计划与 YAML/OpenCV/模型 shell。
- `apps/cabf_flow/annotation_source_core.py` / `annotation_source.py`：requested → master → prediction 标注源决策与 JSON/目录扫描 shell。
- `modules/sew_point_conntect/evaluation_core.py`：连边 normalization、指标、点匹配和 graph lifting 的唯一规则来源。
- `modules/segmentation/tools/export_onnx.py`：Segmentation checkpoint 的动态 ONNX 导出、数值验证与原子发布 shell。
- `cosmos_toolbox/cabf_activity.py`：主界面内的 CAB-F 配置、基准图、模板和 ROI 工作台。
- `cosmos_toolbox/shell_widgets.py`：流程导航、活动宿主、项目检查器与常驻任务摘要。
- `cosmos_toolbox/task_center.py` / `task_ledger.py` / `task_presentation.py`：任务 shell、纯账本和统一呈现投影。
- `cosmos_toolbox/workspaces.py`：native workspace seam；只对显式 legacy `QMainWindow` 执行兼容嵌入。
- `cosmos_toolbox/page_router.py` / `route_state.py`：Qt 页面 router 与纯路由状态。
- `cosmos_toolbox/workbench_state.py`：导航、header ownership、检查器、viewport 与任务摘要的纯 reducer。
- `cosmos_toolbox/ui/`：全工作台共享 theme、primitives 与 responsive projection。
- `cosmos_toolbox/app.py`：单窗口 Shell、能力导航、活动切换与统一样式。

## Functional Core / Imperative Shell

业务决定先由 dependency-light functional core 根据显式 facts 生成状态、结果或计划，imperative shell 再负责 Qt、目录/YAML/JSON、进程/线程、Conda、Torch/ONNX、GPU 和文件写入。主要 seams 包括 Cosmos 请求/结局、能力目录计划、CAB-F ROI/数据集/工作流/标注源/校准、任务账本与呈现、模型选择、训练运行计划、Sew Point/连边推理与评估、YOLO 动作、点边编辑、图像裁剪和数据审阅。

新增规则应先进入 pure core；CLI、项目工作台和 legacy workspace 只做 adapter。任务级模型/runtime 显式持有并在任务内复用，不使用隐式全局 cache。CAB-F 数据集按 `read facts → assess/decide → persist` 运行，裁剪按 `decode → plan → slice/write` 运行，数据审阅按 `plan → preflight/execute → commit/rollback` 运行。

架构回归：

```powershell
conda run -n onnx-gpu python -m pytest -q tests/test_functional_core_architecture.py tests/test_core_import_purity.py
conda run -n onnx-gpu python -m pytest -q
```

视觉与 teardown QA：

```powershell
conda run -n onnx-gpu python scripts/render_capability_gallery.py --output "$env:TEMP\cosmos-toolbox-ui-qa" --width 1280 --height 760
conda run -n onnx-gpu python scripts/render_capability_gallery.py --output "$env:TEMP\cosmos-toolbox-ui-qa-wide" --width 1540 --height 920
```

脚本使用隔离的项目/训练设置副本，并在 offscreen 模式显式加载可用 CJK 字体；不会修改真实工作区持久化文件。

## 模型来源

模型注册中心会先按任务和运行模式筛选兼容实现，再按优先级解析：CAB-F 根入口生产 factory（优先级 300）→ CAB-F 根入口训练 factory（200）→ 工具箱内随迁移代码保存的本地模型（100）→ 外部 Ultralytics（50）。

CAB-F 根入口为缝纫点与连边任务公开惰性的生产 ONNX 推理 factory，只参与 `infer` 模式；训练和导出通过同一根入口取得项目内可训练模型，对应实现不可用时再使用工具箱本地迁移模型。工具箱不直接导入 CAB-F 的 implementation module。

训练产物默认写入本工具箱的 `runs/` / `artifacts/`，不会自动覆盖 `assets/weights`。生产发布必须经过显式 ONNX 导出和兼容性验证。

重复执行模型与 ONNX 契约检查：

```powershell
.\启动Cosmos工具箱.ps1 --preflight
conda run -n onnx-gpu python -m cosmos_toolbox.training.verify_contracts
```
# 数据库浏览（只读）

启动工具箱后，在“数据准备”分组中打开“数据库浏览（只读）”，点击“打开 / 刷新”。
默认读取 Cosmos 根目录的 `cosmos.db`，也可选择其他 SQLite 文件。

- 支持选择数据表、按字段或全部字段搜索包含文本（区分大小写）、排序和每页 200 条分页。
- “字段结构”显示字段类型、非空约束、默认值与主键序号；双击数据单元格查看完整内容。
- 仅使用 SQLite `mode=ro` 和 `query_only` 连接，不调用业务初始化、不建表、不提供编辑或 SQL 执行入口，不安装新依赖。
- 查询在后台执行，单次 SQL 查询超过约 5 秒会中止；运行中的数据库按已提交数据读取，可手动刷新。
- 目前支持 SQLite 文件，不支持 MySQL 等远程数据库；无主键表的默认行顺序由 SQLite 决定。

### 检查项速览

选中 `inspection_result` 的一条记录，点击“查看检查项”，会按 `result_id` 只读加载
`inspection_metadata` 中的所有明细，通过下拉框切换正反面。也可在明细表直接查看，
或点击“粘贴 JSON”→“解析查看”快速阅读复制的结果。

速览分别展示相邻耳片间距、耳片缝线距离、二维码内容及其他项目的保存状态。
缝线距离保留原始顺序并附 x 坐标；相邻间距使用算法保存的从左到右顺序。
没有保存单位时显示“未记录单位”，没有保存判定时不推测合格性；不使用当前配置重算历史结果。
“全部字段”可逐层展开尾部、缝线、坐标与阶段耗时，“原始 JSON”保留完整数据。
