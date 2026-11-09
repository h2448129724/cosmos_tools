# Cosmos 个人工具箱

放置在 Cosmos 的 `tools/` 下，但不进入 Cosmos Git。这是一个轻量的数据与标注工具箱：图像整理与裁剪、CAB-F 点边标注与数据集导出、现场 NG 数据提取、数据库浏览和图片压缩。

工具箱**不包含模型训练与推理**，也不依赖 torch / onnx / onnxruntime / ultralytics。模型训练、ONNX 导出和完整检测请在 Cosmos 主仓库或独立训练环境中完成；这里只负责准备和查看数据。

## 依赖与启动

运行依赖只有 `numpy`、`Pillow`、`opencv-python`、`PySide6`、`openpyxl`、`pyyaml`（见 `requirements.txt`）。

```powershell
.\启动Cosmos工具箱.ps1
.\启动Cosmos工具箱.ps1 --preflight      # 检查依赖
.\启动Cosmos工具箱.ps1 --workspace image
.\启动Cosmos工具箱.ps1 --workspace labeling
```

启动脚本默认查找 conda 环境 `onnx-gpu` 的 Python；任何装好上述依赖的 Python 3.12+ 环境都可以运行，例如 `python -m cosmos_toolbox`。

## 项目工作台

单窗口工作台：顶部项目栏显示当前项目，左侧导航由能力目录生成，中央一次承载一项活动，右侧“项目设置”编辑共享路径，“任务与日志”集中显示后台任务。

项目上下文共享以下信息，保存在被忽略的 `artifacts/project_workspace.json`：

- 项目名称与数据集根目录
- 图片目录与标注目录
- 产物目录

选择数据集根目录后，会优先识别 `master_images` / `images`、`master_annotations` / `annotations` / `labels` 和 `outputs` / `runs` / `artifacts`。

### 单窗口交互约定

- 业务页面都嵌入同一个主窗口，进入业务页时显示统一的标题、说明和“返回”按钮；返回按后进先出逐层回到发起该流程的工作区。
- 切换左侧能力会结束当前业务页面栈，并直接进入所选活动。
- 只有操作系统文件/目录选择、危险操作确认以及错误提示允许使用弹窗。

## 功能一览

| 分组 | 功能 |
| --- | --- |
| CAB-F | CAB-F 数据与标注流程、点边标注、CAB-F ROI 配置 |
| 标注与复核 | 样本审阅、标注可视化、点位标注导出 |
| 现场数据 | NG 原图提取（数据库 / Excel 两种来源）、数据库浏览 |
| 图片处理 | 图片浏览、图片压缩、批量裁剪、逐张裁剪、自动分块、关键字划分、内侧 Mask、YAML ROI 编辑（通用） |
| 产物 | 项目产物、任务与日志 |

每类工作只有一个入口：人工筛选统一用“样本审阅”（A/D 翻页，S 保留，W 移除，Z 撤销，可多步撤销），CAB-F 流程里的“数据筛选”也是它；数据集校验与导出只在 CAB-F 流程第 4 步；通用 YAML ROI 编辑器遇到 CAB-F 配置会自动转到“CAB-F ROI 配置”，因为 CAB-F ROI 必须基于校准基准图坐标。

### CAB-F 数据与标注流程

四个步骤：**数据集准备 → 数据筛选 → 点边标注 → 校验与导出**。

- 数据筛选打开样本审阅，结果写回流程的母图目录。
- 点边标注打开点边一体标注器，直接读写母标注目录，在同一界面里修正缝纫点和连边。
- 校验与导出在页面内的后台线程执行，显示结构化统计（总样本、通过、错误、警告；导出数、跳过数），可取消；导出点数据集与连边数据集。
- 命令行入口：`python scripts/cabf_flow.py validate|export|pipeline|doctor|show-config|init-config`。流程路径保存在本机的 `apps/cabf_flow/config/default_paths.json`（不进 Git）。

### CAB-F ROI 配置

手动选择产品 YAML，再为 TOP/BOTTOM 各选一张全尺寸校准基准图。YAML ROI 只在校准基准图坐标系中查看、创建和调整；配置中的匹配模板只读预览。大尺寸基准图在画布中降采样到最长边 4096，ROI 写回时仍保持全尺寸坐标；保存前会校验越界，并保留 YAML 注释和格式、生成备份。

基准图与匹配模板由生产侧工具生成，工具箱不再包含胶体分割模型。

## 点位标注导出

在左侧“标注与复核”分组打开“点位标注导出”：

- “打开图片”读取一张图片，也支持把图片拖到画布上；打开新图会清空上一张图的点。
- 滚轮以光标位置为锚点缩放（5%–3200%），中键拖动平移，“适配窗口”恢复整图可见。
- 左键在图上标点，点按点击顺序编号；右键删除光标下的点，列表可单选后删除。
- 坐标始终是原图像素坐标：预览降采样到最长边 4096 只影响显示，不影响坐标。
- 右侧表格按顺序显示序号/X/Y，下方实时预览导出内容；“导出坐标 JSON”写出文件，“复制”直接写入剪贴板，“导入坐标”可以读回之前导出的文件。

导出文件是一个二层列表：外层按标注顺序排列，每个元素是 `[x, y]`。

```json
[
  [179, 219],
  [279, 300],
  [560, 220]
]
```

## Functional Core / Imperative Shell

业务决定先由 dependency-light functional core 根据显式 facts 生成状态、结果或计划，imperative shell 再负责 Qt、目录/YAML/JSON、进程/线程和文件写入。主要 seams 包括能力目录计划、CAB-F ROI/数据集/工作流、任务账本与呈现、点边编辑、图像裁剪和数据审阅。

实现位置：

- `cosmos_toolbox/project_context.py` / `project_session.py`：共享项目状态与产物登记。
- `cosmos_toolbox/capability_catalog.py` / `capabilities.py` / `default_capabilities.py`：纯能力目录计划与 Qt factories。
- `cosmos_toolbox/cabf_config.py` / `cabf_activity.py`：CAB-F YAML 文档与 ROI 工作台。
- `apps/cabf_flow/plan.py` / `flow.py`：CAB-F 校验与导出计划、CLI shell。
- `apps/labeling_ui/app/tools/workflow_page.py`：CAB-F 四步流程页（`CabfWorkflowActivity` 直接挂到工作台）。
- `apps/labeling_ui/app/dataset_export_panel.py`：进程内的校验与点 / 连边数据集导出面板。
- `cosmos_toolbox/ng_extract_ui.py`：NG 原图提取的统一入口，内含数据库与 Excel 两种来源。
- `img_tools/core/image_io.py`：全工具箱唯一的图片读写（Unicode 路径、EXIF 方向只处理一次、编码失败抛错）。
- `img_tools/core/review_session.py`：样本审阅的纯状态迁移，包括撤销。
- `img_tools/core/image_cache.py`：逐张浏览的解码缓存与后台预取（当前图前后各一张，文件变化自动重新解码）；`img_tools/ui/qt_image.py` 负责 BGR → Qt 转换，并在页面隐藏时释放缓存。
- `cosmos_toolbox/shell_widgets.py` / `app.py`：导航、活动宿主、项目检查器、任务摘要与单窗口 Shell。
- `cosmos_toolbox/task_center.py` / `task_ledger.py` / `task_presentation.py`：任务 shell、纯账本和统一呈现投影。
- `cosmos_toolbox/page_router.py` / `route_state.py` / `workbench_state.py`：页面路由与工作台 reducer。
- `cosmos_toolbox/file_lock.py`：跨平台输出目录排他锁（Windows `msvcrt`，其他平台 `fcntl`）。
- `cosmos_toolbox/ui/`：共享 theme、primitives 与 responsive projection。
- `apps/point_marking/point_core.py` / `apps/labeling_ui/app/tools/point_mark_page.py`：点位顺序与导出规则及其 Qt adapter。

新增规则应先进入 pure core；CLI、项目工作台和 legacy workspace 只做 adapter。

回归测试：

```powershell
python -m pytest -q
```

视觉 QA（offscreen 渲染所有能力页，不修改真实持久化文件）：

```powershell
python scripts/render_capability_gallery.py --output "$env:TEMP\cosmos-toolbox-ui-qa" --width 1280 --height 760
```

## NG 原图提取

在左侧“现场数据”中打开“NG 原图提取”，按情况选择来源：

- **来源：数据库**：能直接访问 SQLite 和原图时使用，扫描生成只读复制计划，可暂停、停止和续跑，详见 `DB_NG_EXPORT.md`。
- **来源：Excel / 导出 Excel**：只有导出的 Excel，或要在另一台电脑上提取时使用，步骤如下。

1. **数据库 → Excel**：选择 SQLite 数据库和 `.xlsx` 输出路径。默认导出全部用户表；也可每行填写一个表名。每张表对应一个工作表，保留关联 ID、JSON 和图片路径；长文本拆列，大整数按文本保存。默认不覆盖已有文件。
2. **Excel → NG 原图**：导出完成后自动填入 Excel 路径，也可直接选择其他电脑导出的 Excel。填写图片输出目录，按需设置原图盘符（默认 `E:`）、盘符映射根目录、检测起止时间和“只提取 NG 面”。先用“仅检查”查看源图缺失或目标冲突，再执行复制。

默认提取 `inspection_result.ok=0` 的全部关联原图，包含正反面和人工误判记录；“只提取 NG 面”使用 `ng_error.file_id`。优先读取 `filename`，其次 `raw_image`，不使用 `result_image`。时间范围两端均包含边界，结束日期包含当天全部记录。

复制平铺到输出目录，同名异源按来源目录加稳定前缀，保持正反面配对；相同内容跳过，不同内容不覆盖。实际复制自动生成 UTF-8 CSV 清单，含成功、缺失、冲突和未处理状态；页面显示前 200 条。仅检查默认不创建输出目录，也可指定新的 CSV 路径保存检查结果。图片任务可在文件之间停止；Excel 导出期间请等待完成后关闭窗口。两个步骤均在后台执行，记录到“任务与日志”。

需要 `openpyxl`（已列入 `requirements.txt`）。工具箱内也提供命令行入口，参数与原脚本兼容：

```powershell
python scripts/export_db_to_excel.py --db D:/data/cosmos.db --output D:/data/cosmos.xlsx
python scripts/copy_ng_images.py --excel D:/data/cosmos.xlsx --output D:/NG_recheck --source-root F:/ --start-date 2026-10-01 --end-date 2026-10-11 --dry-run
```

去掉 `--dry-run` 执行复制。

# 数据库浏览（只读）

启动工具箱后，在“现场数据”分组中打开“数据库浏览（只读）”，点击“打开 / 刷新”。
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
