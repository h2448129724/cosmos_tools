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
- 三个业务入口仍可独立运行，用于开发和兼容；从 Cosmos 工具箱启动时统一采用嵌入模式，嵌套菜单栏和可浮动面板会被收起或固定。

### 扩展定制功能

顶层导航不再写死在主窗口中。通用或定制功能注册为一个“能力”，能力目录会把它投影到左侧项目流程，并交给活动宿主打开。新增能力应提供原生业务页面；旧 `QMainWindow` 只通过兼容 adapter 继续工作。

训练模块会从 `module.json` 自动生成训练能力。例如 `sew_point_conntect` 会显示为独立训练入口，打开时自动选择对应训练模块，而不再显示训练工具自己的重复模块导航。

运行中的训练会同步到统一任务中心，完成后在项目会话中登记产物。项目设置中的路径和项目会话中的产物血缘是两类状态：前者描述当前输入，后者描述运行产生了什么。

首个原生定制能力是“CAB-F 配置与模板”：手动选择产品 YAML 和 TOP/BOTTOM 初始原图，一次生成全尺寸彩色校准基准图与半尺寸二值匹配模板。YAML ROI 只在校准基准图坐标系中查看、创建和调整；匹配模板仅用于生产标定匹配，不承载 ROI。生成任务在后台读取 Cosmos 的 `assets/config/backend_config.yaml → cab_f.glue_segment.path`，沿用现有权重注册解析，不再使用旧脚本写死的 `.pth`。大尺寸基准图在画布中降采样到最长边 4096，ROI 写回时仍保持全尺寸校准坐标。

实现位置：

- `cosmos_toolbox/project_context.py`：共享项目状态与持久化。
- `cosmos_toolbox/project_session.py`：产物与来源任务的持久化。
- `cosmos_toolbox/capabilities.py`：能力 interface、能力目录与项目流程分组。
- `cosmos_toolbox/default_capabilities.py`：内置和迁移期能力注册。
- `cosmos_toolbox/cabf_config.py`：CAB-F YAML、ROI、模板路径、生产模型解析与模板生成服务。
- `cosmos_toolbox/cabf_activity.py`：主界面内的 CAB-F 配置、基准图、模板和 ROI 工作台。
- `cosmos_toolbox/shell_widgets.py`：流程导航、活动宿主、项目检查器。
- `cosmos_toolbox/task_center.py`：统一任务 interface；任务中心活动页负责状态与日志呈现。
- `cosmos_toolbox/workspaces.py`：工作区 adapter seam，负责把旧 `QMainWindow` 转成嵌入式 `QWidget` 并隐藏子菜单栏。
- `cosmos_toolbox/page_router.py`：业务页面栈、嵌入式页头和嵌套返回。
- `cosmos_toolbox/app.py`：单窗口 Shell、能力导航、活动切换与统一样式。

## 模型来源

模型注册中心会先按任务和运行模式筛选兼容实现，再按优先级解析：`algo/cab_f`（优先级 300）→ `train/models`（200）→ 工具箱内随迁移代码保存的本地模型（100）→ 外部 Ultralytics（50）。

`algo/cab_f` 当前为缝纫点与连边任务注册的是生产 ONNX 推理封装，只声明 `infer` 模式。因此这些任务的推理优先使用它；训练和导出会跳过不兼容的推理封装，优先使用 `train/models`，对应实现不可用时再使用工具箱本地迁移模型。这个顺序既保留 Cosmos 生产实现的优先级，也保证训练代码始终拿到可训练模型。

训练产物默认写入本工具箱的 `runs/` / `artifacts/`，不会自动覆盖 `assets/weights`。生产发布必须经过显式 ONNX 导出和兼容性验证。

重复执行模型与 ONNX 契约检查：

```powershell
.\启动Cosmos工具箱.ps1 --preflight
conda run -n onnx-gpu python -m cosmos_toolbox.training.verify_contracts
```
