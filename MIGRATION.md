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
- 图像、标注与训练窗口通过 `WorkspaceAdapter` 作为子工作区嵌入唯一主窗口。
- 第一阶段曾使用顶部水平工作区标签；第二阶段已替换为由能力目录生成的左侧项目流程导航。
- `WorkspaceAdapter` 会清除旧业务窗口的顶层窗口标记、隐藏嵌套菜单栏，并在图像工作区固定原有 dock，避免出现额外顶层窗口或漂浮面板。
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

旧图像、标注和训练 `QMainWindow` 仍作为兼容 adapters 存在，后续按业务页面逐步替换；新增定制功能不得继续扩展这些旧主窗口。

单窗口模式的界面边界如下：

- 业务表单、编辑器、历史记录、日志和详情页必须嵌入主窗口。
- 只有操作系统文件/目录选择、危险操作确认和错误提示允许保留为弹窗。
- 旧业务窗口仍保留独立启动 fallback；只有独立启动时才沿用原来的 `exec()` 行为。

## 模型解析

注册中心先按任务和模式过滤，再按优先级选择可用实现：`algo/cab_f`（300）→ `train/models`（200）→ 工具箱内随迁移代码保存的本地模型（100）→ 外部 Ultralytics（50）。`algo/cab_f` 当前为缝纫点与连边任务提供生产 ONNX 推理封装，所以这些任务的推理优先使用它；训练/导出会跳过它，默认使用 `train/models`，不存在兼容实现时再落到迁移项目内模型。

| 任务 | 训练模型 | 生产消费者 |
|---|---|---|
| segmentation | `cosmos/train/models/microunet.py` | `algo.models.unet_segmenter` |
| sew_point | 工具箱 `modules/sew_point/model.py` | `algo.cab_f.sew_point_detector` |
| sew_point_connect | `cosmos/train/models/sew_point_connector.py` | `algo.cab_f.sew_point_connector` |
| yolo | `ultralytics.YOLO` | `algo/cab_f` YOLO wrappers |

模型注册中心位于 `cosmos_toolbox/training/model_registry.py`。训练模式会跳过仅支持 ONNX 推理的 `algo/cab_f` provider。

## 验证基线

- `onnx-gpu` 环境预检通过，PyTorch CUDA 和 ONNX Runtime CUDA/TensorRT 可用。
- 迁移、能力架构与单窗口融合测试：193 passed（2026-07-15）。
- 单窗口 offscreen 冒烟验证：嵌套业务页打开后仅有 1 个可见顶层窗口，`PageRouter` 返回深度正确。
- 桌面快捷方式、批处理与 PowerShell 启动链已核查，实际参数冒烟通过且不再依赖 `ll`。
- Segmentation、Sew Point、Sew Point Connect 反向传播 smoke 通过。
- MicroUNet、Sew Point、split EdgeGraphNet ONNX 契约检查通过。
- Sew Point ONNX 动态输入 `64x64` 与 `80x96` 均通过。
- 工具箱位于 Cosmos 忽略的 `/tools/` 下，不会把个人工具代码带入 Cosmos Git。
