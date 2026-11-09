# 工具箱优化验收 · 2026-09-29

## 2026-09-30 · 整体 review 修复增补

本轮改动位于嵌套工具箱仓库；未修改生产原图、历史 run.db、已有标注或 Conda 依赖，未提交/推送。

- 训练运行：安全名称和目录归属、原子唯一目录、启动失败归档、stdout/stderr 排空后终态、快速结束信号顺序、取消/关闭和平台 PATH。
- 模型工具：递归自动标注镜像输出目录；同 stem 不同扩展名冲突在写入前拒绝；训练按相对目录配对；小图和边缘滑窗写回；`keypoint`/`sew` 共用标签规则。
- 配置工具：inline YAML 列表/映射、路径转义、保存前预检、配置提交失败恢复原模板或移除本次新模板。
- 现场数据集：完整 PNG 容器与像素校验；有效人工标注仍可复用；历史导出逐样本暂停/停止，未完成目录保留 INCOMPLETE.txt 与状态报告。停止并导出仍完成已提交样本的导出。
- 架构：现场批次执行从 Qt worker 移至 field_batch；18 个 core 登记 source-effect / inert-import gates。精确审核过的核心哈希迁移保留旧版本续跑，未放开未知算法变更。
- 训练兼容：集中支持现代项目入口与当前 legacy Cosmos 布局，修复契约检查和连边导出的 load_project_package 导入失败；现代入口错误不静默降级。

验证（现有 onnx-gpu）：

1. `python -m pytest tests -q`：242 passed，6 subtests passed。
2. `python -m cosmos_toolbox.training.verify_contracts`：passed=true。CPU 合成输入下分割 max_abs_diff=5.40e-08；针点 64×64 / 80×96 动态尺寸通过；连边双 ONNX max_abs_diff=1.19e-07。使用临时权重和目录，不是现场训练权重准确率验收。
3. 隔离 UI gallery：960×640 下 26 个 capability 页面均成功打开/截图；真实持久化状态保持不变。截图在系统 TEMP/cosmos-review-fixes-ui-20260930。
4. 人工查看现场数据集页面截图，表单滚动和主要动作可见；不把截图加载测试等同于全部交互验收。

仍需真实业务样本验收：14 套现场模型按产品与面别的适用性、NAS 断连/断电、大批量吞吐，以及 JPG 有损压缩对微小缺陷识别的影响。此次未声称这些已通过。

范围：当前 Cosmos 分支与嵌套 cosmos_toolbox 工作区。保留既有未提交修改；未更改生产算法、原图、已有标注或 Conda 依赖，未提交/推送。

## 实现

1. 新增运行与任务关联，默认导出与模型汇总仅包含当前运行范围；包含有效断点复用。历史累计导出为显式选项。预览默认读取最近运行，旧账本预览明确提示历史范围。
2. 配置先合并 inspection.model_params，再解析本地模型；运行签名记录权重、模板、实际算法及工具源码。核对 14 套逻辑模型输入契约，修正当前 legacy YOLO 的 BGR/RGB 调用差异，保留针点/加强布/连边的既有接口。耳片/拉带裁片大小、knife 阈值、缺陷滑窗参数读取有效配置。
3. 现场数据集与数据库 NG 页面接入任务中心；终态等待实际线程结束，协作停止只请求一次，产物登记来源、运行编号与任务关系。
4. 数据库 NG 页改为滚动表单、换行动作栏、文件表格与日志分页。现场页增加导出范围选择。
5. 使用逐图重置、64 MiB 上限的推理缓存。预测结果以像素、模型设置、预处理为键，返回隔离副本。NG 和生成器共享兼容的 YOLO、glue、knife 模型资源。配准事实仅在输入和参数一致时复用，不改变有效区域规则。任务结束解除工具侧闭包和模型引用。

## 自动化

环境：已有 onnx-gpu。命令：`python -m pytest tests -q`。

结果：112 passed，6 subtests passed。

覆盖新增版本/来源隔离、配置/模板/源码变更、RGB 调用、缓存参数隔离/防修改/容量、配准不匹配回退、会话共享条件、任务完成顺序、取消和产物来源登记。`git diff --check` 无空白错误。

## 实际图片

只读输入：`D:\project\changrui\CAB-F\original\D01-R\0907\CAB-F_D01-R_20260907_071153920_top.png`。

配置：`conf/cabf/D01-R.local.yaml`。

临时输出：`C:\Users\tianwei\AppData\Local\Temp\cosmos-optimization-validation-20260929`。

- NG-only：选择 roi_detector、knife_segment、hook_detector、qr_yolo_cut。完成 1 个模型任务，knife 为 OK 被过滤，另外 2 项当前面未启用；技术失败 0。
- 导出公共 ROI 裁片 2 张、框 7 个、preview 2 张；结构校验与预览校验通过。
- 相同配置续跑：4 个模型任务全部跳过；解码和推理耗时为 0。
- 另一次非 NG-only knife 导出：6 张裁片和 6 张 preview，失败 0。保存的 ear/knife/circle mask 与同一 ONNX 的 RGB 参考结果逐像素一致。
- 本次 NG-only 样本预测缓存命中 0、配准复用 0。不将配置/ROI 不同的计算强行合并，不据此宣称端到端加速比例。

## 布局

Qt offscreen，加载系统 Microsoft YaHei 字体及工作台样式。现场数据集页和数据库 NG 页分别检查 960×640、1366×768，100%/125%/150% 缩放；页面保持目标尺寸、按钮可见；查看了截图，表单滚动而非挤压重叠。

这不是现场真实屏幕上的全交互自动化，也不是 14 套模型的全数据集准确率验收。标签仍为待人工复核的伪标签，当前实现没有改写生产端的行为。
