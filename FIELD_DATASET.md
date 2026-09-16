# CAB-F 现场数据集生成

入口：工具箱 → CAB-F 现场数据集生成，或在工具箱目录运行：

```powershell
conda run --no-capture-output -n onnx-gpu python -m cosmos_toolbox.field_dataset_ui
```

选择一个或多个原图目录、输出目录、产品 YAML 和正反面；独立勾选模型。
原图目录支持 Ctrl/Shift 多选添加，也可每行粘贴一个路径。款号自动读取 YAML 的 `inspection.product`，不读取文件名，不需要手动选择 D01-L/D01-R。
同一批目录使用同一份 YAML；不同款号需分批选择对应配置。多个输入目录依次执行，输出按目录名和稳定路径标识隔离，保留独立断点记录；同一批设置再次执行可续跑。
数量上限（包括「试跑前 2 张」）按每个文件夹计算，单目录失败会记录错误并继续下一个目录，停止则不再启动后续目录。
可使用「仅提取图片」或「提取并自动标注」。先设置最多处理 2 张进行预览，再设为 0 处理全部。
暂停在原图之间生效，当前原图处理完成后等待；停止保存已提交结果，相同设置再次开始会跳过完整结果。

命令行示例（工作目录为本工具箱）：

```powershell
conda run --no-capture-output -n onnx-gpu python -m cosmos_toolbox.field_dataset --source D:\project\changrui\CAB-F\original\D01-R\0907 --output D:\project\changrui\cosmos\tmp\CAB-F\field_dataset_test\example --product D01-R --models tail_roi_detector tail_cloth_roi_detector --limit 2
```

## 已实现

- 14 个模型独立选择；共享原图解码、ROI 与模型预测缓存。
- 原图只读；生成结果按产品、正反面、模型和版本分别保存。
- 模型与产品配置内容哈希、来源坐标变换、每模型完成状态。
- SQLite 执行账本、目录提交标记、输出目录进程锁。
- X-AnyLabeling、YOLO 检测/OBB、ImageFolder、逐类别 mask、针点/连边 JSON。
- 每次生成独立候选导出快照，人工修改不会被下一次导出覆盖。
- 同产品同目录同采集秒的 Top/Bottom 共享稳定训练划分（约 80/20）。

## 输出

`datasets` 保存已提交的小图和标签；`exports/candidates_*` 是可搬运复核/训练格式。
`run.db` 保存任务状态，`manifest.jsonl`、`summary.json`、`failures.json` 提供统计与失败详情。
`runs` 保留配置和模型快照。标签均是待复核候选，格式验证通过不代表人工真值。
空检测留在复核目录，不自动进入 YOLO 负样本。

## 当前适用边界

- 生产入口适配当前 `algo/cab_f` 与 `assets/config/backend_config.yaml`。
- 本地 Weight Registry 缓存中必须已有权重；不会自动下载或修改环境。
- 耳片和尾部使用原图公共 ROI 定位；固定加强布/二维码区域要求实际配准成功。
- 产品未配置某侧 ROI、配准失败或上游漏检，会记录对应模型失败，其他分支继续。
- placement 不使用旧临时脚本的颜色启发式自动补框；模型漏检会明确记录。
- 针点及连接采集当前范围为拉带布的 256 像素图块，来源记录为 `tail_cloth`。
- 默认收集所有耳片候选，包括 Bottom，用于模型数据扩充；不按生产 OK/NG 判定筛掉错误样本。
- 不支持在运行中编辑模型参数。更改配置或模型后开始新版本。
- 磁盘余量低于 2 GiB 时停止，已提交记录保留。输出目录不能与输入目录互相包含。
- 已提交文件若被删除，工具会报告缺失并保留剩余内容；在新输出目录重新生成。

## 验证

### 产品配置文件

默认读取 `conf/cabf/D01-L.yaml` 或 `D01-R.yaml`（不带 `.local`）。
界面“产品配置文件”可选任意 YAML，包括原来的 `.local.yaml`；留空恢复默认。
CLI 对应 `--product-config 路径`。程序只读所选文件，不与其他 YAML 合并，
运行日志和模型快照记录实际绝对路径。所选款号仍由界面指定，不从文件名推断。

配置须为 UTF-8 YAML，顶层 `inspection` 为映射。二维码与加强布导出需要
`inspection.conf.top/bottom.decode_image.rois` 或 `reinforcement.roi` 的四坐标
`[x_min, y_min, x_max, y_max]`，以及 `inspection.match_template` 的 top、bottom 模板。
缺少对应面的 ROI 会使该任务失败，不要求仅做其他模型时也补齐这些字段。
当前 dev 分支相对模板路径仍相对于 Cosmos 根目录；例如 `conf/CAB_F_D01_0.png`。
自定义 YAML 所在目录不会改变此规则，也可使用模板绝对路径。
模型权重与阈值仍读取 `assets/config/backend_config.yaml` 的 `cab_f`，不受产品配置选择影响。

### 选择执行环境

界面“执行 Conda 环境”默认 `onnx-gpu`。点击“加载环境列表”后选择环境，
也可手动输入已存在的环境名称。扫描、试跑和生成均在所选环境的独立 Python
进程内执行；暂停、继续和安全停止通过进程间控制传递，日志显示实际 Python 路径。
缺失环境或依赖会明确报错，不会自动安装、升级或退回其他环境。
界面本身不重启，也不更换其环境。若需指定界面的启动环境，启动脚本支持
`scripts/field_dataset.ps1 -CondaEnvironment onnx-gpu`。

```powershell
conda run --no-capture-output -n onnx-gpu python -m pytest tests/test_field_dataset_core.py tests/test_field_dataset_ui.py tests/test_field_export.py tests/test_field_models.py -q
```

现场验证原图：`D:\project\changrui\CAB-F\original\D01-R\0907`。
测试输出与原有拉带数据集独立存放于 `tmp\CAB-F\field_dataset_test`。

2026-09-10 实测：

- 14 模型单组测试：2 张原图、28 个模型任务，22 完成，6 个因缺失 placement、配准失败或该侧未配置 ROI 而失败；导出 2288 个样本，结构验证通过。
- 尾部四模型全批次：100 张原图、400 个模型任务全部完成，400 张样本，导出结构验证通过。
- 全批次验证的模型是 tail_roi_detector、tail_cloth_roi_detector、hook_detector、tail_cloth_seam_classifier；没有把其他模型声称为全量验证通过。
