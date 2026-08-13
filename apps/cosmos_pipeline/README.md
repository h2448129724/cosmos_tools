# Cosmos Headless Pipeline

无生产页面、无硬件接口地执行 Cosmos 完整检测业务链：

`产品配置 → 输入槽 → 检测 → ok_checkers → 绘图 → 产品汇总 → 文件结果（→ 可选数据库）`

输入清单示例：

```yaml
color_space: rgb
cases:
  - id: sample-001
    inputs:
      top:
        - D:/data/sample_top.png
      bottom:
        - D:/data/sample_bottom.png
```

`CAB` 的输入槽名为 `combined`，文件数量由产品配置中的组合相机数量决定；DAB、CAB-F、OS-DAB 的槽名来自 `inspection.conf` 的键。

命令行：

```powershell
conda run --no-capture-output -n onnx-gpu python -u scripts/cosmos_pipeline_test.py `
  --config D:\project\changrui\cosmos\conf\CAB-F\D01-L.local.yaml `
  --backend-config D:\project\changrui\cosmos\assets\config\backend_config.yaml `
  --input-manifest D:\data\cosmos_inputs.yaml `
  --output-dir D:\data\cosmos_pipeline_result `
  --engine auto `
  --dry-run
```

默认不启动 UI、相机、PLC、Socket、同步线程，也不写数据库。只有显式传入 `--persist-db` 才会调用 Cosmos `ResultWriter`。
业务判定为 NG 时默认返回退出码 `2`，技术执行错误返回其他非零码；仅在兼容旧脚本时显式传入
`--allow-ng`，让业务 NG 返回 `0`。`--backend-config` 中的 `path`/`*_path` 模型字段会在推理前完成
权重引用解析并检查文件是否存在，相对路径统一以 Cosmos 根目录为基准。
现场验证建议每次使用新的 `--output-dir`；复用已有目录时，本次 `run_manifest.json` 会覆盖，但旧 case
子目录不会自动删除。

## CAB-F GPU batch 自动标定

自动标定会对 density 的针点检测 batch 和连接 patch batch 做网格搜索。每个候选都在独立 Python
进程运行同一个真实产品输入，避免前一个候选的 ONNX Runtime CUDA arena 污染后续结果。候选必须同时通过：

- Pipeline 技术执行成功；
- 产品判定、消息和 detector 语义摘要与基准候选一致；
- 重复运行没有超过阈值的稳态变慢；
- 峰值显存不超过目标显存减去预留显存。

面向 RTX 4070 12GB 的完整标定：

```powershell
$toolbox = 'D:\project\changrui\cosmos\tools\cosmos_toolbox'
$output = Join-Path $toolbox ("artifacts\cosmos_pipeline\batch_tune_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))

Set-Location $toolbox
conda run --no-capture-output -n onnx-gpu python -u scripts\cosmos_batch_tune.py `
  --config 'D:\project\changrui\cosmos\conf\CAB-F\D01-L.local.yaml' `
  --backend-config 'D:\project\changrui\cosmos\assets\config\backend_config.yaml' `
  --input-manifest "$toolbox\artifacts\cosmos_pipeline_real_dry_run_inputs.yaml" `
  --output-dir $output `
  '--point-batches=4,8,12,16' `
  '--patch-batches=32,64,128' `
  --warmup-cases 1 `
  --repeats 3 `
  --target-vram-mib 12288 `
  --reserve-vram-mib 1536
```

PowerShell 下需要把包含逗号的完整参数加引号。先用
`'--point-batches=8,16' '--patch-batches=32,64' --warmup-cases 0 --repeats 1` 做快速冒烟测试。
中断后可加 `--resume` 复用已完成候选。

主要产物：

- `tuning_report.json`：硬件、静态模型输入 shape、真实 session provider/options/lifecycle、每个候选的性能、显存和质量门禁；
- `recommended_product_override.yaml`：只包含产品配置中的硬件相关 batch override；
- `recommended_hardware_profile.yaml`：把推荐 batch/provider profile 绑定到 GPU、驱动、ORT 和模型 artifact；
- `candidates/<candidate>/onnx_sessions.json`：该候选实际创建的所有 ONNX session；
- `candidates/<candidate>/pipeline.log`：完整 Pipeline stdout/stderr。

`single_line`、长度容错和 batch 等参数跟随产品配置 `conf/CAB-F/*.yaml`；backend 只保存模型路径及模型推理
阈值。batch 推荐应按 GPU 型号、显存、驱动、ONNX Runtime 版本和模型版本重新标定。

如果默认 CUDA EP 的 session 常驻显存无法通过 12GB 门禁，可在新的输出目录加
`--ort-cuda-profile memory_12gb` 做对照。该 profile 仅通过子进程环境变量启用
`kSameAsRequested + HEURISTIC + cudnn_conv_use_max_workspace=0`，生产默认仍保持 ONNX Runtime 默认值。
它不设置每个 session 的 `gpu_mem_limit`，因为多个独立 session 的该限制不是聚合显存预算。
跨 profile 测试时，用 `--quality-reference-report <默认 profile 的 tuning_report.json>` 固定质量基准，
避免每个 profile 只和自己的首个候选比较。

找不到安全候选属于有效标定结果，CLI 默认仍返回 `0` 并在报告中写入 `no_safe_candidate`；CI 若希望将其
作为门禁失败，可加 `--fail-if-no-safe-candidate`，此时返回 `2`。技术执行错误仍返回非零。
