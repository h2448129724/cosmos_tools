# sew_point_conntect

针点连边分类模块。

当前已接入训练工作台，子功能包括：

- 训练
- 单样本推理
- 批量预测
- 可视化评估
- 导出生产 split ONNX
- Sew Point + Connect ONNX 全流程可视化（普通图 / 大图中心滑窗）

## 数据要求

训练输入显式分为：

- `--image_dir`：图片目录
- `--annotation_dir`：边标注 `json` 目录

不再推荐通过单一 `data_dir` 自动推断内部结构。

## 训练

```bash
cd modules/sew_point_conntect
python train.py \
  --image_dir /data/sew_point_connect/images \
  --annotation_dir /data/sew_point_connect/annotations \
  --save_dir ../../artifacts/checkpoints/sew_point_conntect_train \
  --epochs 300 \
  --batch_size 16 \
  --hidden_dim 128 \
  --num_layers 3
```

常用参数：

- `--lr`
- `--weight_decay`
- `--warmup_epochs`
- `--val_ratio`
- `--hidden_dim`
- `--num_layers`
- `--dropout`
- `--k_neighbors`
- `--patch_width`
- `--patch_height`

## 单样本推理

```bash
cd modules/sew_point_conntect
python infer.py \
  --json_path /data/sew_point_connect/sample.json \
  --image_path /data/sew_point_connect/sample.png \
  --model_path ../../artifacts/checkpoints/sew_point_conntect_train/best.pth
```

## 批量预测

```bash
cd modules/sew_point_conntect
python batch_predict.py \
  --image_dir /data/sew_point_connect/images \
  --annotation_dir /data/sew_point_connect/annotations \
  --model_path ../../artifacts/checkpoints/sew_point_conntect_train/best.pth \
  --output_annotation_dir /data/sew_point_connect_pred/annotations \
  --vis_dir /data/sew_point_connect_pred/vis
```

## 可视化评估

```bash
cd modules/sew_point_conntect
python visualize_eval.py \
  --json_path /data/sew_point_connect/sample.json \
  --image_path /data/sew_point_connect/sample.png \
  --model_path ../../artifacts/checkpoints/sew_point_conntect_train/best.pth \
  --out_dir /data/sew_point_connect_eval
```

## ONNX 全流程可视化

工具箱中的“ONNX 全流程可视化”会先用 Sew Point ONNX 检测关键点，再用
Connect split ONNX 预测连边，最终同时保存叠加图和预测 JSON。

Connect 导出包含两个配套文件：

- `connector_patch.onnx`：把每条候选边附近裁取的 `3 x 24 x 96` 图像条带编码成特征向量。
- `connector.onnx`：接收点特征、候选边、几何特征及上述图像特征，执行图消息传递并输出连边 logits。

两个文件来自同一个 `.pth` checkpoint。导出程序分别用 `EdgePatchEncoder` 和
`EdgeGraphCore` 包装原始 `EdgeGraphNet` 的两个部分；默认会在
`connector.onnx` 同目录生成 `connector_patch.onnx`。

“图片类型”支持：

- `small`：整张训练尺寸图片单次关键点推理。
- `large`：使用通用 Sew Point `tile_size + stride` 重叠滑窗找点，合并重复点后再统一连边；不经过 density checker，也不会按黑色区域 skip。

Sew Point 训练数据由 OpenCV 读取并直接转为 CHW，因此该模型的输入通道契约是
`BGR`，全流程可视化与独立 Sew Point ONNX 推理保持同一预处理，不额外转换为 RGB。

## 导出 split ONNX

工具箱中的“导出 split ONNX”会从一个 Connector checkpoint 同时生成：

- `connector.onnx`：图网络核心。
- `connector_patch.onnx`：候选边图像 Patch Encoder。

默认使用 CPU 完成导出后的静态、动态输入数值一致性验证，因此不依赖本机 CUDA/cuDNN
动态库是否完整。选择 CUDA 时会通过 Cosmos 的 ONNX Runtime provider 自检安全回退。

## 输出说明

- 训练权重可保存到 `--save_dir`
- GUI 运行历史统一写入 `artifacts/runs/`
- 如果在 GUI 中右键“删除”历史，只会移动到项目 `.trash/`
