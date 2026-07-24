# sew_point

针点关键点检测模块。

当前模块已接入训练工作台，子功能包括：

- 训练
- 单张图片推理
- 批量小图推理
- 大图滑窗推理
- 导出 ONNX

## 数据要求

训练输入显式分为两个目录：

- `--img_dir`：训练图片目录
- `--ann_dir`：训练标注目录

不再推荐只传一个根目录后自动探测。

## 训练

```bash
cd modules/sew_point
python train.py \
  --img_dir /data/sew_point/images \
  --ann_dir /data/sew_point/annotations \
  --epochs 500 \
  --batch_size 8 \
  --lr 1e-3 \
  --val_ratio 0.15 \
  --save_dir ../../artifacts/checkpoints/sew_point_train
```

常用参数：

- `--img_size`
- `--sigma`
- `--batch_size`
- `--lr`
- `--epochs`
- `--val_ratio`
- `--aug_multiplier`
- `--save_dir`

## 单张图片推理

```bash
cd modules/sew_point
python inference.py \
  --image /data/sew_point/demo.bmp \
  --model ../../artifacts/checkpoints/sew_point_train/best.pth \
  --threshold 0.5 \
  --cluster_dist 3
```

## 批量小图推理

```bash
cd modules/sew_point
python tools/batch_infer.py \
  --input_dir /data/sew_point/images \
  --output_dir /data/sew_point/predictions \
  --model ../../artifacts/checkpoints/sew_point_train/best.onnx \
  --threshold 0.3 \
  --output_format master

说明：

- `--output_format master` 会直接输出 CAB-F 母格式，可直接给 `sew_point_conntect` 和 `img_tools` 后续流程使用
- `--output_format labelme` 保留旧行为，输出点标注 LabelMe JSON
```

## 大图滑窗推理

```bash
cd modules/sew_point
python tools/predict_large_image.py \
  --image /data/sew_point/large.png \
  --model ../../artifacts/checkpoints/sew_point_train/best.pth \
  --output /data/sew_point/large_pred.png \
  --tile-size 256 \
  --stride 192 \
  --batch-size 32
```

## 导出 ONNX

```bash
cd modules/sew_point
python export_onnx.py \
  --model ../../artifacts/checkpoints/sew_point_train/best.pth \
  --output ../../artifacts/checkpoints/sew_point_train/best.onnx \
  --input_size 256
```

## 说明

- GUI 中的训练历史会统一写入 `artifacts/runs/`
- 如果通过 GUI 生成命令，可以直接复制到服务器执行
- 模块自身脚本仍可单独使用
