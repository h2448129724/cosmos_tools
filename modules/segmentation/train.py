import argparse
import sys
from pathlib import Path

# Support direct execution from any working directory:
# ``python modules/segmentation/train.py``.
TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))

from trainer import train_model
# python segmentation/train.py --img_dir D:\project\changrui\CAB-F\glue\train\images --ann_dir D:\project\changrui\CAB-F\glue\train\annotations --model microunet --image_size 256 --product inforcement --label_name 'glue' --pretrained_ckpt D:\project\changrui\cosmos\assets\weights\cab_f\cab_extract_glue.pth

def main():
    # python segmentation/train.py --img_dir D:\project\changrui\CAB-F\reinforcement\data_2208\images --ann_dir D:\project\changrui\CAB-F\reinforcement\data_2208\annotations --model microunet --image_size 2208 --product inforcement --label_name 'reinforcement'
    parser = argparse.ArgumentParser(description='UNet分割训练：兼容二分类和多标签分割')
    parser.add_argument('--img_dir',    type=str, required=True, help='图像目录')
    parser.add_argument('--ann_dir',    type=str, required=True, help='标注目录')
    parser.add_argument('--model',      type=str, default='microunet', help='选择模型，例如 microunet')
    parser.add_argument('--image_size', type=int, default=256, help='训练时统一缩放到的尺寸')
    parser.add_argument('--label_name', type=str, default='glue', help='要训练的标签名')
    parser.add_argument(
        '--task_type',
        type=str,
        default='binary',
        choices=['binary', 'multilabel'],
        help='binary=旧二分类softmax；multilabel=多通道sigmoid',
    )
    parser.add_argument(
        '--label_names',
        nargs='+',
        default=None,
        help='multilabel任务的标签列表，例如: --label_names ear knife circle',
    )
    parser.add_argument('--epochs',     type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--val_split',  type=float, default=0.1)
    parser.add_argument('--patience',   type=int, default=50)
    parser.add_argument('--save_dir',   type=str, default='checkpoints')
    parser.add_argument('--product',    type=str, default='CAB-F')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='学习率预热 epoch 数')
    parser.add_argument('--class_weight_mult', type=float, default=5.0, help='glue 类别权重放大倍数')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 进程数')
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='预训练权重路径，用于微调')
    parser.add_argument('--strict_load', action='store_true', help='严格加载预训练权重')
    args = parser.parse_args()

    print("=" * 50)
    if args.task_type == 'multilabel' and not args.label_names:
        parser.error('--task_type multilabel requires --label_names, for example: --label_names ear knife circle')

    target_labels = args.label_names if args.task_type == 'multilabel' else [args.label_name]

    print(f"任务类型: {args.task_type}")
    print("=" * 50)
    print(f"开始训练模型: {args.model}")
    print(f"训练尺寸: {args.image_size}")
    print(f"目标标签: {target_labels}")
    print(f"数据目录: {args.img_dir}")
    print(f"标注目录: {args.ann_dir}")

    train_model(
        img_dir=args.img_dir,
        ann_dir=args.ann_dir,
        model_name=args.model,
        image_size=args.image_size,
        target_label=args.label_name,
        target_labels=target_labels,
        task_type=args.task_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        patience=args.patience,
        model_save_dir=args.save_dir,
        product=args.product,
        warmup_epochs=args.warmup_epochs,
        class_weight_mult=args.class_weight_mult,
        num_workers=args.num_workers,
        pretrained_ckpt=args.pretrained_ckpt,
        strict_load=args.strict_load,
    )


if __name__ == '__main__':
    main()
