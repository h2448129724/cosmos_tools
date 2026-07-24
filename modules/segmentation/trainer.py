import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import BinarySegmentationDataset
from losses import DiceLoss, MultiLabelDiceLoss, dice_coef, multilabel_dice_coef
from model_registry import get_model


def _scheduler_pct_start(warmup_epochs: int, epochs: int) -> float:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    pct_start = float(warmup_epochs) / float(epochs)
    return min(0.95, max(0.01, pct_start))


def save_prediction_samples(
    model,
    dataloader,
    device,
    epoch,
    save_dir="training_samples",
    task_type: str = "binary",
    target_labels: list[str] | None = None,
):
    os.makedirs(save_dir, exist_ok=True)
    task_type = str(task_type).lower()
    target_labels = target_labels or ["glue"]
    model.eval()
    with torch.no_grad():
        imgs, masks = next(iter(dataloader))
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        if task_type == "multilabel":
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy().astype(np.uint8)
        else:
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > 0.5).cpu().numpy().astype(np.uint8)
        for i in range(min(4, len(imgs))):
            img = imgs[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img = img.astype(np.uint8)
            if task_type == "multilabel":
                n_labels = min(len(target_labels), masks.shape[1])
                plt.figure(figsize=(5 * (1 + n_labels), 10))
                plt.subplot(2, 1 + n_labels, 1)
                plt.imshow(img)
                plt.title("Original Image")
                plt.axis("off")
                for label_idx in range(n_labels):
                    label = target_labels[label_idx]
                    true_mask = masks[i, label_idx].cpu().numpy()
                    pred_mask = preds[i, label_idx]
                    plt.subplot(2, 1 + n_labels, 2 + label_idx)
                    plt.imshow(true_mask, cmap="gray")
                    plt.title(f"True {label}")
                    plt.axis("off")
                    plt.subplot(2, 1 + n_labels, 1 + n_labels + 2 + label_idx)
                    plt.imshow(pred_mask, cmap="gray")
                    plt.title(f"Pred {label}")
                    plt.axis("off")
            else:
                true_mask = masks[i].cpu().numpy()
                pred_mask = preds[i]
                pred_prob = probs[i].cpu().numpy()
                plt.figure(figsize=(20, 5))
                plt.subplot(141)
                plt.imshow(img)
                plt.title("Original Image")
                plt.axis("off")
                plt.subplot(142)
                plt.imshow(true_mask, cmap="gray")
                plt.title(f"True {target_labels[0]} Mask")
                plt.axis("off")
                plt.subplot(143)
                plt.imshow(pred_mask, cmap="gray")
                plt.title(f"Predicted {target_labels[0]} Mask")
                plt.axis("off")
                plt.subplot(144)
                plt.imshow(pred_prob, cmap="hot", alpha=0.7)
                plt.title(f"{target_labels[0]} Probability")
                plt.axis("off")
            plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.png"), dpi=150, bbox_inches="tight")
            plt.close()


def train_model(
    img_dir: str,
    ann_dir: str,
    model_name: str = "microunet",
    image_size: int = 1024,
    target_label: str = "glue",
    target_labels: list[str] | tuple[str, ...] | None = None,
    task_type: str = "binary",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 10,
    model_save_dir: str = "checkpoints",
    product: str = "UGE",
    warmup_epochs: int = 10,
    class_weight_mult: float = 5.0,
    num_workers: int = 4,
    pretrained_ckpt: str | None = None,
    strict_load: bool = False,
):
    from datetime import datetime

    today = datetime.now().strftime("%y%m%d")
    model_save_dir = os.path.join(model_save_dir, today)
    os.makedirs(model_save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_type = str(task_type).lower()
    if task_type not in {"binary", "multilabel"}:
        raise ValueError(f"Unsupported task_type: {task_type}")
    if target_labels is None:
        target_labels = [target_label]
    target_labels = [str(label) for label in target_labels]
    if task_type == "binary":
        target_labels = [target_label]
    elif not target_labels:
        raise ValueError("target_labels must not be empty for multilabel segmentation")

    img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    ann_paths = []
    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(ann_dir, f"{base}.json")
        png_path = os.path.join(ann_dir, f"{base}.png")
        if os.path.isfile(json_path):
            ann_paths.append(json_path)
        elif os.path.isfile(png_path):
            ann_paths.append(png_path)
        else:
            ann_paths.append(None)

    import random

    paired = list(zip(img_paths, ann_paths))
    random.shuffle(paired)
    img_paths, ann_paths = zip(*paired)

    n_val = int(len(img_paths) * val_split)
    train_ds = BinarySegmentationDataset(
        img_paths[n_val:],
        ann_paths[n_val:],
        augment=True,
        image_size=image_size,
        target_label=target_label,
        target_labels=target_labels,
        task_type=task_type,
    )
    val_ds = BinarySegmentationDataset(
        img_paths[:n_val],
        ann_paths[:n_val],
        augment=False,
        image_size=image_size,
        target_label=target_label,
        target_labels=target_labels,
        task_type=task_type,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers // 2), pin_memory=True
    )

    train_ds_all = BinarySegmentationDataset(
        img_paths,
        ann_paths,
        augment=False,
        image_size=image_size,
        target_label=target_label,
        target_labels=target_labels,
        task_type=task_type,
    )
    loader_all = DataLoader(train_ds_all, batch_size=1, shuffle=False, num_workers=0)
    n_classes = len(target_labels) if task_type == "multilabel" else 2
    missing_ann_count = sum(ann_path is None for ann_path in ann_paths)
    if missing_ann_count:
        print(f"有 {missing_ann_count} 张图片没有对应标注，按全背景样本处理")
    print("计算类别权重...")
    if task_type == "multilabel":
        pos_counts = torch.zeros(n_classes, dtype=torch.float)
        total_pixels = 0
        for _, mask_t in loader_all:
            pos_counts += mask_t.sum(dim=(0, 2, 3)).float()
            total_pixels += mask_t.shape[0] * mask_t.shape[2] * mask_t.shape[3]
        neg_counts = torch.full_like(pos_counts, float(total_pixels)) - pos_counts
        pos_weight = (neg_counts / (pos_counts + 1e-6)) * float(class_weight_mult)
        pos_weight = pos_weight.to(device)
        stats = ", ".join(f"{label}={pos_counts[i]:.0f}" for i, label in enumerate(target_labels))
        weights = ", ".join(f"{label}={pos_weight[i]:.4f}" for i, label in enumerate(target_labels))
        print(f"正样本像素统计: {stats}")
        print(f"BCE pos_weight: {weights}")
        primary_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(1, -1, 1, 1))
        dice_loss = MultiLabelDiceLoss()
    else:
        pix_counts = torch.zeros(n_classes, dtype=torch.float)
        for _, mask_t in loader_all:
            for class_id in range(n_classes):
                pix_counts[class_id] += torch.sum(mask_t == class_id)
        print(f"类别像素统计: 背景={pix_counts[0]:.0f}, {target_label}={pix_counts[1]:.0f}")
        class_weights = 1.0 / (pix_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * n_classes
        class_weights[1] *= class_weight_mult
        class_weights = class_weights.to(device)
        print(f"类别权重: 背景={class_weights[0]:.4f}, {target_label}={class_weights[1]:.4f}")
        primary_loss = nn.CrossEntropyLoss(weight=class_weights)
        dice_loss = DiceLoss()

    model = get_model(model_name, in_channels=3, n_classes=n_classes).to(device)
    if pretrained_ckpt:
        if not os.path.isfile(pretrained_ckpt):
            raise FileNotFoundError(f"预训练权重不存在: {pretrained_ckpt}")
        print(f"加载预训练权重: {pretrained_ckpt}")
        try:
            state = torch.load(pretrained_ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(pretrained_ckpt, map_location=device)

        if not strict_load:
            model_state = model.state_dict()
            skipped = []
            filtered_state = {}
            for key, value in state.items():
                if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
                    filtered_state[key] = value
                else:
                    skipped.append(key)
            state = filtered_state
        else:
            skipped = []
        load_result = model.load_state_dict(state, strict=strict_load)
        if strict_load:
            print("已按 strict=True 严格加载预训练权重")
        else:
            missing = list(load_result.missing_keys)
            unexpected = list(load_result.unexpected_keys)
            if missing:
                print(f"未匹配到的参数({len(missing)}): {missing[:10]}")
            if unexpected:
                print(f"多出的参数({len(unexpected)}): {unexpected[:10]}")
            if skipped:
                print(f"因形状不匹配跳过的参数({len(skipped)}): {skipped[:10]}")
            if not missing and not unexpected:
                print("预训练权重已完全匹配加载")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=_scheduler_pct_start(warmup_epochs, epochs),
    )

    best_val_dice = 0.0
    early_stopping_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss_ce = primary_loss(logits, masks)
            loss_dice = dice_loss(logits, masks)
            loss = loss_ce + loss_dice
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                l_ce = primary_loss(logits, masks)
                l_di = dice_loss(logits, masks)
                if task_type == "multilabel":
                    batch_dice = multilabel_dice_coef(logits, masks).item()
                else:
                    batch_dice = dice_coef(logits, masks).item()
                val_loss += (l_ce + l_di).item()
                val_dice += batch_dice
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(
            f"Epoch {epoch}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Dice={avg_val_dice:.4f}"
        )

        if epoch % 10 == 0:
            save_prediction_samples(
                model,
                val_loader,
                device,
                epoch,
                save_dir=os.path.join(model_save_dir, "training_samples"),
                task_type=task_type,
                target_labels=target_labels,
            )

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            early_stopping_counter = 0
            print(f"New best validation Dice: {best_val_dice:.4f}, saving model...")
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"{product}_{model_name}_best_model.pth"))
            save_prediction_samples(
                model,
                val_loader,
                device,
                "best",
                save_dir=os.path.join(model_save_dir, "training_samples"),
                task_type=task_type,
                target_labels=target_labels,
            )
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

        torch.save(model.state_dict(), os.path.join(model_save_dir, f"{product}_{model_name}_last_model.pth"))

    return model
