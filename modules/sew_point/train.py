r"""
Keypoint detection via heatmap regression (UNet + Attention).
Usage:
    python -m sew_point.train --img_dir /path/to/images --ann_dir /path/to/annotations
    python -m sew_point.train --img_dir /path/to/images --ann_dir /path/to/annotations --epochs 500 --batch_size 8
    python -m sew_point.train --predict image.bmp --model checkpoints/best.pth
    python -m sew_point.train --img_dir "D:\project\changrui\CAB-F\缝纫点检测\train\images" --ann_dir "D:\project\changrui\CAB-F\缝纫点检测\train\annotations" --epochs 500 --batch_size 8
"""

import argparse
import math
import os
import random

import cv2
import numpy as np
import torch
from scipy.ndimage import maximum_filter
from torch.utils.data import DataLoader

from .model_registry import DEFAULT_MODEL, get_model, model_choices
from .datasets import KeypointDataset

ROOT = os.path.dirname(os.path.abspath(__file__))


# ───────────────────────── Training ───────────────────────


IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")


def collect_samples(img_dir, ann_dir):
    samples = []
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    for name in sorted(os.listdir(ann_dir)):
        if not name.lower().endswith(".json"):
            continue
        jf = os.path.join(ann_dir, name)
        base = os.path.splitext(name)[0]
        for ext in IMG_EXTS:
            img_path = os.path.join(img_dir, base + ext)
            if os.path.exists(img_path):
                samples.append((img_path, jf))
                break
    return samples


def adaptive_wing_loss(pred, target, omega=14, theta=0.5, epsilon=1.0, alpha=2.1):
    """Adaptive Wing Loss for heatmap regression."""
    delta = (target - pred).abs()

    A = omega * (1 / (1 + (theta / epsilon) ** (alpha - target))) * \
        (alpha - target) * ((theta / epsilon) ** (alpha - target - 1)) / epsilon
    C = theta * A - omega * torch.log(1 + (theta / epsilon) ** (alpha - target))

    loss = torch.where(
        delta < theta,
        omega * torch.log(1 + (delta / epsilon) ** (alpha - target)),
        A * delta - C,
    )
    return loss


def train(args, epoch_callback=None, stop_event=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = collect_samples(args.img_dir, args.ann_dir)
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    print(f"Image dir: {args.img_dir}")
    print(f"Annotation dir: {args.ann_dir}")
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = KeypointDataset(train_samples, augment=True,
                               aug_multiplier=args.aug_multiplier,
                               sigma=args.sigma, img_size=args.img_size)
    val_ds = KeypointDataset(val_samples, augment=False,
                             sigma=args.sigma, img_size=args.img_size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=max(1, args.num_workers // 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model_name).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, (args.epochs - args.warmup_epochs))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def loss_fn(pred, target):
        awing = adaptive_wing_loss(pred, target)
        pos_weight = 1.0 + 15.0 * target
        weighted_awing = (pos_weight * awing).mean()

        eps = 1e-6
        pred_c = pred.clamp(eps, 1 - eps)
        gamma = 2.0
        bce_neg = -(pred_c ** gamma) * (1 - target) * torch.log(1 - pred_c)
        bce_pos = -5.0 * ((1 - pred_c) ** gamma) * target * torch.log(pred_c)
        focal = (bce_pos + bce_neg).mean()

        return weighted_awing + focal

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        if stop_event and stop_event.is_set():
            print("Training stopped by user.")
            break

        model.train()
        train_loss = 0.0
        for imgs, heatmaps in train_dl:
            imgs, heatmaps = imgs.to(device), heatmaps.to(device)
            pred = model(imgs)
            loss = loss_fn(pred, heatmaps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps in val_dl:
                imgs, heatmaps = imgs.to(device), heatmaps.to(device)
                pred = model(imgs)
                loss = loss_fn(pred, heatmaps)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_ds)

        scheduler.step()

        is_best = val_loss < best_val_loss

        if epoch_callback:
            epoch_callback(epoch, train_loss, val_loss, scheduler.get_last_lr()[0], is_best)

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
            f"{'  *best*' if is_best else ''}"
        )

        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "schema_version": 1,
                    "task": "sew_point",
                    "model_key": args.model_name,
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "best_val_loss": best_val_loss,
                },
                os.path.join(save_dir, "best.pth"),
            )

    torch.save(
        {
            "schema_version": 1,
            "task": "sew_point",
            "model_key": args.model_name,
            "model_state": model.state_dict(),
            "args": vars(args),
            "best_val_loss": best_val_loss,
        },
        os.path.join(save_dir, "last.pth"),
    )
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to {save_dir}/")


# ───────────────────────── Inference ──────────────────────


def detect_peaks(heatmap, threshold=0.5, cluster_dist=3):
    """Find local maxima, cluster nearby peaks, keep best per cluster."""
    from scipy.cluster.hierarchy import fcluster, linkage

    hm = heatmap.squeeze()
    local_max = maximum_filter(hm, size=5)
    peaks = (hm == local_max) & (hm > threshold)
    ys, xs = np.where(peaks)
    scores = hm[ys, xs]

    if len(xs) == 0:
        return []

    if len(xs) == 1:
        return [(xs[0].item(), ys[0].item(), scores[0].item())]

    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    Z = linkage(coords, method="complete", metric="euclidean")
    labels = fcluster(Z, t=cluster_dist, criterion="distance")

    kept = []
    for cid in np.unique(labels):
        mask = labels == cid
        idx = np.argmax(scores[mask])
        idxs = np.where(mask)[0]
        best = idxs[idx]
        kept.append((xs[best].item(), ys[best].item(), scores[best].item()))

    return kept


def predict_heatmap_tta(model, img_np, device):
    """TTA: original + 3 flips + 3 rotations, average heatmaps."""
    h, w = img_np.shape[:2]

    def to_tensor(img):
        return torch.from_numpy(
            np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
        ).unsqueeze(0).to(device)

    def infer(tensor):
        with torch.no_grad():
            return model(tensor).cpu().numpy()[0, 0]

    heatmaps = []

    # Original
    heatmaps.append(infer(to_tensor(img_np)))

    # Horizontal flip
    flipped_h = img_np[:, ::-1, :].copy()
    hm = infer(to_tensor(flipped_h))
    heatmaps.append(hm[:, ::-1])

    # Vertical flip
    flipped_v = img_np[::-1, :, :].copy()
    hm = infer(to_tensor(flipped_v))
    heatmaps.append(hm[::-1, :])

    # Both flips
    flipped_hv = img_np[::-1, ::-1, :].copy()
    hm = infer(to_tensor(flipped_hv))
    heatmaps.append(hm[::-1, ::-1])

    # 90/180/270 rotations
    for k in [1, 2, 3]:
        rot = np.rot90(img_np, k).copy()
        rot_resized = cv2.resize(rot, (w, h))
        hm = infer(to_tensor(rot_resized))
        hm_resized = cv2.resize(hm, (w, h))
        hm_back = np.rot90(hm_resized, 4 - k)
        heatmaps.append(hm_back)

    avg_hm = np.mean(heatmaps, axis=0)
    return avg_hm[np.newaxis, ...]


def predict(image_path, model_path=None, threshold=0.5, save_dir=None, model_name=DEFAULT_MODEL):
    if model_path is None:
        model_path = os.path.join(ROOT, "checkpoints", "best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_name = str(checkpoint.get("model_key") or model_name)
        state = checkpoint["model_state"]
    else:
        state = checkpoint
    model = get_model(model_name).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    img = cv2.imread(image_path)

    heatmap = predict_heatmap_tta(model, img, device)

    peaks = detect_peaks(heatmap, threshold=threshold)
    print(f"Detected {len(peaks)} points:")
    for x, y, score in peaks:
        print(f"  ({x}, {y}) score={score:.3f}")

    for x, y, score in peaks:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    base = os.path.splitext(image_path)[0]
    out_path = base + "_pred.png"
    cv2.imwrite(out_path, img)
    print(f"Visualization saved to {out_path}")

    hm_vis = (heatmap.squeeze() * 255).clip(0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
    hm_path = base + "_heatmap.png"
    cv2.imwrite(hm_path, hm_color)
    print(f"Heatmap saved to {hm_path}")

    return peaks


# ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Keypoint detection UNet training")
    parser.add_argument("--img_dir", type=str, default=None, help="训练图片目录")
    parser.add_argument("--ann_dir", type=str, default=None, help="训练标注目录")
    parser.add_argument("--predict", type=str, default=None,
                        help="单图推理路径")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", type=str, default=None,
                        help="推理时使用的模型路径")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                        choices=model_choices(), help="训练/推理模型结构")

    # Training params
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian heatmap sigma")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(ROOT, "checkpoints"))
    parser.add_argument("--aug_multiplier", type=int, default=6,
                        help="数据增强倍数")
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.predict:
        predict(args.predict, model_path=args.model,
                threshold=args.threshold, save_dir=args.save_dir,
                model_name=args.model_name)
    else:
        if not args.img_dir or not args.ann_dir:
            parser.error("--img_dir and --ann_dir are required for training")
        train(args)


if __name__ == "__main__":
    main()
