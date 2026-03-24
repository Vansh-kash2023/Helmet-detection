from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, models, transforms


def canonical_class_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"helmet", "with_helmet", "person_with_helmet"}:
        return "helmet"
    if normalized in {"no_person", "noperson", "no_rider", "no__person"}:
        return "no_person"
    if normalized in {
        "no_helmet",
        "without_helmet",
        "person_no_helmet",
        "person_without_helmet",
        "nohelmet",
    }:
        return "no_helmet"
    return normalized


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, eval_transform


def create_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    classes: List[str],
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = {name: 0 for name in classes}
    class_total = {name: 0 for name in classes}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for i in range(labels.size(0)):
                label_idx = int(labels[i].item())
                pred_idx = int(preds[i].item())
                class_name = classes[label_idx]
                class_total[class_name] += 1
                if label_idx == pred_idx:
                    class_correct[class_name] += 1

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    per_class_acc = {
        class_name: (class_correct[class_name] / class_total[class_name]) if class_total[class_name] else 0.0
        for class_name in classes
    }

    return avg_loss, accuracy, per_class_acc


def create_loaders(
    data_root: Path,
    batch_size: int,
    image_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], List[str]]:
    train_transform, eval_transform = build_transforms(image_size)

    train_path = data_root / "train"
    val_path = data_root / "val"
    test_path = data_root / "test"

    split_layout_exists = train_path.exists() and val_path.exists() and test_path.exists()

    if split_layout_exists:
        train_ds = datasets.ImageFolder(train_path, transform=train_transform)
        val_ds = datasets.ImageFolder(val_path, transform=eval_transform)
        test_ds = datasets.ImageFolder(test_path, transform=eval_transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        raw_classes = train_ds.classes
        canonical_classes = [canonical_class_name(name) for name in raw_classes]
        return train_loader, val_loader, test_loader, canonical_classes, raw_classes

    # Fallback: class folders directly inside data_root.
    base_train_ds = datasets.ImageFolder(data_root, transform=train_transform)
    if len(base_train_ds) < 3:
        raise ValueError("Not enough images found in data root to create train/val/test splits.")

    base_eval_ds = datasets.ImageFolder(data_root, transform=eval_transform)
    indices = list(range(len(base_train_ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = max(1, int(0.7 * n_total))
    n_val = max(1, int(0.15 * n_total))
    n_test = n_total - n_train - n_val

    if n_test < 1:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    if not val_idx:
        val_idx = test_idx[:1]
    if not test_idx:
        test_idx = val_idx[:1]

    train_subset = Subset(base_train_ds, train_idx)
    val_subset = Subset(base_eval_ds, val_idx)
    test_subset = Subset(base_eval_ds, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    raw_classes = base_train_ds.classes
    canonical_classes = [canonical_class_name(name) for name in raw_classes]

    print(
        f"[INFO] Using automatic split from data root: "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )

    return train_loader, val_loader, test_loader, canonical_classes, raw_classes


def save_checkpoint(checkpoint_path: Path, model: nn.Module, classes: List[str], image_size: int) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "image_size": image_size,
    }
    torch.save(payload, checkpoint_path)


def train_command(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    checkpoint_path = Path(args.checkpoint)
    labels_path = Path(args.labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_loader, val_loader, _, classes, raw_classes = create_loaders(
        data_root,
        args.batch_size,
        args.image_size,
        args.seed,
    )
    print(f"[INFO] Raw folder classes: {raw_classes}")
    print(f"[INFO] Canonical classes used: {classes}")

    model = create_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device, classes)

        print(
            f"[EPOCH {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(checkpoint_path, model, classes, args.image_size)
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            labels_path.write_text(json.dumps(classes, indent=2), encoding="utf-8")
            print(f"[INFO] New best checkpoint saved to: {checkpoint_path}")

    print(f"[DONE] Training complete. Best val accuracy: {best_val_acc:.4f}")


def load_for_eval(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, List[str], int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classes = payload["classes"]
    image_size = int(payload.get("image_size", 224))

    model = create_model(num_classes=len(classes))
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(device)

    return model, classes, image_size


def test_command(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    checkpoint_path = Path(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model, classes, image_size = load_for_eval(checkpoint_path, device)

    _, _, test_loader, classes_from_data, raw_classes = create_loaders(
        data_root,
        args.batch_size,
        image_size,
        args.seed,
    )
    print(f"[INFO] Raw folder classes: {raw_classes}")

    if classes != classes_from_data:
        raise ValueError(
            "Class mismatch between checkpoint and dataset folders. "
            f"checkpoint={classes}, data={classes_from_data}"
        )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, per_class_acc = evaluate(model, test_loader, criterion, device, classes)

    print(f"[RESULT] test_loss={test_loss:.4f}")
    print(f"[RESULT] test_acc={test_acc:.4f}")
    print("[RESULT] per-class accuracy:")
    for class_name, score in per_class_acc.items():
        print(f"  - {class_name}: {score:.4f}")


def run_camera_command(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model, classes, image_size = load_for_eval(checkpoint_path, device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class_to_idx = {canonical_class_name(name): i for i, name in enumerate(classes)}

    if "no_person" not in class_to_idx or "no_helmet" not in class_to_idx or "helmet" not in class_to_idx:
        raise ValueError(
            "Model classes must include no_person, no_helmet, helmet. "
            f"Found: {classes}"
        )

    camera = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(args.camera_id)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam with camera-id={args.camera_id}")

    print("[INFO] Camera started. Press Ctrl+C to stop.")

    last_status = ""
    last_log_time = 0.0
    frame_count = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue

            frame_count += 1
            if frame_count % max(args.frame_stride, 1) != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()

            pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
            pred_label = canonical_class_name(classes[pred_idx])
            pred_conf = float(probs[pred_idx])

            now = time.time()
            should_log = pred_label != last_status or (now - last_log_time) >= args.log_interval
            if not should_log:
                continue

            if pred_conf < args.threshold:
                print(f"[INFO] Uncertain prediction confidence={pred_conf:.2f}")
            elif pred_label == "no_person":
                print(f"[INFO] No person detected confidence={pred_conf:.2f}")
            elif pred_label == "helmet":
                print(f"[SAFE] Person detected with helmet confidence={pred_conf:.2f}")
            elif pred_label == "no_helmet":
                print(f"[ALERT] Person detected without helmet confidence={pred_conf:.2f}")
            else:
                print(f"[INFO] Predicted class={pred_label} confidence={pred_conf:.2f}")

            last_status = pred_label
            last_log_time = now
    except KeyboardInterrupt:
        print("\n[INFO] Stopping camera inference.")
    finally:
        camera.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helmet classification model (terminal only)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data-root", default="data", help="Root folder containing train/val/test")
    train_parser.add_argument("--checkpoint", default="models/helmet_classifier.pth", help="Path to save model checkpoint")
    train_parser.add_argument("--labels", default="models/helmet_classifier_labels.json", help="Path to save class labels")
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train_command)

    test_parser = subparsers.add_parser("test", help="Test trained model")
    test_parser.add_argument("--data-root", default="data", help="Root folder containing train/val/test")
    test_parser.add_argument("--checkpoint", default="models/helmet_classifier.pth", help="Path to trained model checkpoint")
    test_parser.add_argument("--batch-size", type=int, default=32)
    test_parser.add_argument("--seed", type=int, default=42)
    test_parser.set_defaults(func=test_command)

    run_parser = subparsers.add_parser("run", help="Run live webcam inference in terminal")
    run_parser.add_argument("--checkpoint", default="models/helmet_classifier.pth", help="Path to trained model checkpoint")
    run_parser.add_argument("--camera-id", type=int, default=0, help="Webcam id")
    run_parser.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold")
    run_parser.add_argument("--log-interval", type=float, default=1.0, help="Seconds between repeated logs")
    run_parser.add_argument("--frame-stride", type=int, default=2, help="Run inference on every Nth frame")
    run_parser.set_defaults(func=run_camera_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
