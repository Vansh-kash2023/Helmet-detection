from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

LOGGER = logging.getLogger(__name__)
EXPECTED_CLASSES = ["helmet", "no_helmet"]


@dataclass
class TrainingArtifacts:
    checkpoint_path: Path
    class_to_idx: Dict[str, int]
    best_val_acc: float


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_tfms, eval_tfms


def _validate_dataset_layout(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    for split in ["train", "val", "test"]:
        split_path = data_root / split
        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing required split folder: {split_path}. "
                "Expected strict layout: data/train, data/val, data/test"
            )

        for class_name in EXPECTED_CLASSES:
            class_path = split_path / class_name
            if not class_path.exists():
                raise FileNotFoundError(
                    f"Missing class folder: {class_path}. "
                    "Only helmet and no_helmet classes are supported."
                )


def create_dataloaders(
    data_root: Path,
    batch_size: int,
    image_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int]]:
    _validate_dataset_layout(data_root)

    train_tfms, eval_tfms = build_transforms(image_size)

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tfms)

    if sorted(train_ds.classes) != sorted(EXPECTED_CLASSES):
        raise ValueError(
            f"Unexpected classes in train split: {train_ds.classes}. "
            f"Expected exactly: {EXPECTED_CLASSES}"
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes, train_ds.class_to_idx


def create_model(dropout: float = 0.3) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # MobileNetV3-Small classifier structure:
    # [Linear(576, 1024), Hardswish, Dropout, Linear(1024, 1000)]
    in_features = model.classifier[-1].in_features
    model.classifier[-2] = nn.Dropout(p=dropout)
    model.classifier[-1] = nn.Linear(in_features, len(EXPECTED_CLASSES))
    return model


def freeze_backbone(model: nn.Module) -> None:
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None,
) -> EpochMetrics:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    return EpochMetrics(
        loss=running_loss / max(total, 1),
        accuracy=running_correct / max(total, 1),
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int], List[List[float]]]:
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[List[float]] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())
            all_prob.extend(probs.detach().cpu().tolist())

    return (
        running_loss / max(total, 1),
        running_correct / max(total, 1),
        all_true,
        all_pred,
        all_prob,
    )


def train_model(
    data_root: Path,
    checkpoint_path: Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
    head_epochs: int = 5,
    fine_tune_epochs: int = 10,
    head_lr: float = 1e-3,
    fine_tune_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    force_cpu: bool = False,
) -> TrainingArtifacts:
    device = get_device(force_cpu=force_cpu)
    LOGGER.info("Device: %s", device)

    train_loader, val_loader, _, classes, class_to_idx = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )

    model = create_model(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0

    # Stage 1: train classifier head only.
    freeze_backbone(model)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(head_params, lr=head_lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    for epoch in range(1, head_epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_metrics.loss)

        LOGGER.info(
            "[HEAD %d/%d] train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            head_epochs,
            train_metrics.loss,
            train_metrics.accuracy,
            val_metrics.loss,
            val_metrics.accuracy,
        )

        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            save_checkpoint(checkpoint_path, model, class_to_idx, image_size)

    # Stage 2: unfreeze all layers and fine-tune.
    unfreeze_backbone(model)
    optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    for epoch in range(1, fine_tune_epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_metrics.loss)

        LOGGER.info(
            "[FINE %d/%d] train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            fine_tune_epochs,
            train_metrics.loss,
            train_metrics.accuracy,
            val_metrics.loss,
            val_metrics.accuracy,
        )

        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            save_checkpoint(checkpoint_path, model, class_to_idx, image_size)

    LOGGER.info("Best validation accuracy: %.4f", best_val_acc)
    return TrainingArtifacts(checkpoint_path=checkpoint_path, class_to_idx=class_to_idx, best_val_acc=best_val_acc)


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    class_to_idx: Dict[str, int],
    image_size: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "image_size": image_size,
        "model_name": "mobilenet_v3_small",
        "classes": EXPECTED_CLASSES,
    }
    torch.save(payload, checkpoint_path)
    LOGGER.info("Saved checkpoint: %s", checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, int], int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_to_idx = payload["class_to_idx"]
    image_size = int(payload.get("image_size", 224))

    if set(class_to_idx.keys()) != set(EXPECTED_CLASSES):
        raise ValueError(
            f"Checkpoint classes are invalid: {class_to_idx.keys()}. "
            f"Expected exactly: {EXPECTED_CLASSES}"
        )

    model = create_model(dropout=0.3)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_to_idx, image_size


def predict_person_crop(
    model: nn.Module,
    crop_bgr,
    device: torch.device,
    image_size: int,
    class_to_idx: Dict[str, int],
) -> Tuple[str, float]:
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    eval_tfms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = eval_tfms(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    label = idx_to_class[pred_idx]
    return label, confidence
