from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.helmet_model import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train helmet vs no_helmet classifier")
    parser.add_argument("--data-root", default="data", help="Dataset root with train/val/test")
    parser.add_argument("--checkpoint", default="models/helmet_classifier.pth", help="Output checkpoint path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--force-cpu", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args = build_parser().parse_args()
    artifacts = train_model(
        data_root=Path(args.data_root),
        checkpoint_path=Path(args.checkpoint),
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        head_lr=args.head_lr,
        fine_tune_lr=args.fine_tune_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        force_cpu=args.force_cpu,
    )

    print(f"[DONE] checkpoint={artifacts.checkpoint_path}")
    print(f"[DONE] best_val_acc={artifacts.best_val_acc:.4f}")
    print(f"[DONE] class_to_idx={artifacts.class_to_idx}")


if __name__ == "__main__":
    main()
