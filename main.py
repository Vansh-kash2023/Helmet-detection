from __future__ import annotations

import argparse
import logging
from pathlib import Path

from models.helmet_model import train_model
from models.yolo_pipeline import run_realtime_camera


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple helmet detection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save model")
    train_parser.add_argument("--data-root", default="data")
    train_parser.add_argument("--checkpoint", default="models/helmet_classifier.pth")
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--head-epochs", type=int, default=5)
    train_parser.add_argument("--fine-tune-epochs", type=int, default=3)
    train_parser.add_argument("--head-lr", type=float, default=1e-3)
    train_parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--dropout", type=float, default=0.3)
    train_parser.add_argument("--force-cpu", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run realtime camera inference")
    run_parser.add_argument("--checkpoint", default="models/helmet_classifier.pth")
    run_parser.add_argument("--camera-id", type=int, default=0)
    run_parser.add_argument("--yolo-weights", default="yolov8n.pt")
    run_parser.add_argument("--person-conf", type=float, default=0.35)
    run_parser.add_argument("--helmet-conf", type=float, default=0.55)
    run_parser.add_argument("--frame-stride", type=int, default=2)
    run_parser.add_argument("--force-cpu", action="store_true")
    run_parser.add_argument("--save-output", action="store_true")
    run_parser.add_argument("--output-path", default="outputs/camera_output.mp4")
    run_parser.add_argument("--no-show", action="store_true")

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
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
        return

    run_realtime_camera(
        classifier_checkpoint=Path(args.checkpoint),
        camera_id=args.camera_id,
        yolo_weights=args.yolo_weights,
        person_conf_threshold=args.person_conf,
        helmet_conf_threshold=args.helmet_conf,
        frame_stride=args.frame_stride,
        show_window=not args.no_show,
        save_output=args.save_output,
        output_path=Path(args.output_path),
        force_cpu=args.force_cpu,
    )


if __name__ == "__main__":
    main()
