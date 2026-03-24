from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.yolo_pipeline import run_realtime_camera


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run realtime YOLO + helmet classifier pipeline")
    parser.add_argument("--checkpoint", default="models/helmet_classifier.pth", help="Trained classifier checkpoint")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--yolo-weights", default="yolov8n.pt", help="YOLO weights path or model name")
    parser.add_argument("--person-conf", type=float, default=0.35, help="YOLO person confidence threshold")
    parser.add_argument("--helmet-conf", type=float, default=0.55, help="Helmet classifier confidence threshold")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--output-path", default="outputs/camera_output.mp4")
    parser.add_argument("--no-show", action="store_true", help="Disable visualization window")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()

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
