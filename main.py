from __future__ import annotations

import argparse

from models.helmet_model_cli import run_camera_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start live webcam helmet monitoring using trained model"
    )
    parser.add_argument(
        "--checkpoint",
        default="models/helmet_classifier.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam id")
    parser.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument(
        "--log-interval",
        type=float,
        default=1.0,
        help="Seconds between repeated logs",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Run inference on every Nth frame",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_camera_command(args)


if __name__ == "__main__":
    main()
