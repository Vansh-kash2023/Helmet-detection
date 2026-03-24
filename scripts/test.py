from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from models.helmet_model import create_dataloaders
from models.helmet_model import evaluate_model
from models.helmet_model import get_device
from models.helmet_model import load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate helmet classifier on strict test split")
    parser.add_argument("--data-root", default="data", help="Dataset root with train/val/test")
    parser.add_argument("--checkpoint", default="models/helmet_classifier.pth", help="Trained checkpoint path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--show-cm", action="store_true", help="Show confusion matrix in a GUI window")
    parser.add_argument(
        "--cm-image",
        default="outputs/confusion_matrix.png",
        help="Path to save confusion matrix image",
    )
    parser.add_argument("--no-save-cm", action="store_true", help="Skip saving confusion matrix image")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args()

    device = get_device(force_cpu=args.force_cpu)
    model, class_to_idx, image_size = load_checkpoint(Path(args.checkpoint), device)

    _, _, test_loader, classes, _ = create_dataloaders(
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
    )

    if class_to_idx != test_loader.dataset.class_to_idx:
        raise ValueError(
            "Class mapping mismatch between checkpoint and dataset. "
            f"checkpoint={class_to_idx}, dataset={test_loader.dataset.class_to_idx}"
        )

    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred, _ = evaluate_model(model, test_loader, criterion, device)

    print(f"[RESULT] test_loss={test_loss:.4f}")
    print(f"[RESULT] test_acc={test_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    print("[RESULT] confusion_matrix (rows=true, cols=pred):")
    print(cm)

    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("[RESULT] classification_report:")
    print(report)

    if args.show_cm or not args.no_save_cm:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required for confusion matrix plotting. "
                "Install with: pip install matplotlib"
            ) from exc

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
        ax.set_title("Helmet Classifier Confusion Matrix")
        fig.tight_layout()

        if not args.no_save_cm:
            cm_path = Path(args.cm_image)
            cm_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(cm_path, dpi=160)
            print(f"[RESULT] confusion matrix image saved: {cm_path}")

        if args.show_cm:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
