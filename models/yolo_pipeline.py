from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from models.helmet_model import get_device
from models.helmet_model import load_checkpoint
from models.helmet_model import predict_person_crop

LOGGER = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]
    label: str
    confidence: float


class HelmetYoloPipeline:
    def __init__(
        self,
        classifier_checkpoint: Path,
        yolo_weights: str = "yolov8n.pt",
        person_conf_threshold: float = 0.35,
        helmet_conf_threshold: float = 0.55,
        imgsz: int = 640,
        force_cpu: bool = False,
    ) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics") from exc

        self.device = get_device(force_cpu=force_cpu)
        self.person_conf_threshold = person_conf_threshold
        self.helmet_conf_threshold = helmet_conf_threshold
        self.imgsz = imgsz

        self.classifier, self.class_to_idx, self.image_size = load_checkpoint(classifier_checkpoint, self.device)
        self.person_detector = YOLO(yolo_weights)

        LOGGER.info("Pipeline initialized on device=%s", self.device)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[List[DetectionResult], np.ndarray]:
        detections: List[DetectionResult] = []
        annotated = frame_bgr.copy()

        # class 0 is person in COCO labels used by YOLOv8 pre-trained models.
        yolo_results = self.person_detector.predict(
            source=frame_bgr,
            conf=self.person_conf_threshold,
            classes=[0],
            imgsz=self.imgsz,
            verbose=False,
        )

        if not yolo_results:
            return detections, annotated

        result = yolo_results[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return detections, annotated

        h, w = frame_bgr.shape[:2]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            x1i = max(0, min(w - 1, int(x1)))
            y1i = max(0, min(h - 1, int(y1)))
            x2i = max(0, min(w - 1, int(x2)))
            y2i = max(0, min(h - 1, int(y2)))

            if x2i <= x1i or y2i <= y1i:
                continue

            crop = frame_bgr[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                continue

            label, conf = predict_person_crop(
                model=self.classifier,
                crop_bgr=crop,
                device=self.device,
                image_size=self.image_size,
                class_to_idx=self.class_to_idx,
            )

            if conf < self.helmet_conf_threshold:
                display_label = f"uncertain {conf:.2f}"
                color = (0, 215, 255)
            elif label == "helmet":
                display_label = f"helmet {conf:.2f}"
                color = (0, 200, 0)
            else:
                display_label = f"no_helmet {conf:.2f}"
                color = (0, 0, 255)

            cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), color, 2)
            cv2.putText(
                annotated,
                display_label,
                (x1i, max(20, y1i - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            detections.append(DetectionResult((x1i, y1i, x2i, y2i), label, conf))

        return detections, annotated


def run_realtime_camera(
    classifier_checkpoint: Path,
    camera_id: int = 0,
    yolo_weights: str = "yolov8n.pt",
    person_conf_threshold: float = 0.35,
    helmet_conf_threshold: float = 0.55,
    frame_stride: int = 2,
    show_window: bool = True,
    save_output: bool = False,
    output_path: Path | None = None,
    force_cpu: bool = False,
) -> None:
    pipeline = HelmetYoloPipeline(
        classifier_checkpoint=classifier_checkpoint,
        yolo_weights=yolo_weights,
        person_conf_threshold=person_conf_threshold,
        helmet_conf_threshold=helmet_conf_threshold,
        force_cpu=force_cpu,
    )

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera id={camera_id}")

    writer = None
    fps_start = time.time()
    frames_seen = 0
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frames_seen += 1
            if frames_seen % max(frame_stride, 1) != 0:
                continue

            processed += 1
            detections, annotated = pipeline.process_frame(frame)

            helmet_count = sum(1 for d in detections if d.label == "helmet")
            no_helmet_count = sum(1 for d in detections if d.label == "no_helmet")

            elapsed = max(time.time() - fps_start, 1e-6)
            fps = processed / elapsed
            LOGGER.info(
                "persons=%d helmet=%d no_helmet=%d fps=%.2f",
                len(detections),
                helmet_count,
                no_helmet_count,
                fps,
            )

            if save_output:
                if writer is None:
                    if output_path is None:
                        output_path = Path("outputs/camera_output.mp4")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0,
                        (w, h),
                    )
                writer.write(annotated)

            if show_window:
                cv2.imshow("Helmet Detection", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    except KeyboardInterrupt:
        LOGGER.info("Stopping realtime detection.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
