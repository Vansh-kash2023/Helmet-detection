# Helmet Detection System (YOLO + CNN)

Production-ready traffic helmet detection pipeline:

1. YOLO detects person bounding boxes.
2. Each person crop is passed to a MobileNetV3 classifier.
3. Classifier predicts only `helmet` or `no_helmet`.
4. Green box for helmet, red box for no helmet.

## Required Dataset Layout (Strict)

```text
data/
|-- train/
|   |-- helmet/
|   `-- no_helmet/
|-- val/
|   |-- helmet/
|   `-- no_helmet/
`-- test/
    |-- helmet/
    `-- no_helmet/
```


## Model And Training Strategy

- Backbone: MobileNetV3-Small (ImageNet pretrained)
- Head: dropout 0.3 + binary classifier
- Training stages:
  - Stage 1: freeze backbone, train classifier head
  - Stage 2: unfreeze all layers, fine-tune end-to-end
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau
- Augmentations: RandomResizedCrop, rotation, ColorJitter, horizontal flip

## Project Structure

```text
Helmet-detection/
|-- data/
|-- models/
|   |-- helmet_model.py
|   `-- yolo_pipeline.py
|-- scripts/
|   |-- train.py
|   |-- test.py
|   `-- run_camera.py
|-- main.py
|-- requirements.txt
`-- README.md
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Simple Usage (Recommended)

Train and save model: (Optional)

```bash
python main.py train --data-root data --checkpoint models/helmet_classifier.pth
```

Run realtime detection:

```bash
python main.py run --checkpoint models/helmet_classifier.pth --camera-id 0
```

## Optional Advanced Commands

Train with custom settings:

```bash
python scripts/train.py --data-root data --checkpoint models/helmet_classifier.pth --head-epochs 5 --fine-tune-epochs 3
```

If you want to skip fine-tuning completely:

```bash
python scripts/train.py --data-root data --checkpoint models/helmet_classifier.pth --head-epochs 5 --fine-tune-epochs 0
```

## Test

```bash
python scripts/test.py --data-root data --checkpoint models/helmet_classifier.pth
```

## Run Camera Pipeline

Main entrypoint:

```bash
python main.py run --checkpoint models/helmet_classifier.pth --camera-id 0 --person-conf 0.35 --helmet-conf 0.55 --frame-stride 2
```

Script entrypoint:

```bash
python scripts/run_camera.py --checkpoint models/helmet_classifier.pth --camera-id 0 --person-conf 0.35 --helmet-conf 0.55 --frame-stride 2
```

Useful options:

- `--yolo-weights yolov8n.pt`
- `--save-output --output-path outputs/camera_output.mp4`
- `--no-show`
- `--force-cpu`

## Libraries Used

- torch
- torchvision
- ultralytics
- opencv-python
- numpy
- Pillow
- scikit-learn

## Author
Vansh Kashyap