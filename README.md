# Helmet Detection For Traffic Safety (Terminal Only)

## Problem Statement
Road safety systems need a reliable way to detect whether a rider is wearing a helmet in real time.
Manual monitoring is difficult, error-prone, and not scalable across continuous camera feeds.

This project solves a 3-class classification problem from camera frames:

- `no_person`: no rider/person present in the frame
- `no_helmet`: rider/person present but no helmet
- `helmet`: rider/person present with helmet

## Solution Implemented
We built a complete terminal-based ML workflow:

1. Train a deep learning classifier on your dataset using GPU (CUDA) if available.
2. Save the trained model in the `models` folder.
3. Run a root-level realtime inference command (`main.py`) that reads webcam frames and prints live decisions in terminal.

### Realtime Inference Output
When running `main.py`, the system prints:

- `[INFO] No person detected ...`
- `[ALERT] Person detected without helmet ...`
- `[SAFE] Person detected with helmet ...`

## Why This Is Effective

1. Real-time monitoring: works continuously on webcam feed.
2. Practical class design: directly models operational outcomes (`no_person`, `no_helmet`, `helmet`).
3. Fast inference: lightweight backbone supports near real-time behavior.
4. GPU acceleration: training uses CUDA on NVIDIA GPU (GTX 1650 in this setup).
5. Fully terminal-driven: simple commands for training, testing, and deployment-like runtime.

## Model Used

- Architecture: `MobileNetV3-Small` (transfer learning using `torchvision.models`)
- Task: 3-class image classification
- Training details:
  - Cross entropy loss
  - Adam optimizer
  - Standard image normalization
  - Data augmentation (flip, rotation, color jitter)

Saved model artifacts:

- `models/helmet_classifier.pth`
- `models/helmet_classifier_labels.json`

## Libraries Used

- `torch` and `torchvision`: model training and inference
- `opencv-python`: webcam capture for realtime predictions
- `numpy`: numerical operations
- `Pillow`: image backend used by torchvision transforms

See pinned versions in `requirements.txt`.

## Project Directory Structure

```text
Helmet-detection/
|-- data/
|   |-- Helmet/
|   |-- Person_no_helmet/
|   `-- no_person/
|-- models/
|   |-- helmet_model_cli.py
|   |-- helmet_classifier.pth
|   `-- helmet_classifier_labels.json
|-- main.py
|-- requirements.txt
`-- README.md
```

## Clone And Run

### 1. Clone Repository

```bash
git clone https://github.com/Vansh-kash2023/Helmet-detection.git
cd Helmet-detection
```

### 2. Create And Activate Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train Model (Optional)

```bash
python models/helmet_model_cli.py train --data-root data --epochs 12 --batch-size 32 --checkpoint models/helmet_classifier.pth --labels models/helmet_classifier_labels.json
```

### 5. Evaluate Model (Optional)

```bash
python models/helmet_model_cli.py test --data-root data --checkpoint models/helmet_classifier.pth --batch-size 32
```

### 6. Run Realtime Webcam Prediction

```bash
python main.py --checkpoint models/helmet_classifier.pth --camera-id 0
```

Stop runtime with `Ctrl+C`.

Optional runtime flags:

- `--threshold 0.55`
- `--log-interval 1.0`
- `--frame-stride 2`

What each flag means:

- `--threshold 0.55`: Minimum confidence required to trust a prediction. If model confidence is below this value, output is shown as uncertain.
- `--log-interval 1.0`: Prevents repeated logs from printing too frequently. For the same class, logs are throttled to roughly once every 1 second.
- `--frame-stride 2`: Runs inference on every 2nd frame instead of every frame, reducing GPU/CPU usage and improving runtime smoothness.

Typical tuning guidance:

- Increase `--threshold` to reduce false positives.
- Decrease `--threshold` if you are missing true detections.
- Increase `--frame-stride` on slower machines.
- Decrease `--frame-stride` for faster, more responsive detection.

## Dataset Used

This project uses the helmet dataset from Kaggle:

- https://www.kaggle.com/datasets/andrewmvd/helmet-detection

For this implementation, data is organized into 3 class folders used by the classifier:

- `Helmet`
- `Person_no_helmet`
- `no_person`

## Author
Vansh Kashyap