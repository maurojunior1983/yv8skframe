# YV8SKFRAME

<p align="center">
  <img src="assets/pipeline.svg" alt="YV8SKFRAME pipeline" width="100%">
</p>

<p align="center">
  <b>Detection of Picture-in-Picture advertising keyframes in soccer broadcast videos</b><br>
  Training, validation and inference pipeline using <b>YOLOv8</b>, <b>YOLO26</b> and <b>Vision Transformer (ViT)</b>.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue">
  <img alt="Ultralytics" src="https://img.shields.io/badge/Ultralytics-YOLO-orange">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-video_processing-green">
  <img alt="Research" src="https://img.shields.io/badge/Application-sports_video_analysis-purple">
</p>

---

## Overview

`YV8SKFRAME` is a research-oriented repository for detecting the **start** and **end** keyframes of graphical **Picture-in-Picture (PiP) advertisements** in soccer match broadcasts.

The project compares three visual learning approaches:

- **YOLOv8** - object detection baseline for localizing PiP advertising regions.
- **YOLO26** - recent YOLO-family detector evaluated under the same temporal post-processing pipeline.
- **Vision Transformer (ViT)** - frame-level image classifier used as a comparative model.

The pipeline was designed for an audiovisual editing scenario: given a soccer broadcast video, the system identifies the time interval in which a PiP advertisement appears, supporting semi-automatic or automatic video-editing workflows.

---

## Important path disclaimer

The examples in this repository assume that the full project folder is downloaded or cloned directly to the root of the `C:` drive:

```text
C:\YV8SKFRAME
```

Therefore, the expected paths are similar to:

```text
C:\YV8SKFRAME\Fr_DataSet_S_K_Frame
C:\YV8SKFRAME\Vi_DataSet_S_K_Frame
C:\YV8SKFRAME\Codes
```

If you place the repository elsewhere, update the paths in the notebooks and scripts accordingly.

### External video subsets

The validation and test video subsets are stored externally on Google Drive because the video files are too large to be kept directly in this GitHub repository. After downloading them, place the folders according to the structure shown below.

| Subset | Google Drive folder | Suggested local path | Description |
|---|---|---|---|
| `VALID` | [Download VALID videos](https://drive.google.com/drive/folders/1SAf2bOr3g8WqvnCdbMRNaHNIgrl_ZHjV?usp=drive_link) | `C:\YV8SKFRAME\Vi_DataSet_S_K_Frame\VALID` | Folder containing the video files used for validation and empirical temporal-parameter tuning. |
| `TEST` | [Download TEST videos](https://drive.google.com/drive/folders/1TM8sxrKdBjjbc76si9SqsdXeLJVVDQ3v?usp=drive_link) | `C:\YV8SKFRAME\Vi_DataSet_S_K_Frame\TEST` | Folder containing the video files used for final model inference and test-set reporting. |

---

## Repository structure

```text
YV8SKFRAME/
│
├── Fr_DataSet_S_K_Frame/
│   ├── train/                 # Training frames and YOLO labels
│   ├── valid/                 # Validation frames and YOLO labels
│   ├── test/                  # Test frames and YOLO labels
│   ├── data.yaml              # YOLO dataset configuration
│   ├── training_results/      # Trained model outputs
│   ├── YOLOv8_Detect_results/ # YOLOv8 inference spreadsheets
│   ├── YOLO26_Detect_results/ # YOLO26 inference spreadsheets
│   └── ViT_Detect_results/    # ViT inference spreadsheets
│
├── Vi_DataSet_S_K_Frame/
│   ├── VALID/                 # Validation videos downloaded from external Google Drive folder
│   └── TEST/                  # Test videos downloaded from external Google Drive folder
│
├── DataSet_SoccerKeyFrame - Org/
│   └── DataSet_SoccerKeyFrame.xlsx  # Ground-truth start/end timecodes
│
├── Codes/
│   ├── train/                 # Training scripts
│   ├── valid/                 # Validation and parameter tuning scripts
│   └── detect/                # Final inference scripts
│
├── YV8SKFRAME_walkthrough.ipynb
└── README.md
```

> **Note about generated-result folders:** the folders `training_results/`, `YOLOv8_Detect_results/`, `YOLO26_Detect_results/` and `ViT_Detect_results/` are intentionally kept in the repository as empty directories. Their contents are not included because trained weights, logs, plots and inference spreadsheets can become large. As new experiments are executed, the corresponding outputs will be automatically saved into these folders by the training and inference scripts.

---

## Research workflow

### 1. Train the models

Use the frame dataset and the training scripts:

```text
C:\YV8SKFRAME\Fr_DataSet_S_K_Frame\train
C:\YV8SKFRAME\Codes\train
```

Typical YOLO training command:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or YOLO("yolo26n.pt")

results = model.train(
    data="C:/YV8SKFRAME/Fr_DataSet_S_K_Frame/data.yaml",
    epochs=50,
    imgsz=(640, 360),
    batch=16,
    project="C:/YV8SKFRAME/Fr_DataSet_S_K_Frame/training_results",
    name="yolov8n_Soccer-Key-Frames"
)
```

For YOLO26, change the model and experiment name:

```python
model = YOLO("yolo26n.pt")
name = "yolo26n_Soccer-Key-Frames"
```

Typical ViT training setup:

```python
from transformers import ViTForImageClassification, AutoImageProcessor, TrainingArguments, Trainer

MODEL_ID = "google/vit-base-patch16-224-in21k"
OUTPUT_DIR = "C:/YV8SKFRAME/Fr_DataSet_S_K_Frame/ViT_training_results/vit_Soccer-Key-Frames"

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = ViTForImageClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    id2label={0: "without_pip", 1: "with_pip"},
    label2id={"without_pip": 0, "with_pip": 1}
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-4,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,
    remove_unused_columns=False
)

# The repository training scripts build the PyTorch datasets, collate function,
# metric computation and Trainer object using the frame folders and labels.
# See: C:/YV8SKFRAME/Codes/train
```

---

### 2. Validate and tune temporal parameters

Use the validation videos and validation scripts:

```text
C:\YV8SKFRAME\Vi_DataSet_S_K_Frame\VALID
C:\YV8SKFRAME\Codes\valid
```

The models produce frame-level detections. A temporal post-processing stage is then applied to convert these detections into PiP intervals. The parameters below were empirically adjusted on the validation subset.

| Parameter | YOLOv8 | YOLO26 | ViT |
|---|---:|---:|---:|
| `threshold_entry` | 0.90 | 0.30 | 0.40 |
| `threshold_exit` | 0.98 | 0.98 | 0.94 |
| `min_frames_entry` | 15 | 2 | 15 |
| `min_frames_exit` | 3 | 3 | 3 |
| `smoothing_window` | 5 | 5 | 5 |
| `min_duration_sec` | 2 | 2 | 2 |
| `min_gap_sec` | 2 | 2 | 2 |

In practical terms, this stage performs:

1. confidence smoothing over consecutive frames;
2. entry and exit decisions based on two different thresholds: `threshold_entry` and `threshold_exit`;
3. removal of very short false segments;
4. merging of detections separated by very small gaps;
5. conversion of frame indices into drop-frame timecodes.

---

### 3. Run final inference on the test set

After tuning the temporal parameters, use the test videos and detection scripts:

```text
C:\YV8SKFRAME\Vi_DataSet_S_K_Frame\TEST
C:\YV8SKFRAME\Codes\detect
```

The detection scripts generate spreadsheets with:

- detected start and end timecodes;
- ground-truth start and end timecodes;
- start/end false positive deviations;
- start/end false negative deviations;
- total frame error per video;
- gradual Fibonacci-based scores.

---

## Reported performance

The evaluated models achieved no false positives in the final test subset. The main difference was temporal recall, especially around the gradual appearance of the PiP graphic.

| Model | Precision (%) | Recall (%) | F1 (%) | mAP@0.5 (%) |
|---|---:|---:|---:|---:|
| YOLOv8 | 100.0 | 50.0 | 66.7 | 100.0 |
| YOLO26 | 100.0 | 51.7 | 68.1 | 100.0 |
| ViT | 100.0 | 52.0 | 68.4 | 100.0 |

---

## Jupyter Notebook walkthrough

A complete explanatory notebook is available in this repository:

```text
YV8SKFRAME_walkthrough.ipynb
```

It explains:

- how the folders are expected to be organized;
- how to check the dataset paths;
- how YOLOv8, YOLO26 and ViT are trained;
- how validation videos are used for parameter tuning;
- how inference is performed on the test videos;
- how the final Excel reports are generated.

The notebook is intentionally written as a guided walkthrough, so it can be used both for reproduction and for presentation of the project.

---

## Main dependencies

```bash
pip install ultralytics opencv-python pandas numpy matplotlib openpyxl torch torchvision transformers pillow
```

Depending on your environment, GPU-enabled PyTorch may require a specific CUDA-compatible installation. Check the official PyTorch installation selector before training large experiments locally.

---

## Minimal example: loading a trained YOLO model

```python
from ultralytics import YOLO

model_path = "C:/YV8SKFRAME/Fr_DataSet_S_K_Frame/training_results/yolo26n_Soccer-Key-Frames/weights/best.pt"
model = YOLO(model_path)

results = model("C:/YV8SKFRAME/Fr_DataSet_S_K_Frame/test/images/example.jpg", save=True)
```

---

## Citation

If this repository helps your work, please cite the associated paper:

```bibtex
@inproceedings{mendes2026pipkeyframes,
  title     = {Detecção de Keyframes de Publicidade Picture-in-Picture em Vídeos de Partidas de Futebol},
  author    = {Mendes Junior, Mauro Nunes and Andrade, Fábio Augusto de Alcantara and Passos, Wesley Lobato and Gois, Jonathan Nogueira and Lima, Amaro Azevedo de and Araujo, Gabriel Matos},
  booktitle = {XLIV Simpósio Brasileiro de Telecomunicações e Processamento de Sinais},
  year      = {2026},
  address   = {Salvador, BA, Brazil}
}
```

---

## Project status

This repository is part of an academic research project on automatic keyframe detection for sports broadcast editing. The current implementation focuses on soccer videos with PiP advertising overlays. Future extensions may include higher frame-rate sampling, additional detectors, temporal models, and broader sports-broadcast datasets.
