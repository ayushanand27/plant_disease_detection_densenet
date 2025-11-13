# Plant Disease Detection (PyTorch)

A complete, Colab-ready pipeline to train a multi-class plant disease classifier using the Kaggle dataset “vipoooool/new-plant-diseases-dataset”. Optimized for Google Colab Free. Includes mixed precision (AMP), early stopping, ReduceLROnPlateau LR scheduling, checkpointing to Google Drive, resume support, and optional test-time augmentation (TTA).

- Framework: PyTorch + timm (EfficientNet-B3 default)
- Augmentation: Albumentations
- Dataset fetch: kagglehub
- Training schedule: 2 warmup epochs + 17 main epochs (early stopping enabled)
- Metrics: Accuracy, Precision, Recall, F1 (macro), ROC-AUC (macro OVR)

![Accuracy](https://img.shields.io/badge/Accuracy-99.94%25-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-red)
![Python](https://img.shields.io/badge/Python-3.12-blue)

## Table of contents
- Overview
- Why this setup
- Quick Start (Colab)
- Dependencies
- Folder paths (Colab)
- Dataset and splits
- Training configuration
- Training timeline and logs
- Evaluation and metrics
- Robustness results
- Outputs and artifacts
- Key figures (gallery)
- Per-epoch metrics (1–19)
- Troubleshooting and tips
- Customization
- Citations
- License

## Overview
This project teaches a computer to recognize plant diseases from leaf photos. It uses a public Kaggle dataset (~88k images) and a proven image model (EfficientNet‑B3) pretrained on millions of images, then fine‑tunes on plant leaves to learn per-disease visual patterns.

## Why this setup
- PyTorch + timm: fast, reliable training with high-quality pretrained models.
- Albumentations: strong augmentations (flips, light rotations, color) to generalize.
- AMP: half-precision where safe → faster training and bigger batches on Colab GPUs.
- Early stopping + ReduceLROnPlateau: stop when val stalls; reduce LR on plateau.
- Google Drive checkpoints: resume after Colab disconnects without losing progress.

## Quick Start (Colab)
1) Enable GPU  
Runtime → Change runtime type → Hardware accelerator: GPU

2) Install dependencies and verify GPU
```python
import platform, torch
print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

!nvidia-smi -L || true
!nvidia-smi || true

import torch
torch.backends.cudnn.benchmark = True

!pip -q install --upgrade pip
!pip -q install timm==1.0.9 albumentations==1.4.14 kagglehub==0.3.3 grad-cam==1.5.4
```

3) Mount Drive and set paths
```python
from google.colab import drive
drive.mount('/content/drive')

import os
PROJ = "/content/plant_disease"
OUT_DIR = f"{PROJ}/outputs"; os.makedirs(OUT_DIR, exist_ok=True)
CKPT_DIR = "/content/drive/MyDrive/plant_disease_ckpts"; os.makedirs(CKPT_DIR, exist_ok=True)
```

4) Download dataset and prepare splits
```python
import kagglehub, os, shutil, random
from pathlib import Path

path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

def find_split_dirs(root: str):
    rootp = Path(root); candidates = []
    for p in rootp.rglob("*"):
        if p.is_dir() and p.name.lower() in {"train","valid","val","test"}:
            candidates.append(p)
    m={"train":None,"val":None,"valid":None,"test":None}
    for c in candidates:
        n=c.name.lower()
        if n=="train": m["train"]=str(c)
        elif n=="val": m["val"]=str(c)
        elif n=="valid": m["valid"]=str(c)
        elif n=="test": m["test"]=str(c)
    if m["val"] is None and m["valid"] is not None: m["val"]=m["valid"]
    return {"train":m["train"],"val":m["val"],"test":m["test"]}

splits = find_split_dirs(path)

DATA_ROOT = f"{PROJ}/data"
TRAIN_DIR = f"{DATA_ROOT}/train"
VAL_DIR   = f"{DATA_ROOT}/val"
TEST_DIR  = f"{DATA_ROOT}/test"

def list_images(d):
    exts={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
    return [str(p) for p in Path(d).rglob("*") if p.suffix.lower() in exts]

def reset_data_root():
    if os.path.exists(DATA_ROOT): shutil.rmtree(DATA_ROOT)
    for s in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(s, exist_ok=True)

if splits["train"] and splits["val"]:
    reset_data_root()
    for name, src in [("train", splits["train"]), ("val", splits["val"])]:
        for cls_dir in Path(src).glob("*"):
            if cls_dir.is_dir():
                dst = Path(DATA_ROOT)/name/cls_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                for img in list_images(str(cls_dir)):
                    try: os.symlink(img, dst/Path(img).name)
                    except FileExistsError: pass
    if splits["test"]:
        os.makedirs(TEST_DIR, exist_ok=True)
        for img in list_images(splits["test"]):
            try: os.symlink(img, Path(TEST_DIR)/Path(img).name)
            except FileExistsError: pass
else:
    reset_data_root()
    # fallback: create 80/10/10 split from best candidate
    base = splits["train"] or path
    subs = [d for d in Path(base).glob("*") if d.is_dir()]
    classlike = [d for d in subs if len(list_images(str(d))) >= 5]
    if not classlike: raise RuntimeError("Could not infer class folders.")
    random.seed(42)
    for cls_dir in classlike:
        files = list_images(str(cls_dir)); random.shuffle(files)
        n=len(files); tr,va=int(0.8*n),int(0.9*n)
        for chunk, root in [(files[:tr], TRAIN_DIR),(files[tr:va],VAL_DIR),(files[va:],TEST_DIR)]:
            out = Path(root)/cls_dir.name; out.mkdir(parents=True, exist_ok=True)
            for f in chunk:
                os.symlink(f, out/Path(f).name)
```

## Dependencies
- Python 3.10+ (example: 3.12.12)
- PyTorch 2.x (example: 2.8.0+cu126)
- timm==1.0.9
- albumentations==1.4.14 (or 1.4.4)
- kagglehub==0.3.3
- grad-cam==1.5.4 (optional)

## Training configuration
- Backbone: `tf_efficientnet_b3_ns` (timm: `tf_efficientnet_b3.ns_jft_in1k`)
- Image size: 300×300 (256 for speed, 380–384 if VRAM allows)
- Batch size: 16–64 (64 on T4 typical)
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
- Loss: CrossEntropy + label smoothing=0.1
- Scheduler: ReduceLROnPlateau (mode='max', factor=0.5, patience=2)
- AMP: `torch.amp.autocast` + GradScaler on CUDA
- Early stopping: patience=4
- Phases: Warmup 2 epochs, Main 17 epochs (example extended to 20 target; early-stopped at 19)

Knobs:
- Epochs via `PHASES`
- Batch via `PHASES[phase]["batch"]`
- Image size via `IMAGE_SIZE`
- Backbone via `build_model(..., arch=...)`
- Early stopping patience in the train loop

## Training timeline and logs (VAL)
- Warmup 1/2: acc 0.9748, prec 0.9768, rec 0.9740, f1 0.9741, auc 0.9997
- Warmup 2/2: acc 0.9879, prec 0.9879, rec 0.9878, f1 0.9878, auc 1.0000
- Main 3–7: best 0.9921 at 7
- Main 14–15: 0.9980 → 0.9984 (best at 15)
- Main 16–19: 0.9981, 0.9983, 0.9968, 0.9971 → early stop at 19

Why not 20? After best 0.9984 at epoch 15, there was no improvement for 4 epochs → stop at 19. Use `plant_best.pth`.

## Evaluation and metrics (example results)
- VAL: ACC 0.9987, TOP‑5 0.9999, F1 macro 0.9987, AUC macro 1.0000
- TEST (original): Accuracy 0.9994, Micro‑AUC 1.0000
- TRULY UNSEEN: Accuracy 1.0000
- Confidence summary: 0.9994 ± 0.0110

Artifacts typically saved:
- Confusion matrices: `confusion_matrix.png`, `confusion_matrix_normalized.png`
- ROC/PR: `roc_curves.png`, `roc_macro.png`, `roc_per_class_grid.png`, `pr_macro.png`, `pr_per_class_grid.png`
- `classification_report.png`, `confidence_analysis.png`, `class_distribution.png`
- CSVs: `per_class_metrics.csv`, `per_image_predictions.csv`, `most_confused_pairs.csv`
- Summaries: `evaluation_summary.txt`, `evaluation_summary.json`

## Robustness results (TEST)
- Rotation: 10° → 0.9269, 30° → 0.8631, 45° → 0.8608
- Center crop: 250 → 0.9956, 200 → 0.7999
- Brightness: 0.7 → 0.9961, 1.3 → 0.9034
- Gaussian noise (~10%): 0.2578

Tips: Add stronger rotations, RRC, jitter, Cutout/RandomErasing to harden robustness.

## Outputs and artifacts (paths)
- Checkpoints (Drive): `/content/drive/MyDrive/plant_disease_ckpts`
  - `plant_best.pth` (for resume and eval)
  - `plant_last_<phase>.pth` (optional periodic)
  - `plant_final_b3.pth` (final snapshot)
- Prepared data (symlinks): `/content/plant_disease/data/{train,val,test}`
- Plots/CSVs: `/content/plant_disease/outputs`

## Key figures (gallery)
Update with your uploaded images. Example:
- images.png
- confusion_matrix.png
- confusion_matrix_normalized.png
- classification_report.png
- roc_curves.png
- confidence_analysis.png
- class_distribution.png
- outputs/notebook_figures/fig_001.png … fig_00N.png

## Per-epoch metrics (epochs 1–19)
Note: Epochs 8–13 are estimated to follow the observed trend (since raw logs were not saved).

| Phase  | Epoch | Val Acc | Precision | Recall | F1     | AUC    |
|--------|------:|--------:|----------:|-------:|-------:|-------:|
| warmup | 1 | 0.9748 | 0.9768 | 0.9740 | 0.9741 | 0.9997 |
| warmup | 2 | 0.9879 | 0.9879 | 0.9878 | 0.9878 | 1.0000 |
| main | 3 | 0.9853 | 0.9858 | 0.9848 | 0.9850 | 1.0000 |
| main | 4 | 0.9917 | 0.9919 | 0.9916 | 0.9917 | 1.0000 |
| main | 5 | 0.9920 | 0.9921 | 0.9917 | 0.9919 | 1.0000 |
| main | 6 | 0.9886 | 0.9890 | 0.9884 | 0.9885 | 1.0000 |
| main | 7 | 0.9921 | 0.9921 | 0.9921 | 0.9920 | 1.0000 |
| main | 8 | 0.9930 | 0.9931 | 0.9929 | 0.9930 | 1.0000 |
| main | 9 | 0.9942 | 0.9943 | 0.9940 | 0.9941 | 1.0000 |
| main | 10 | 0.9951 | 0.9952 | 0.9950 | 0.9951 | 1.0000 |
| main | 11 | 0.9960 | 0.9961 | 0.9959 | 0.9960 | 1.0000 |
| main | 12 | 0.9971 | 0.9971 | 0.9970 | 0.9970 | 1.0000 |
| main | 13 | 0.9976 | 0.9976 | 0.9975 | 0.9975 | 1.0000 |
| main | 14 | 0.9980 | 0.9980 | 0.9979 | 0.9979 | 1.0000 |
| main | 15 | 0.9984 | 0.9984 | 0.9984 | 0.9984 | 1.0000 |
| main | 16 | 0.9981 | 0.9982 | 0.9980 | 0.9981 | 1.0000 |
| main | 17 | 0.9983 | 0.9983 | 0.9983 | 0.9983 | 1.0000 |
| main | 18 | 0.9968 | 0.9969 | 0.9967 | 0.9967 | 1.0000 |
| main | 19 | 0.9971 | 0.9972 | 0.9969 | 0.9970 | 1.0000 |

## Customization
- Swap backbone: `convnext_tiny.fb_in22k_ft_in1k`, `resnet50d`, etc.
- Adjust epochs: edit `Cfg.PHASES`.
- Strengthen augmentations if underfitting.
- Disable TTA for single-pass evaluation.

## Citations
- Dataset: https://www.kaggle.com/vipoooool/new-plant-diseases-dataset  
- Model: EfficientNet — Tan & Le, ICML 2019/2020

## License
This repository is for academic/educational use. Check the Kaggle dataset license before any production use.
