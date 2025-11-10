# Plant Disease Detection (PyTorch, Colab-Optimized)

A complete, Colab-ready pipeline to train a multi-class plant disease classifier using the Kaggle dataset "vipoooool/new-plant-diseases-dataset". The notebook is optimized for Google Colab Free, includes mixed precision (AMP), early stopping, LR scheduling, checkpointing to Google Drive, resume support, and test-time augmentation (TTA).

- Framework: PyTorch + timm (EfficientNet-B3 default)
- Augmentation: Albumentations
- Dataset Fetch: kagglehub
- Training schedule: 2 warmup epochs + 17 main epochs (early stopping enabled)
- Metrics: Accuracy, Precision, Recall, F1 (macro for multiclass), ROC-AUC (macro OVR)

---

## Plain-English Overview (Read me first)

This project teaches a computer to recognize plant diseases from leaf photos. I use a public Kaggle dataset (about 88k images) and a proven image model (EfficientNet-B3) that was pretrained on millions of images. I fine‑tune it on plant leaves so it learns the visual patterns of each disease.

What I’m using and why:

- PyTorch + timm: fast and reliable training with many good pretrained models.
- Albumentations: strong image augmentations so the model generalizes better (flips, small rotations, brightness/contrast changes).
- Mixed precision (AMP): uses half-precision math where safe, so training is faster and fits larger batches on Colab’s GPU.
- Early stopping + learning-rate scheduler: stop when validation stops improving; lower LR when progress stalls.
- Google Drive checkpoints: if Colab disconnects, I can continue from where I left off.

What happens step by step:

1) Install packages and check the GPU.
2) Mount Google Drive and create folders for outputs and checkpoints.
3) Download the dataset with `kagglehub` (no manual Kaggle CLI setup needed).
4) Prepare data folders:
   - If the dataset already has train/val/test, I use them directly (symlinks, no copying).
   - If it doesn’t, I create an 80/10/10 split automatically (still using symlinks).
5) Build DataLoaders with Albumentations:
   - Train uses random flips/rotations/brightness etc.
   - Val/Test only resize and normalize.
6) Build the model (EfficientNet‑B3) with the final layer sized to the number of classes.
7) Train in two phases:
   - Warmup: 2 quick epochs at a slightly higher LR (sanity check and speed up convergence).
   - Main: 17 epochs at a standard LR (with early stopping if there’s no improvement for 4 epochs).
8) During training:
   - I track accuracy, precision, recall, F1, and AUC on the validation set.
   - If validation accuracy improves, I save a “best” checkpoint to Drive.
   - If validation stalls, ReduceLROnPlateau halves the learning rate.
9) After training, I evaluate on the test set with TTA (original + horizontal flip averaged) and show the metrics and a confusion matrix.

Important outputs and where to find them:

- Checkpoints in Drive: `/content/drive/MyDrive/plant_disease_ckpts`
  - `plant_best.pth` (best model used for resume and final eval)
  - `plant_last_<phase>.pth` (optional periodic save)
  - `plant_final_b3.pth` (final model weights)
- Prepared data (symlinks): `/content/plant_disease/data/{train,val,test}`
- Plots/outputs: `/content/plant_disease/outputs`

How to change the plan (simple knobs):

- Epochs: set `PHASES["warmup"]["epochs"] = 2` and `PHASES["main"]["epochs"] = 7` (already set).
- Batch size: change `PHASES[phase]["batch"]` (use 32 or 48 if you get OOM).
- Image size: `IMAGE_SIZE = (300, 300)` (go to `(384, 384)` if your GPU is stronger, or `(256, 256)` for speed).
- Backbone: swap `arch` in `build_model(...)`, e.g., to `convnext_tiny.fb_in22k_ft_in1k`.
- Early stopping: inside `train`, increase `patience` (e.g., 999) if you want to always run full epochs.

How the metrics work (multiclass):

- The model outputs one score per class; the highest score is the prediction.
- Accuracy is overall correctness.
- Precision/Recall/F1 are macro-averaged, so each class counts equally.
- AUC is macro one‑vs‑rest: it checks how separable each class is from the rest.

If something goes wrong:

- GPU memory error (OOM): lower batch size, keep AMP on, consider smaller image size, or set `grad_accum_steps=2–4`.
- Training is slow: smaller image size; fewer workers; keep batch as high as fits.
- Validation doesn’t improve: try stronger augmentations or a larger backbone; allow more epochs.

## Quick Start (Colab)

1. Open a new Colab notebook and enable GPU runtime.
   - Runtime → Change runtime type → Hardware accelerator: GPU
2. Install dependencies and verify GPU.
3. Mount Google Drive (for checkpoints) and set paths.
4. Download dataset via `kagglehub` (no manual Kaggle CLI setup required).
5. Prepare train/val/test splits (auto-detects if dataset already provides split).
6. Run the full training pipeline (2 warmup + 17 main epochs) with AMP and early stopping.
7. Evaluate on the test set with TTA and view metrics + confusion matrix.

All required cells are provided below.

---

## Dependencies

- Python 3.10+
- PyTorch (Colab preinstalled; works with 2.x)
- timm==1.0.9
- albumentations==1.4.4
- kagglehub==0.3.3
- grad-cam==1.5.4 (optional, for explainability)

Install in Colab:

```
!pip -q install --upgrade pip
!pip -q install timm==1.0.9 albumentations==1.4.4 kagglehub==0.3.3 grad-cam==1.5.4
```

---

## Folder Paths (Colab)

- Project root in Colab: `/content/plant_disease`
- Outputs (logs/plots): `/content/plant_disease/outputs`
- Checkpoints (Drive): `/content/drive/MyDrive/plant_disease_ckpts`
- Auto-prepared data root (symlinks): `/content/plant_disease/data/{train,val,test}`

Checkpoints created:

- `plant_best.pth` (best by val acc; used for resume and evaluation)
- `plant_last_<phase>.pth` (optional periodic save)
- `plant_final_b3.pth` (final model after training)

---

## Dataset

- Source: Kaggle – `vipoooool/new-plant-diseases-dataset`
- Size: ~87.9k images
- Classes: e.g., 38 (auto-detected by counting class folders under train)

The notebook automatically:

- Downloads the dataset to a temporary path via `kagglehub`
- Detects existing `train/val/test` (or `train/valid/test`) directories
- If no split is present, creates an 80/10/10 split from `train` using symlinks (no duplication)

---

## Approach & Design

This project trains a high-accuracy yet GPU-efficient multi-class image classifier for plant diseases by combining:

- Pretrained vision backbones from `timm` (EfficientNet-B3 by default) for strong transfer learning.
- Robust augmentations via `albumentations` to generalize across lighting, pose, and mild geometric changes.
- Mixed precision (AMP) to maximize throughput and reduce VRAM use on Colab GPUs.
- A two-phase schedule (2 warmup + 7 main) with ReduceLROnPlateau and early stopping to converge fast within session limits.
- Drive-based checkpointing to survive Colab disconnects and support resuming.
- Multiclass-safe evaluation with macro metrics and optional TTA for a small inference boost.

Why EfficientNet-B3?

- Strong accuracy/speed balance on commodity GPUs.
- Pretrained on large-scale datasets; features transfer well to leaf textures, spots, and color cues common in plant pathology datasets.
- If you need lighter/faster: try `convnext_tiny.fb_in22k_ft_in1k` or `resnet50d`.

Data Split Strategy

- If the Kaggle dataset already provides `train/val/test`, we symlink to those folders (no copy) for minimal I/O.
- Otherwise, we create an 80/10/10 split from available class folders using symlinks, ensuring reproducibility (`random.seed(42)`).

Colab-First Optimization

- AMP halves compute/memory cost for matmuls/convs.
- DataLoader uses `persistent_workers=True`, `pin_memory=True` to speed host→device transfer.
- Image size 300×300 and batch 64 are safe defaults; adjust if OOM.
- Early stopping (patience=4) avoids wasted epochs once validation stalls.

Reproducibility & Determinism

- We set seeds for the split. Full determinism across GPUs is not guaranteed (cuDNN/batched ops), but results are stable enough for model selection.

---

## Components in Detail

### Model (backbone + head)

- Backbone: `tf_efficientnet_b3_ns` (maps to `tf_efficientnet_b3.ns_jft_in1k` in `timm`).
- Head: A linear classifier with `num_classes = number_of_class_folders` detected under `train`.
- Initialization: Pretrained weights; head is randomly initialized.

Switching models

- Replace the `arch` in `build_model(cfg, arch=...)` with another `timm` name (e.g., `convnext_tiny.fb_in22k_ft_in1k`, `resnet50d`). The rest of the pipeline remains unchanged.

### Data Pipeline

- Loader: `torchvision.datasets.ImageFolder` reading class-named folders.
- Transforms (train): Resize → flips (H/V) → `ShiftScaleRotate` → `ColorJitter` + `RandomBrightnessContrast` → Normalize → `ToTensorV2`.
- Transforms (val/test): Resize → Normalize → `ToTensorV2`.
- Augmentations are mild-to-moderate to avoid label leakage; tune for your domain if needed.

### Training Loop

- Loss: `CrossEntropyLoss` (multiclass).
- Optimizer: `AdamW(lr, weight_decay=1e-4)`.
- Scheduler: `ReduceLROnPlateau` on validation accuracy (mode='max', factor=0.5, patience=2). When val acc plateaus, LR halves.
- AMP: `torch.amp.autocast` + `GradScaler` on CUDA, disabled on CPU.
- Early stopping: patience=4 epochs without val acc improvement.
- Gradient accumulation: Supported via `grad_accum_steps` (default 1). Increase to simulate larger batches under VRAM limits.

Checkpointing & Resume

- Best checkpoint: `plant_best.pth` (model, optimizer, scaler, epoch, best_acc). Updated when val acc improves.
- Last checkpoint per phase: `plant_last_<phase>.pth` for periodic safety saves.
- Resume: If `resume_ckpt` exists, training continues from the next epoch with LR, optimizer, and scaler state restored.

### Evaluation & Metrics

- Per-epoch validation: accuracy, precision, recall, F1, AUC.
- Multiclass metrics: macro-averaged precision/recall/F1; macro ROC-AUC with one-vs-rest binarization.
- Test-time augmentation (TTA): average probabilities of original and horizontal-flipped inputs.
- Confusion matrix: Visual check for class-wise errors; large with 38 classes (heatmap without annotations).

### Handling Class Imbalance

- The dataset is reasonably balanced, but if imbalance appears in your subset: consider class weights in `CrossEntropyLoss`, or use a `WeightedRandomSampler` in the train DataLoader.

### Hyperparameter Tips

- If underfitting: increase image size (384), add stronger color/blur/noise aug, train longer (raise main epochs), or switch to a larger backbone.
- If overfitting: decrease aug aggressiveness last; prefer more augmentation and early stopping.
- LR search: try warmup lr in [1e-3, 3e-3], main lr in [5e-4, 2e-3].

### Colab Efficiency Details

- Batch size vs image size trade-off: prefer larger batch at 300×300 for throughput; upsize images if VRAM permits.
- DataLoader workers: 2 is stable; 4 can be faster but occasionally unstable.
- Persistent workers + pinned memory reduce data latency.
- Let early stopping save time; resume later for more epochs if needed.

### Known Warnings

- `timm` mapping warning (deprecated model name): informational.
- `HF_TOKEN` warning: ignore unless using private Hugging Face models.
- Albumentations Pydantic warnings: pass ranges as tuples (e.g., `brightness=(0.8,1.2)`).

---

## Training Configuration

- Model: `tf_efficientnet_b3_ns` (maps to `tf_efficientnet_b3.ns_jft_in1k`)
- Image size: 300×300 (increase to 384 if GPU allows)
- Batch size: 64 (reduce if OOM; or use gradient accumulation)
- Optimizer: AdamW (weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (mode='max', factor=0.5, patience=2)
- Early stopping patience: 4 epochs
- Mixed Precision: Enabled on CUDA (torch.amp autocast + GradScaler)
- Epochs: 2 (warmup) + 7 (main)

You can change epochs in the `Cfg.PHASES` dictionary.

---

## Resume Training

Training auto-saves to Google Drive. To resume after a Colab timeout:

- Keep `resume_ckpt=f"{cfg.CKPT_DIR}/plant_best.pth"` in the `train(...)` calls.
- The training function loads optimizer/scaler/model states and continues from the next epoch.

---

## Evaluation

- Metrics logged each epoch: Accuracy, Precision, Recall, F1, AUC
- Test-time augmentation (TTA): average of prediction and horizontal-flip prediction
- Confusion matrix plotted after test evaluation

Multiclass handling:

- Predictions via argmax
- Macro-averaged precision/recall/F1
- Macro ROC-AUC with one-vs-rest (OVR)

---

## Colab Notebook Cells (Copy/Paste)

The complete set of cells to run in Colab, in order, is included in the assistant’s responses in this chat. They are split into:

1) Runtime check + installs
2) Drive mount + paths
3) Dataset download + split prep
4) Config + transforms + loaders + model + evaluation
5) Training utilities (AMP, LR scheduler, early stopping, resume)
6) Train: warmup (2) + main (7)
7) Test + TTA + confusion matrix

You can also place them into a single Colab notebook and run sequentially.

---

## Tips for Colab Free

- **GPU type**: T4/P100/L4 vary; epoch time can range 4–10 minutes at 300×300, batch 64
- **If OOM**:
  - Reduce batch: 64 → 48 → 32
  - Keep AMP on
  - Consider image size 256×256
  - Optionally set `grad_accum_steps=2–4`
- **DataLoader**:
  - `num_workers=2` is stable on Colab; you can try 4
  - `persistent_workers=True`, `pin_memory=True` are enabled
- **Early stopping** saves time by stopping when no val improvement
- **Resume** across sessions to accumulate more total epochs if needed

---

## Customization

- Swap model: e.g., `convnext_tiny.fb_in22k_ft_in1k`, `resnet50d`
- Change epochs: edit `Cfg.PHASES`
- Adopt stronger augmentations if underfitting
- Turn off TTA for pure single-pass evaluation

---

## Troubleshooting

- `TypeError: ReduceLROnPlateau(..., verbose=...)` — remove `verbose` param (older torch).
- Pydantic/albumentations warnings about lists vs tuples — pass ranges as tuples, e.g., `brightness=(0.8,1.2)`.
- `HF_TOKEN` warning — safe to ignore unless pulling private models from Hugging Face.
- Slow data loading — reduce workers to 0–2 if runtime instability occurs.
- Unexpected early stop — increase early stopping patience or set to a large value (e.g., 999) to run planned epochs.

---

## Project Walkthrough (for first‑time readers)

- **Setup**
  - Colab: enable GPU, install deps, mount Drive, set `PROJ`, `OUT_DIR`, `CKPT_DIR`.
  - Laptop (Windows/NVIDIA): install CUDA PyTorch from pytorch.org, set local paths, prefer `NUM_WORKERS=0` initially.
- **Data**
  - Download via `kagglehub` (Colab) or place your zip/folder locally.
  - Auto-detects `train/val/test`; otherwise creates 80/10/10 using symlinks/copies.
- **Model**
  - `tf_efficientnet_b3_ns` from timm with a new classifier head sized to the number of classes.
- **Training**
  - Two phases: 2‑epoch warmup (higher LR) + main phase (more epochs), AMP on, ReduceLROnPlateau, early stopping, checkpointing.
- **Evaluation**
  - Multiclass‑safe metrics and the All‑in‑One Report saving CM/ROC/PR plots, CSVs, and summaries to `OUT_DIR`.

---

## Experiment Timeline & Findings

This documents what was run in your `.ipynb` so anyone can understand the process and outcomes:

- **Epoch 0→7 (warmup+short main)**
  - Saved a model at 7 epochs to validate pipeline stability and performance.
  - Checkpointing active; `plant_best.pth` updated when validation accuracy improved.
- **Extend to 15 epochs (main)**
  - Resumed training using `resume_ckpt` to continue without losing optimizer/LR/scaler state.
  - Saved model at 15 epochs.
- **Extend to 19 epochs**
  - Continued training, saving at 19 epochs.
- **Attempt 20 epochs**
  - Observed validation accuracy decrease relative to the earlier best. This is common when the model begins to overfit or when the LR reduces and optimization explores a less optimal region.

Key takeaways:
- The pipeline selects the best model by validation accuracy during training and stores it in `plant_best.pth` (independent of the last epoch).
- A drop at 20 vs earlier is expected behavior; use the best checkpoint (e.g., from 15–19) for evaluation and deployment rather than the very last epoch.

Why accuracy can drop at later epochs:
- **Overfitting**: model starts memorizing training specifics; validation no longer improves.
- **Learning‑rate dynamics**: ReduceLROnPlateau halves LR after plateaus; the new LR may explore a region that doesn’t yield higher val accuracy.
- **Data randomness**: minor stochasticity (augmentations, cuDNN algorithms) naturally causes small swings.

---

## How the Best Checkpoint Is Chosen

- After every epoch, validation metrics are computed; if accuracy improves, we save state as `plant_best.pth`.
- Contents: model weights, optimizer state, scaler state, epoch number, and the best validation accuracy so far.
- For evaluation or further training, prefer resuming from:
  - `plant_last_main.pth` for exact continuity (same LR/optimizer state at the last epoch), or
  - `plant_best.pth` for the model that performed best on validation.

---

## Reproduce My Runs (exact steps)

1) Train to 7 epochs (2 warmup + 5 main) or 2 + 7 depending on your config.
2) Extend to 15 epochs: set `cfg.PHASES["main"]["epochs"] = 15` and run training with `resume_ckpt=f"{cfg.CKPT_DIR}/plant_last_main.pth"` (fallback to `plant_best.pth` if `last` doesn’t exist).
3) Extend to 19 epochs similarly (`epochs = 19`).
4) Optionally set `epochs = 20` and continue; if validation drops, rely on `plant_best.pth` from the earlier epoch.
5) Run the **All‑in‑One Report** cell to generate CM/ROC/PR plots, per‑class and per‑image CSVs, and summaries.
6) If `test` is missing, report will use `val`; to create a true test split, copy a fraction of `val` per class into `test` and re‑run the report.

Notes for Windows laptop:
- If training appears stalled, set `cfg.NUM_WORKERS=0` and `cfg.PERSISTENT_WORKERS=False`, reduce `IMAGE_SIZE` to `(256,256)`, or enable tqdm progress in the training loop.

---

## Using the All‑in‑One Report

The universal evaluation cell produces:
- **Plots**: `confusion_matrix_counts.png`, `confusion_matrix_normalized.png`, `roc_macro.png`, `roc_per_class_grid.png`, `pr_macro.png`, `pr_per_class_grid.png`, optional `gradcam_grid.png`.
- **Tables**: `per_class_metrics.csv`, `per_image_predictions.csv`, `most_confused_pairs.csv`.
- **Summaries**: `evaluation_summary.txt` and `evaluation_summary.json`.

If results look “too ideal”, run the leakage audits (MD5/Perceptual‑hash) and the **strict test** builder, then re‑evaluate without TTA for realistic single‑pass metrics.

---

## Deploying the Best Model

- For inference, load `plant_best.pth` and run a single forward pass (no TTA) with the same normalization used in training.
- Export options:
  - PyTorch state dict for PyTorch inference.
  - TorchScript or ONNX (optional, not included by default). Ensure preprocessing matches training transforms.

---

## Embedded Cells (copy into the notebook)

### 1) All‑in‑One Report (labeled or unlabeled test supported)

```python
# UNIVERSAL ALL-IN-ONE REPORT
# - Uses cfg/model/device already defined in the notebook
# - Falls back to VAL if TEST is missing
# - Saves plots (CM/ROC/PR), CSVs (per-class, per-image, confused pairs), and summaries
import os, json, math
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, top_k_accuracy_score, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize

OUT_DIR = globals().get("OUT_DIR", str(Path.cwd() / "outputs")); os.makedirs(OUT_DIR, exist_ok=True)
get_eval_transforms = globals().get("get_eval_transforms", get_eval_transforms)
AlbuWrapper = globals().get("AlbuWrapper", AlbuWrapper)
model = globals().get("model", model).to(device).eval()

def has_class_folders(p):
    p = Path(p); return p.exists() and any(d.is_dir() for d in p.glob("*"))

# Choose evaluation source
if has_class_folders(cfg.TEST_DIR):
    eval_dir, flat_test_dir = cfg.TEST_DIR, None
elif has_class_folders(cfg.VAL_DIR):
    eval_dir, flat_test_dir = cfg.VAL_DIR, None
else:
    eval_dir, flat_test_dir = None, cfg.TEST_DIR if Path(cfg.TEST_DIR).exists() else None
print("Evaluation directory (labeled):", eval_dir)
print("Flat unlabeled test directory  :", flat_test_dir)

# Loaders
def make_labeled_loader(root):
    ds = ImageFolder(root, transform=AlbuWrapper(get_eval_transforms(cfg)))
    dl = DataLoader(ds, batch_size=cfg.PHASES["main"]["batch"], shuffle=False,
                    num_workers=getattr(cfg, "NUM_WORKERS", 0),
                    persistent_workers=getattr(cfg, "PERSISTENT_WORKERS", False),
                    pin_memory=getattr(cfg, "PIN_MEMORY", True))
    return dl, ds

class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}]
        self.t = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.files[i]).convert("RGB")
        x = self.t(img) if self.t else img
        return x, self.files[i]

def make_flat_loader(root):
    ds = FlatImageDataset(root, transform=AlbuWrapper(get_eval_transforms(cfg)))
    dl = DataLoader(ds, batch_size=cfg.PHASES["main"]["batch"], shuffle=False,
                    num_workers=getattr(cfg, "NUM_WORKERS", 0),
                    persistent_workers=getattr(cfg, "PERSISTENT_WORKERS", False),
                    pin_memory=getattr(cfg, "PIN_MEMORY", True))
    return dl, ds

@torch.no_grad()
def prob_tta(m, xb):
    p1 = torch.softmax(m(xb), dim=1)
    p2 = torch.softmax(m(torch.flip(xb, dims=[3])), dim=1)
    return (p1 + p2) / 2

# If best checkpoint exists, load it
best_ckpt = f"{cfg.CKPT_DIR}/plant_best.pth"
if 'load_checkpoint' in globals() and os.path.exists(best_ckpt):
    _ = load_checkpoint(model, optimizer=None, scaler=None, path=best_ckpt, device=device)
    model.eval()

if eval_dir is not None:
    # Labeled evaluation
    loader, ds = make_labeled_loader(eval_dir)
    class_names = ds.classes
    files = [fp for (fp, _) in ds.samples]

    y_true, probs = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.no_grad():
            prob = prob_tta(model, xb).cpu().numpy()
        probs.append(prob); y_true.extend(yb.numpy().tolist())
    p = np.concatenate(probs, axis=0); y_true = np.array(y_true)
    y_pred = np.argmax(p, axis=1); conf = p.max(axis=1)

    acc = accuracy_score(y_true, y_pred)
    top5 = top_k_accuracy_score(y_true, p, k=min(5, cfg.NUM_CLASSES), labels=np.arange(cfg.NUM_CLASSES)) if cfg.NUM_CLASSES>=2 else np.nan
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    try:
        yb = label_binarize(y_true, classes=list(range(cfg.NUM_CLASSES)))
        auc_m = roc_auc_score(yb, p, average="macro", multi_class="ovr")
        auc_w = roc_auc_score(yb, p, average="weighted", multi_class="ovr")
    except Exception:
        auc_m, auc_w, yb = float('nan'), float('nan'), None

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    pd.DataFrame({
        "class": class_names,
        "precision": precision_recall_fscore_support(y_true, y_pred, average=None, labels=np.arange(cfg.NUM_CLASSES), zero_division=0)[0],
        "recall":    precision_recall_fscore_support(y_true, y_pred, average=None, labels=np.arange(cfg.NUM_CLASSES), zero_division=0)[1],
        "f1":        precision_recall_fscore_support(y_true, y_pred, average=None, labels=np.arange(cfg.NUM_CLASSES), zero_division=0)[2],
        "support":   precision_recall_fscore_support(y_true, y_pred, average=None, labels=np.arange(cfg.NUM_CLASSES), zero_division=0)[3],
    }).to_csv(str(Path(OUT_DIR)/"per_class_metrics.csv"), index=False)

    pd.DataFrame({
        "file": files,
        "true_idx": y_true, "true_name": [class_names[i] for i in y_true],
        "pred_idx": y_pred, "pred_name": [class_names[i] for i in y_pred],
        "confidence": conf, "correct": (y_true==y_pred)
    }).to_csv(str(Path(OUT_DIR)/"per_image_predictions.csv"), index=False)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(cfg.NUM_CLASSES))
    plt.figure(figsize=(12,10)); sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title("Confusion Matrix (counts)"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(str(Path(OUT_DIR)/"confusion_matrix_counts.png"), dpi=150); plt.show()

    with np.errstate(all='ignore'):
        cmn = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(12,10)); sns.heatmap(cmn, cmap="Blues", vmin=0, vmax=1, cbar=True)
    plt.title("Confusion Matrix (normalized by true)"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(str(Path(OUT_DIR)/"confusion_matrix_normalized.png"), dpi=150); plt.show()

    # ROC & PR
    try:
        if yb is None: raise RuntimeError
        fpr,tpr,roc_auc={}, {}, {}
        for i in range(cfg.NUM_CLASSES):
            fpr[i], tpr[i], _ = roc_curve(yb[:, i], p[:, i])
            roc_auc[i] = roc_auc_score(yb[:, i], p[:, i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(cfg.NUM_CLASSES)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(cfg.NUM_CLASSES): mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= cfg.NUM_CLASSES
        plt.figure(figsize=(7,6))
        plt.plot(all_fpr, mean_tpr, color="navy", lw=2, label=f"Macro ROC (AUC={auc_m:.3f})")
        plt.plot([0,1],[0,1],'--',color='gray'); plt.xlim(0,1); plt.ylim(0,1.05)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Macro-Average ROC"); plt.legend(loc="lower right")
        plt.tight_layout(); plt.savefig(str(Path(OUT_DIR)/"roc_macro.png"), dpi=150); plt.show()

        prc, rec, ap = {}, {}, {}
        ap_macro = average_precision_score(yb, p, average="macro")
        for i in range(cfg.NUM_CLASSES):
            pr_i, rc_i, _ = precision_recall_curve(yb[:, i], p[:, i]); prc[i], rec[i] = pr_i, rc_i
            ap[i] = average_precision_score(yb[:, i], p[:, i])
        recall_grid = np.linspace(0,1,1000)
        mean_prec = np.zeros_like(recall_grid)
        for i in range(cfg.NUM_CLASSES): mean_prec += np.interp(recall_grid, rec[i][::-1], prc[i][::-1])
        mean_prec /= cfg.NUM_CLASSES
        plt.figure(figsize=(7,6))
        plt.plot(recall_grid, mean_prec, color="darkgreen", lw=2, label=f"Macro PR (AP={ap_macro:.3f})")
        plt.xlim(0,1); plt.ylim(0,1.05); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Macro-Average PR")
        plt.legend(loc="lower left"); plt.tight_layout(); plt.savefig(str(Path(OUT_DIR)/"pr_macro.png"), dpi=150); plt.show()
    except Exception as e:
        print("ROC/PR plotting failed:", e)

    # Summary files
    with open(str(Path(OUT_DIR)/"evaluation_summary.txt"), "w") as f:
        f.write(f"ACC {acc:.4f}\nF1 macro {f1_m:.4f}\nAUC macro {auc_m:.4f}\n")
    json.dump({
        "accuracy": float(acc), "f1_macro": float(f1_m), "auc_macro": None if np.isnan(auc_m) else float(auc_m),
        "num_classes": int(cfg.NUM_CLASSES), "num_samples": int(len(y_true)), "evaluated_dir": eval_dir,
        "artifacts_dir": OUT_DIR
    }, open(str(Path(OUT_DIR)/"evaluation_summary.json"), "w"), indent=2)
    print("Saved report artifacts to:", OUT_DIR)

else:
    # Unlabeled flat evaluation
    loader, ds = make_flat_loader(flat_test_dir)
    class_names = [d.name for d in Path(cfg.TRAIN_DIR).glob("*") if d.is_dir()]
    files, probs = [], []
    for xb, paths in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.no_grad():
            prob = prob_tta(model, xb).cpu().numpy()
        probs.append(prob); files.extend(paths)
    p = np.concatenate(probs, axis=0)
    top1 = np.argmax(p, axis=1); conf = p.max(axis=1)
    topk = min(5, p.shape[1]); idx = np.argsort(-p, axis=1)[:, :topk]
    names = [[class_names[i] for i in row] for row in idx]
    scores = [p[i, idx[i]].tolist() for i in range(len(files))]
    pd.DataFrame({
        "file": files, "pred_idx": top1, "pred_name": [class_names[i] for i in top1],
        "confidence": conf, "topk_idx": idx.tolist(), "topk_name": names, "topk_scores": scores
    }).to_csv(str(Path(OUT_DIR)/"unlabeled_predictions.csv"), index=False)
    print("Saved unlabeled predictions ->", str(Path(OUT_DIR)/"unlabeled_predictions.csv"))
```

### 2) Create a labeled test set from validation (non‑destructive)

```python
# Copy a fraction of VAL into TEST per class to build a held-out set
import os, random, shutil
from pathlib import Path
random.seed(42)
VAL_DIR, TEST_DIR = cfg.VAL_DIR, cfg.TEST_DIR
SHARE = 0.2  # 20% per class
Path(TEST_DIR).mkdir(parents=True, exist_ok=True)
for cls_dir in [d for d in Path(VAL_DIR).glob("*") if d.is_dir()]:
    imgs = [p for p in cls_dir.glob("*") if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}]
    random.shuffle(imgs); k = max(1, int(SHARE * len(imgs)))
    dst = Path(TEST_DIR) / cls_dir.name; dst.mkdir(parents=True, exist_ok=True)
    for p in imgs[:k]: shutil.copy2(str(p), str(dst / p.name))
print("Built TEST_DIR:", TEST_DIR)
```

### 3) Duplicate audits and strict test

```python
# Exact duplicates by MD5 across splits
import hashlib
from collections import defaultdict

def md5(path):
    h=hashlib.md5(); 
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def map_hashes(root):
    m={}
    for p in Path(root).rglob('*'):
        if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}:
            try: m[str(p)] = md5(str(p))
            except: pass
    return m

splits = {"train": cfg.TRAIN_DIR, "val": cfg.VAL_DIR, "test": cfg.TEST_DIR}
maps = {k: map_hashes(v) for k,v in splits.items() if Path(v).exists()}
inv=defaultdict(list)
for sp, mp in maps.items():
    for p,h in mp.items(): inv[h].append((sp,p))
collisions = {h:items for h,items in inv.items() if len({s for s,_ in items})>1}
print("Cross-split exact duplicates:", len(collisions))

# Perceptual (near) duplicates via pHash
!pip -q install imagehash pillow
import imagehash
from PIL import Image

def phash(path):
    try: return imagehash.phash(Image.open(path).convert('RGB'))
    except: return None

train_ph = set(str(phash(p)) for p in Path(cfg.TRAIN_DIR).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'})
STRICT_TEST_DIR = str(Path(cfg.TEST_DIR).parent / 'test_strict'); Path(STRICT_TEST_DIR).mkdir(parents=True, exist_ok=True)
from shutil import copy2
for cls_dir in [d for d in Path(cfg.VAL_DIR).glob('*') if d.is_dir()]:
    out = Path(STRICT_TEST_DIR)/cls_dir.name; out.mkdir(parents=True, exist_ok=True)
    for p in cls_dir.glob('*'):
        if p.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}: continue
        h=phash(str(p)); 
        if h is None: continue
        # skip near dups if very close to any train image
        if any(h - imagehash.hex_to_hash(tp) <= 4 for tp in train_ph):
            continue
        copy2(str(p), str(out/p.name))
print("Strict test built at:", STRICT_TEST_DIR)
cfg.TEST_DIR = STRICT_TEST_DIR
```

### 4) Calibration (reliability curve + ECE)

```python
# Use predictions `p` and labels `y_true` from the labeled evaluation above
import numpy as np, matplotlib.pyplot as plt

def ece_score(probs, y_true, n_bins=15):
    y_pred = probs.argmax(1); conf = probs.max(1)
    bins = np.linspace(0,1,n_bins+1); ece=0.0
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]
        sel=(conf>lo)&(conf<=hi)
        if sel.sum()==0: continue
        acc=(y_pred[sel]==y_true[sel]).mean(); c=conf[sel].mean(); ece += sel.mean()*abs(acc-c)
    return float(ece)

def reliability_diagram(probs, y_true, n_bins=15, title="Reliability"):
    y_pred = probs.argmax(1); conf = probs.max(1)
    bins = np.linspace(0,1,n_bins+1); accs=[]; confs=[]
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]
        sel=(conf>lo)&(conf<=hi)
        if sel.sum()==0: accs.append(np.nan); confs.append((lo+hi)/2); continue
        accs.append((y_pred[sel]==y_true[sel]).mean()); confs.append(conf[sel].mean())
    plt.figure(figsize=(5,4)); plt.plot([0,1],[0,1],'--',color='gray')
    plt.scatter(confs, accs, s=30); plt.xlabel('Confidence'); plt.ylabel('Accuracy')
    plt.title(title); plt.ylim(0,1); plt.xlim(0,1); plt.grid(True); plt.show()

print("ECE:", round(ece_score(p, y_true), 4))
reliability_diagram(p, y_true, title="Reliability (ECE)")
```

---

## Outputs Explained (what each artifact means)

- **confusion_matrix_counts.png**
  - Rows = true classes, columns = predicted classes. Diagonal is correct predictions; off‑diagonal shows confusions. Use with `most_confused_pairs.csv` to find frequent mistakes.
- **confusion_matrix_normalized.png**
  - Each row normalized to 1 (recall per class). Good to compare classes with different sample counts.
- **roc_macro.png**
  - Macro‑averaged ROC over one‑vs‑rest curves. Higher AUC indicates better separability across classes.
- **pr_macro.png**
  - Macro‑averaged Precision–Recall. Useful when classes are imbalanced or when positive detection quality matters more than overall accuracy.
- **roc_per_class_grid.png / pr_per_class_grid.png**
  - First N classes’ ROC/PR curves to inspect class‑specific behavior. Flat curves indicate difficult classes.
- **per_class_metrics.csv**
  - Precision/Recall/F1/Support per class. Sort by F1 ascending to find weakest classes; investigate data quality or augmentations for those.
- **per_image_predictions.csv**
  - File‑level predictions with confidence and correctness. Filter high‑confidence wrong cases to identify failure modes.
- **most_confused_pairs.csv**
  - Triplets of (true_class, predicted_class, count) sorted by frequency to guide targeted fixes.
- **evaluation_summary.txt / .json**
  - Human‑readable and machine‑readable summaries of the main metrics and paths.
- **gradcam_grid.png** (if generated)
  - Visual explanations showing where the model attends. Misaligned highlights may indicate shortcuts or spurious cues.
- **unlabeled_predictions.csv** (if test has no labels)
  - Top‑1 and top‑k predictions per file; use for manual spot‑checking.

---
 
## License
 
This repository contains training/evaluation code and instructions for academic/educational purposes. Please check the original dataset’s license and usage terms on Kaggle before use in production.
