# 50.039 Deep Learning Project - DiabeticRetinopathy

Binary classification machine learning problem for detecting severe Diabetic Retinopathy (anomaly) via fundus photography. Models are trained to distinguish between **Healthy** and **Severe DR** retinal images, exploring U-Net variants as the primary architecture.

---

## Setup

### Requirements

Install the required packages:

```bash
pip install torch torchvision matplotlib kagglehub ipython
```

### GPU (Recommended)

CUDA is strongly recommended for training. CUDA requires an **NVIDIA GPU** with at least **8 GB of VRAM**.

To verify CUDA is available:

```python
import torch
print(torch.cuda.is_available())  # True if CUDA is ready
```

### Running on Google Colab (no GPU available)

If you don't have a compatible NVIDIA GPU, you can run the notebook on [Google Colab](https://colab.research.google.com/) with a free T4 GPU:

1. Upload the following files/folders to a single folder in your Google Drive: `main.ipynb`, `compare.py`, `models/`, and `utils/`.
2. Open `main.ipynb` in Colab, or go to **File → Open notebook → Google Drive**.
3. Go to **Runtime → Change runtime type → T4 GPU**.
4. Mount your Google Drive and navigate to the project folder in the first cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   import os
   os.chdir('/content/drive/MyDrive/<path-to-project-folder>')
   ```
5. Install the extra dependency:
   ```python
   !pip install kagglehub
   ```
6. Run all cells.

---

## Dataset

The dataset is downloaded automatically from Kaggle on first run. It contains retinal fundus images in two classes: **Healthy** and **Severe DR** (1190 images total).

```python
from utils import import_dataset
import_dataset()  # Downloads and prepares the dataset into ./dataset/
```

---

## Initialising and Training a Model

### 1. Prepare data loaders

```python
from utils import initialize_dataset, split_dataset, initialize_dataloaders

full_dataset = initialize_dataset()
train_dataset, test_dataset, validation_dataset = split_dataset(full_dataset)
train_dataloader, test_dataloader, validation_dataloader = initialize_dataloaders(
    train_dataset, test_dataset, validation_dataset, batch_size=32
)
```

### 2. Instantiate a model

Available architectures:

| Model | Class |
|-------|-------|
| U-Net (baseline) | `UNetClassifier` |
| Attention U-Net | `AUNetClassifier` |
| Residual U-Net | `ResUNetClassifier` |
| Attention + Residual U-Net | `AResUNetClassifier` |
| EfficientNet-B0 (pretrained) | `EfficientNetB0Classifier` |
| U-Net without skip connections | `NoSkipUNetClassifier` |

```python
import torch
import torch.nn as nn
from models import UNetClassifier  # swap for any model above

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetClassifier().to(device)
```

### 3. Train

```python
from utils import train_model

train_model(
    model=model,
    criterion=nn.CrossEntropyLoss(),
    train_dataloader=train_dataloader,
    validation_dataloader=validation_dataloader,
    device=device,
    lr=0.001,
    num_epochs=20,
    save_dir="./outputs/compare/plots",
    verbose=True,
)
```

Training loss/accuracy plots are saved as JPEG files to `save_dir`.

### 4. Save a trained model

```python
from utils import save_model

save_model(model, name="UNetClassifier", epoch=20, dir="./outputs/models")
# Saves to: ./outputs/models/UNetClassifier_20-<date>.pt
```

### Compare all architectures at once

To train and benchmark all six models automatically:

```python
from compare import compare_architecture

compare_architecture(
    train_dataloader, validation_dataloader, test_dataloader,
    device=device,
    reload=False,  # set True to retrain from scratch, discarding saved checkpoints
)
```

Trained models are saved to `./outputs/models/compare/` and a metrics table is printed.

---

## Evaluating a Model

```python
from utils import evaluate_model

accuracy, f1, f2, anomaly_detection_rate = evaluate_model(
    model=model,
    dataloader=test_dataloader,
    output_dir="./outputs",
    device=device,
)

print(f"Accuracy: {accuracy:.2f}% | F1: {f1:.2f} | F2: {f2:.2f} | Anomaly Detection: {anomaly_detection_rate:.2f}%")
```

Segmentation output images are saved as PNG files to `output_dir` when the dataloader has `shuffle=False`.

---

## Visualisation Utilities

All visualisation functions are in `utils/visualization.py`.

### Dataset samples

```python
from utils import visualize_dataset
visualize_dataset()  # Plots 5 samples from each class (Healthy / Severe DR)
```

### Training and validation curves

```python
from utils import plot_train_val
plot_train_val(
    train_loss_list, train_acc_list,
    val_loss_list, val_acc_list,
    save=True, name="UNetClassifier",
    save_dir="./outputs/compare/plots",
)
```

### Anomaly detection outputs

```python
from utils import plot_anomalies
plot_anomalies(
    validation_dataset, test_dataset,
    val_dir="./outputs/val",
    test_dir="./outputs/test",
)
# Side-by-side: input image vs. model segmentation output for anomalous cases
```

### Class distribution per batch

```python
from utils import plot_anomaly_distribution
plot_anomaly_distribution(
    train_dataloader, validation_dataloader,
    batch_size=32,
    save=True, name="distribution",
    save_dir="./outputs/compare/plots",
)
```

---

## Project Structure

```
DiabeticRetinopathyDL/
├── main.ipynb                  # End-to-end workflow notebook
├── compare.py                  # Multi-model training & benchmarking
├── models/
│   ├── baselineunet.py         # UNetClassifier
│   ├── attentionunet.py        # AUNetClassifier
│   ├── residualunet.py         # ResUNetClassifier
│   ├── aresunet.py             # AResUNetClassifier
│   ├── efficientnet_b0.py      # EfficientNetB0Classifier
│   └── noskipunet.py           # NoSkipUNetClassifier
└── utils/
    ├── loader.py               # Dataset download, loading, splitting
    ├── model.py                # train_model(), evaluate_model()
    ├── io.py                   # save_model(), save_image()
    └── visualization.py        # Plotting utilities
```
