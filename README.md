# GT_WHAR_BiGRU
# 🏃‍♂️ GT-WHAR: Graph-Based Temporal Framework for Wearable HAR

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-blue.svg)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the rigorous PyTorch reproduction of the baseline **BiGRU + GIN** architecture from the research paper: 
> *GT-WHAR: A Generic Graph-Based Temporal Framework for Wearable Human Activity Recognition With Multiple Sensors (Zou et al., IEEE TETCI 2024)*.

This codebase serves as the foundational **Phase 1** baseline for an M.Tech research thesis focusing on Deep Learning for Time-Series IMU data.

---

## 🧠 Architecture Overview

Unlike standard CNNs that treat sensor data as flat images, this framework physically maps wearable IMU sensors to the human skeleton to create a **Body-Sensing Graph**.

1. **Spatial Extraction (Body-Node Attention Graph Network):** 
   Utilizes Graph Isomorphism Networks (GIN) combined with a Node Attention branch (Tanh) to learn the structural kinematics between 5 body nodes (Torso, Left Arm, Right Arm, Left Leg, Right Leg).
2. **Temporal Extraction (Bidirectional Graph-GRU):** 
   The spatial graph embeddings are passed through a custom Bidirectional Gated Recurrent Unit (BiGRU) cell. The spatial graphs are interwoven directly into the reset, update, and candidate gates of the GRU to capture long-term sequential dependencies.
3. **Input Shape:** `[Batch, 125 timesteps, 5 nodes, 9 features]` 
   *(5 seconds of movement sampled at 25Hz).*

---

## 📂 Project Structure

```text
MTech_GT_WHAR/
├── download_data.py          # Fetches raw ZIP from UCI ML Repository
├── process_real_dsads.py     # Parses text files into PyTorch Tensors
├── main.py                   # Master training loop with LOSO Validation
├── requirements.txt          # Python dependencies
├── results.txt               # 🏆 Final experimental results & metrics
├── checkpoints/              # Stateful save-states for crash recovery
├── har_data/                 # Raw extracted UCI data
├── dataset/
│   ├── processed/            # Holds final .pt tensors (X, Y, P)
│   └── dsads_loader.py       # PyTorch Dataset and LOSO Dataloader
├── models/
│   └── gt_whar.py            # The GIN + BiGRU Neural Architecture
└── utils/
    └── metrics.py            # Accuracy & Macro-F1 calculators

```text    
## 🛠️ Installation & Setup

It is highly recommended to use a virtual environment. For optimal performance,
training should be performed on a CUDA-enabled GPU.

### 1️⃣ Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
# OR
venv\Scripts\activate          # Windows CMD / PowerShell
```

### 2️⃣ Install PyTorch with CUDA Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3️⃣ Install Remaining Dependencies

```bash
pip install torch-geometric numpy scikit-learn tqdm
```

---

## 🚀 Execution Pipeline

The entire workflow can be reproduced in three simple steps.

### Step 1: Download the Dataset

Automatically downloads the **Daily and Sports Activities (DSADS)** dataset from
the UCI Machine Learning Repository.

```bash
python download_data.py
```

### Step 2: Process the Raw Files

Parses all 9,120 raw text files, groups samples by Subject ID for LOSO
cross-validation, reshapes the data into a spatial-temporal graph format, and
stores the processed tensors as `.pt` files.

```bash
python process_real_dsads.py
```

### Step 3: Train the Model

Starts the **8-Fold Leave-One-Subject-Out (LOSO)** training and evaluation
pipeline.

```bash
python main.py
```

---

## ⚡ Key Features

### 🔹 Strict LOSO Validation Protocol

To eliminate **data leakage** and evaluate true cross-subject generalization,
the framework employs **Leave-One-Subject-Out (LOSO)** cross-validation.

- Training is performed on **7 subjects**.
- Evaluation is conducted on **1 completely unseen subject**.
- The process is repeated for all 8 subjects and the results are averaged.

### 🔹 Automatic Checkpointing & Crash Recovery

Training graph-based recurrent models can be computationally expensive.
Therefore, the framework provides **stateful checkpointing**.

- After every epoch, the model weights, optimizer state, current fold, and epoch
  are saved to:

```text
checkpoints/latest_checkpoint.pth
```

- If training is interrupted due to a crash or power failure, simply run:

```bash
python main.py
```

The script will automatically detect the latest checkpoint and resume training
from the exact point where it stopped.

---

## 📊 Results

The framework reports performance using:

- **Classification Accuracy**
- **Macro F1-Score**

Detailed fold-wise metrics and the final averaged cross-subject performance are
saved in:

```text
results.txt
```

The reproduced **GIN + BiGRU** baseline achieves performance closely matching
the original paper:

| Metric | Performance |
|--------|-------------|
| Accuracy | ~87% |
| Macro F1 | ~86% |

These results demonstrate strong cross-subject generalization on the DSADS
benchmark and provide a reliable baseline for future research extensions.