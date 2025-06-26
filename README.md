# Vision-transformer
A fully customized implementation of Vision Transformer (ViT) in PyTorch, including training, validation, evaluation metrics, and visualization. Built from scratch without relying on prebuilt ViT libraries.
# 🔍 Vision Transformer from Scratch (Custom ViT)

This repository contains a fully custom implementation of a Vision Transformer (ViT) model built using PyTorch. It includes detailed components like patch embedding, positional encoding, multi-head self-attention, transformer encoders, and a classification head. The model is trained and evaluated on image classification datasets.

## 🚀 Features

- Vision Transformer (ViT) implemented **from scratch**
- Patch Embedding & Positional Embedding
- Multi-head Self Attention and MLP Blocks
- Full training loop with:
  - Checkpointing
  - Learning rate scheduling
  - Validation & Test evaluation
- Auto-generation of:
  - Loss curves
  - Top-1 and Top-3 accuracy plots
  - Confusion matrices
  - Model statistics (FLOPs, params, inference time)

## 🖼️ Dataset Structure (Example)

SoyMCData/
├── train/
├── val/
└── test/



## 📦 Installation

pip install torch torchvision matplotlib seaborn scikit-learn ptflops tqdm


## ⚙️ How to Run
python vit_test.py


* Modify `save_dir` and `SoyMCData` path if needed.

## 📊 Outputs

All outputs are stored in the results folder:

* `results.json`: Summary of metrics
* `loss_curves.png`: Training and validation loss
* `top1_top3_comparison.png`: Accuracy trends
* `train/test_confusion_matrix.png`
* `trained_model.pth`: Final model weights

## 📈 Sample Output (Console)

Epoch 1/5:
  Train Top-1: 83.42%
  Train Top-3: 96.50%
  Val Top-1: 81.37%
  Val Top-3: 94.11%
...
Final Test Results:
  Top-1: 85.17%
  Top-3: 97.20%
