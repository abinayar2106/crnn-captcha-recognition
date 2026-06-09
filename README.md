# Distorted Visual Sequence Recognition — CRNN + CTC

A sequence recognition model built to decode 6-character alphanumeric strings from heavily distorted grayscale images, using a CRNN architecture with CTC loss.

## Problem

Given 20,000 training images of distorted sequences, predict the correct character string despite:
- Heavy black blob occlusion over characters
- Background noise and visual artifacts
- Irregular spacing, blur, and shape deformation

**Evaluation metric:** Character Error Rate (CER) — lower is better  
**Vocabulary:** 38 characters (alphanumeric)

## Architecture

```
Input (1×32×128) → CNN Backbone → BiLSTM → CTC Decoder → Predicted Sequence
```

- **CNN (5 conv blocks):** Extracts spatial features column by column, compressing the image to 32 feature vectors (one per time step)
- **BiLSTM (2 layers, 256 hidden units):** Reads feature columns bidirectionally — left-to-right and right-to-left — to resolve ambiguous characters using surrounding context
- **CTC Loss:** Handles variable alignment between input frames and output characters without needing character-level annotation

## Results

| Model | CER |
|-------|-----|
| Baseline CRNN | 0.0210 |
| + Attention mechanism | 0.0138 |
| Improvement | ~34% relative reduction |

Training ran for 30 epochs on a T4 GPU (Google Colab). Notable behaviour: loss plateaued for the first 18 epochs before a sudden breakthrough at epoch 19, a known characteristic of CTC optimization.

## Key Design Decisions

- **Distortion-aware augmentation:** Training data augmented with synthetic blob occlusion and noise to match test distribution
- **Prefix beam search decoding:** Replaces greedy CTC decoding for better sequence accuracy
- **Mixed precision training:** `torch.amp.autocast` for faster T4 training without accuracy loss
- **Attention layer:** Added over BiLSTM outputs to weight the most informative time steps

## Visualizations

| Plot | Description |
|------|-------------|
| `training_curves.png` | Loss and CER over 30 epochs — shows the plateau-breakthrough pattern |
| `model_comparison.png` | Baseline vs attention model performance |
| `attention_visualization.png` | Which time steps the model attends to per prediction |
| `confusion_visualization.png` | Worst confused character pairs (e.g. C→G) under occlusion |
| `preprocessing_comparison.png` | Raw vs augmented image samples |

## Repo Structure

```
crnn-captcha-recognition/
├── crnn_baseline.ipynb          # Full training pipeline, analysis, and visualizations
├── submission_AbinayaR_*.csv    # Final predictions on test set
├── visualization/               # Saved plots from training and evaluation
└── README.md
```

## Requirements

```
torch
torchvision
numpy
pandas
matplotlib
Pillow
```

Run on Google Colab with GPU runtime (T4 recommended).
