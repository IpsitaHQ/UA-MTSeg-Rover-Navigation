# Uncertainty-Aware Multimodal Terrain Segmentation and Path Planning for Autonomous Planetary Rovers

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the research codebase, experimental pipelines, and final synthesis for:
**"Uncertainty-Aware Terrain Semantic Segmentation and Traversability Estimation for Autonomous Planetary Rover Navigation"**.

The project investigates whether predictive uncertainty (derived via normalized Shannon entropy) from early-fusion multimodal segmentation can reliably detect terrain classification errors and regularize offline 2D image-space path planning to avoid ambiguous hazard boundaries with minimal geometric detour overhead.

---

## Key Experimental Findings

All reported results were evaluated on the **190 validation images** ($12,451,840$ total pixels) from the BASEPROD 80/20 train/validation split:

1. **Multimodal Early Fusion Advantage:**
   - **RGB Baseline U-Net (3-channel, 5 epochs):** Validation $\text{mIoU} = \mathbf{0.2862}$, Pixel Accuracy = $\mathbf{0.6687}$.
   - **Multimodal U-Net (5-channel: RGB + Depth + Thermal, 25 epochs, best epoch 24):** Validation $\text{mIoU} = \mathbf{0.6462}$, Pixel Accuracy = $\mathbf{0.8609}$.
   - **Performance Delta:** $\mathbf{+0.3600\text{ mIoU}}$ ($+36.00\text{ percentage points}$, $+125.8\%$ relative gain) and $\mathbf{+0.1922\text{ pixel accuracy}}$ ($+19.22\text{ percentage points}$, $+28.8\%$ relative gain).
2. **Predictive Entropy & Error Detection:**
   - Normalized predictive Shannon entropy strongly separates correct and erroneous classifications ($\mathbf{2.54\times}$ higher mean entropy in incorrect pixels: $0.5022$ vs. $0.1977$).
   - **AUROC for Segmentation Error Detection:** $\mathbf{0.8675}$.
   - While low-entropy regions ($H_{\text{norm}} \le 0.2$, comprising $55.1\%$ of pixels) have an empirical error rate of only $1.83\%$, high-entropy regions ($H_{\text{norm}} \ge 0.8$) exhibit a $63.97\%$ error rate.
3. **Risk-Aware Offline Path Planning (Primary Configuration: $0.7 C_s + 0.3 C_u$):**
   - Evaluated using 8-connected grid $A^*$ search on fixed diagonal coordinates $(5, 5) \to (250, 250)$.
   - **Offline Path-Planning Success Rate:** $\mathbf{47.37\%}$ ($90/190$ frames) for both baseline and uncertainty-aware methods (obstacle capping at $C \le 239$ preserved reachable topology with zero false obstacles).
   - **Mean Route Uncertainty Reduction:** $\mathbf{-39.68\%}$ ($0.2723 \to 0.1643$), with $100\%$ of reachable routes rerouted away from high-entropy decision boundaries.
   - **Mean Path-Length Detour Overhead:** Just $\mathbf{+1.11\%}$ ($+4.25\text{ px}$, $384.81\text{ px} \to 389.06\text{ px}$).
4. **Parameter Sensitivity Ablation ($w_u \in [0.0, 0.5]$):**
   - Confirmed a monotonic risk-distance trade-off: $w_u=0.1$ achieved a $-29.19\%$ uncertainty reduction at $+0.44\%$ length overhead; $w_u=0.5$ reached $-47.02\%$ uncertainty reduction at $+2.91\%$ overhead. Configuration 2 ($w_u=0.3$) provides an effective, balanced operational trade-off.

---

## Research Limitations & Scientific Scope

To maintain rigorous scientific integrity, the following boundaries of this study are explicitly declared:
- **Validation-Set Scope:** Evaluations are conducted on 190 validation frames from the BASEPROD dataset. No independent held-out Martian test set is available or claimed.
- **Offline 2D Image-Space Proxy:** Path planning operates on 2D camera-frame pixel grids ($256 \times 256$), not in a 3D digital elevation model (DEM) or vehicle body coordinate frame. The $47.37\%$ success rate reflects offline path reachability under fixed diagonal endpoints where 100 scenes have boundary obstacles, and must **not** be termed "navigation accuracy".
- **Engineered Traversability Proxies:** Terrain cost weights and uncertainty penalties are heuristic proxies, not supervised physical ground-truth traversability.
- **No Physical Rover Experiments:** No physical rover platform, analog testbed, or field trials were conducted. Vehicle kinodynamics (turn radius, ground clearance, tip-over risk, wheel slip) are not modeled.
- **Single-Pass Predictive Uncertainty:** Uncertainty is measured via normalized Shannon entropy over softmax probabilities, capturing combined aleatoric and model uncertainty rather than decomposed epistemic uncertainty.

---

## Repository Structure & Notebook Overview

```text
├── figures/
│   └── final_synthesis/       # Tracked publication-quality figures (Figs 1–6)
├── notebooks/
│   ├── 01_Dataset_Explorer.ipynb                         # Dataset pairing, modal stats & class labels
│   ├── 02_Preprocessing.ipynb                            # Image resizing, depth & thermal normalization
│   ├── 03_Feature_Extraction.ipynb                       # Spatial & spectral feature analysis
│   ├── 04_Data_Visulization.ipynb                        # Modality overlay & distribution inspection
│   ├── 05_Semantic_Segmentation.ipynb                    # Reference RGB-only U-Net baseline (5 epochs)
│   ├── 06_Traversability_and_Path_Planning.ipynb         # Heuristic cost mapping & baseline A* search
│   ├── 07_Navigation_Pipeline_Evaluation.ipynb           # Reference planning reachability analysis
│   ├── 08_Multimodal_RGB_Depth_Thermal_Segmentation.ipynb # Multimodal early-fusion U-Net baseline
│   ├── 09_Uncertainty_Estimation.ipynb                   # Stage A: Predictive Shannon entropy evaluation
│   ├── 10_Uncertainty_Aware_Path_Planning.ipynb          # Stage B: Paired 0.7/0.3 path planning evaluation
│   ├── 11_Uncertainty_Weight_Sensitivity.ipynb           # Stage C: Parameter sensitivity ablation
│   └── 12_Final_Results_and_Analysis.ipynb               # Master synthesis, publication tables & figures
├── utils/
│   ├── dataset.py             # PyTorch Dataset and DataLoader routines
│   ├── metrics.py             # Confusion matrix, mIoU, entropy, and AUROC calculators
│   ├── models.py              # Scratch-built MultimodalUNet PyTorch architecture
│   ├── planning.py            # 8-connected grid A* search and cost-map generators
│   └── training.py            # Early-fusion batch training and validation utilities
├── generate_final_results_synthesis.py # Reproducibility script for figures and Notebook 12
├── run_uncertainty_analysis.py         # Headless CLI runner for Stage A
├── run_stage_b_path_planning.py        # Headless CLI runner for Stage B
├── run_stage_c_weight_sensitivity.py   # Headless CLI runner for Stage C
└── requirements.txt                    # Core Python dependencies
```

---

## Reproducibility and Inspection Guide

### 1. Inspecting Results Immediately (No Dataset or Weights Required)
The final synthesis notebook and publication figures are self-contained:
- **[`notebooks/12_Final_Results_and_Analysis.ipynb`](notebooks/12_Final_Results_and_Analysis.ipynb)** loads all master tables, cross-verification breakdowns, and rendered publication figures from `figures/final_synthesis/` immediately upon cloning without requiring external datasets or GPU checkpoints.
- Exploratory notebooks **`01`–`04`** can also be inspected directly for data audit documentation.

### 2. Rerunning Experimental Forward Passes
To rerun model training or inference passes from scratch:
- **Requirements:**
  1. Local copy of the **BASEPROD labeled dataset** (RGB, Depth, Thermal, and Mask rasters).
  2. Model weights checkpoint (`outputs/multimodal_segmentation/best_multimodal_unet.pt`).
- **Notice on Distribution:** In compliance with repository size constraints, the raw dataset and large model checkpoints are **not distributed** in this Git repository.

---

## Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/IpsitaHQ/UA-MTSeg-Rover-Navigation.git
   cd UA-MTSeg-Rover-Navigation
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
   Open `notebooks/12_Final_Results_and_Analysis.ipynb` to view the consolidated findings and publication figures.

---

## Citation & Metadata
- **Project:** Autonomous Planetary Rover Navigation Research
- **Dataset:** BASEPROD Planetary Analog Surface Dataset
- **Framework:** PyTorch, Torchvision, NumPy, Pandas, Matplotlib