# Faculty Demonstration Guide: Uncertainty-Aware Rover Navigation Project

**Project Title:** Uncertainty-Aware Terrain Semantic Segmentation and Traversability Estimation for Autonomous Planetary Rover Navigation  
**Repository:** [UA-MTSeg-Rover-Navigation](https://github.com/IpsitaHQ/UA-MTSeg-Rover-Navigation)  
**Target Environment:** VS Code with Python 3.12 (CUDA-enabled)

---

## 1. Quick Start: Opening the Project in VS Code

### Step 1: Open the Repository Root
Launch VS Code directly in the repository directory (do not open a subdirectory like `notebooks/`):
```bash
code c:\Users\Anushka\OneDrive\Desktop\rover
```

### Step 2: Configure Python Interpreter & Jupyter Kernel
1. Open the Command Palette (`Ctrl + Shift + P`).
2. Type and select: **`Python: Select Interpreter`**.
3. Select the local Python installation:
   - **`Python 3.12.7 ('base') C:\Program Files\Python312\python.exe`** (NVIDIA RTX 2050 GPU detected).
4. When opening any `.ipynb` notebook, check the top-right corner of the editor:
   - Ensure the kernel is set to **`Python 3.12.7`**. If prompted, click **`Select Kernel`** -> **`Python Environments...`** -> choose Python 3.12.7.

---

## 2. Recommended Demonstration Flow (15–20 Minutes)

| Stage | Focus Area | Notebook / Script | Key Faculty Takeaway |
| :---: | :--- | :--- | :--- |
| **1** | **Executive Synthesis** | `notebooks/12_Final_Results_and_Analysis.ipynb` | Overview of all findings, master tables, and publication trade-off figures. |
| **2** | **Multimodal Segmentation** | `notebooks/08` vs. `notebooks/05` + `utils/models.py` | 5-channel early fusion gains (+36.00 pp mIoU) over RGB-only baseline. |
| **3** | **Uncertainty & Error Detection** | `notebooks/09_Uncertainty_Estimation.ipynb` | Predictive Shannon entropy detects errors with AUROC = 0.8675 (2.54× separation). |
| **4** | **Risk-Aware Path Planning** | `notebooks/10_Uncertainty_Aware_Path_Planning.ipynb` | 39.68% route uncertainty reduction at only +1.11% detour cost (+4.25 px). |
| **5** | **Weight Sensitivity Ablation** | `notebooks/11_Uncertainty_Weight_Sensitivity.ipynb` | Monotonic risk-distance trade-off confirms robustness across weights $w_u \in [0.1, 0.5]$. |
| **6** | **Dataset & Engineering Audit** | `notebooks/01` & `notebooks/04` | Rigorous data pairing, class distributions, and modality overlay verification. |

---

## 3. Detailed Notebook-by-Notebook Guide (01 through 12)

### Notebook 12: Final Results and Experimental Analysis (START HERE)
- **Path:** [`notebooks/12_Final_Results_and_Analysis.ipynb`](notebooks/12_Final_Results_and_Analysis.ipynb)
- **Execution Status:** **Fully runnable live in < 3 seconds** (zero external dependencies).
- **Purpose:** Compiles and cross-verifies all research milestones into publication-ready tables and figures.
- **Cells to Run:** Click **`Run All`**. All master tables (Tables 1–6) and high-resolution figures (Figs 1–6) render instantly from tracked paths.
- **Key Results to Highlight:**
  - Table 1: RGB baseline (mIoU 0.2862) vs. Multimodal UNet (mIoU 0.6462).
  - Table 3 & Fig 3: Error detection AUROC (0.8675) and binned error calibration.
  - Table 4 & Fig 4: Paired planning metrics across 90 reachable frames (-39.7% risk, +1.1% length).
  - Table 5 & Fig 5: Monotonic weight sensitivity ablation ($w_u \in [0.0, 0.5]$).

---

### Notebook 08: Multimodal RGB + Depth + Thermal Segmentation
- **Path:** [`notebooks/08_Multimodal_RGB_Depth_Thermal_Segmentation.ipynb`](notebooks/08_Multimodal_RGB_Depth_Thermal_Segmentation.ipynb)
- **Execution Status:** **Fully runnable live in < 3 seconds** (`RUN_TRAINING = False` by default).
- **Purpose:** Demonstrates the 5-channel early-fusion U-Net architecture, displays the actual 25-epoch training history table, and renders progression curves.
- **Live Demo Strategy:**
  - **Click Run All or run cells sequentially:** Cell 9 automatically detects `RUN_TRAINING = False`, bypasses GPU training, and binds the verified **Epoch 24 checkpoint** (`outputs/multimodal_segmentation/best_multimodal_unet.pt`).
  - **Show 25-Epoch Training History Table:** Cell 11 displays all 25 recorded epochs loaded directly from `outputs/multimodal_segmentation/training_metrics_25ep.csv`.
  - **Show 3-Panel Progression Curves:** Displays validation mIoU progression (with Epoch 24 marked as the peak), Loss curves, and Pixel Accuracy curves.
  - **Show Verified Evaluation:** Displays the +36.00 pp mIoU gain over the RGB baseline and per-class IoU across all 8 classes without overwriting any files.

---

### Notebook 05: Semantic Segmentation (RGB Baseline)
- **Path:** [`notebooks/05_Semantic_Segmentation.ipynb`](notebooks/05_Semantic_Segmentation.ipynb)
- **Execution Status:** Reference baseline from canonical repository.
- **Purpose:** Documents the 3-channel RGB-only U-Net baseline (mIoU = 0.2862, Acc = 0.6687).
- **Live Demo Strategy:**
  - Open and scroll through pre-rendered training curves and confusion matrix outputs.
  - **DO NOT RUN TRAINING:** Preserves the canonical 5-epoch baseline without overwriting logs.

---

### Notebook 09: Uncertainty Estimation & Error Detection
- **Path:** [`notebooks/09_Uncertainty_Estimation.ipynb`](notebooks/09_Uncertainty_Estimation.ipynb)
- **Execution Status:** **Fully runnable live in < 2 seconds**.
- **Purpose:** Evaluates normalized predictive Shannon entropy $H_{\text{norm}}(p)$ across 12.45M validation pixels.
- **Cells to Run:** Run Cell 1 and Cell 2.
- **Key Results to Highlight:**
  - Erroneous pixels have 2.54× higher mean entropy ($0.5022$ vs. $0.1977$).
  - High-entropy bins ($H \ge 0.8$) exhibit a $63.97\%$ error rate compared to only $1.83\%$ in low-entropy regions ($H \le 0.2$).

---

### Notebook 10: Uncertainty-Aware Path Planning (Stage B)
- **Path:** [`notebooks/10_Uncertainty_Aware_Path_Planning.ipynb`](notebooks/10_Uncertainty_Aware_Path_Planning.ipynb)
- **Execution Status:** **Fully runnable live in < 2 seconds**.
- **Purpose:** Compares Semantic Baseline ($C_A = C_s$) against Uncertainty-Aware ($C_B = 0.7 C_s + 0.3 C_u$) using 8-connected grid $A^*$.
- **Cells to Run:** Run Cell 1 (displays aggregate paired statistics) and Cell 2 (displays representative visual trajectories).
- **Key Results to Highlight:**
  - 100% of reachable routes rerouted away from ambiguous hazard boundaries.
  - Average detour overhead is only +4.25 px (+1.11%).
  - Obstacle cost capping ($C \le 239$) strictly preserved reachability (zero false obstacles).

---

### Notebook 11: Uncertainty-Weight Sensitivity (Stage C)
- **Path:** [`notebooks/11_Uncertainty_Weight_Sensitivity.ipynb`](notebooks/11_Uncertainty_Weight_Sensitivity.ipynb)
- **Execution Status:** **Fully runnable live in < 2 seconds**.
- **Purpose:** Parameter ablation across weights $w_u \in [0.0, 0.1, 0.3, 0.5]$ across all 190 validation frames.
- **Cells to Run:** Run Cell 1 (summary table) and Cell 2 (3-panel trade-off visualization).
- **Key Results to Highlight:**
  - Monotonic risk reduction: $w_u=0.1$ (-29.2%), $w_u=0.3$ (-39.7%), $w_u=0.5$ (-47.0%).
  - Demonstrates that the Stage B finding is robust and not a fine-tuned artifact.

---

### Notebooks 01–04, 06–07: Foundations & Exploratory Analysis
- **Notebook 01:** Dataset pairing and 8-class inventory on 950 BASEPROD images.
- **Notebook 02:** Resizing ($256 \times 256$), depth scaling (/ 255.0), and thermal scaling.
- **Notebook 03 & 04:** Spatial/spectral feature analysis and modality overlay plots.
- **Notebook 06 & 07:** Reference single-channel cost mapping and navigation evaluation.
- **Live Demo Strategy:** Open these notebooks to show thorough foundational engineering if faculty ask about data integrity.

---

## 4. Reusable Modular Utilities (`utils/`)

When demonstrating the codebase architecture, highlight the clean modular design in `utils/`:
- **[`utils/models.py`](utils/models.py):** Implements `MultimodalUNet`, a 4-stage early-fusion U-Net taking 5 input channels (RGB + Depth + Thermal) with skip connections and 7.77M parameters.
- **[`utils/metrics.py`](utils/metrics.py):** Contains vectorized confusion matrix calculation, class-wise IoU, normalized Shannon entropy, and scikit-learn AUROC calculation.
- **[`utils/planning.py`](utils/planning.py):** Implements an 8-connected grid $A^*$ algorithm with diagonal corner-cut prevention, heuristic cost blending, and obstacle thresholding.
- **[`utils/dataset.py`](utils/dataset.py):** Implements PyTorch Dataset loading with metric depth/thermal scaling and synchronized mask transforms.

---

## 5. Saved Results vs. Freshly Computed Results

To be completely transparent during the demonstration, distinguish between precomputed artifacts and live executions:

| Result Artifact | Status | Source & Location |
| :--- | :--- | :--- |
| **Model Checkpoint** | Saved Artifact | `outputs/multimodal_segmentation/best_multimodal_unet.pt` (31.1 MB, Epoch 24). |
| **190-Frame Inference Metrics** | Saved Artifact | `outputs/multimodal_segmentation/final_evaluation.json` and `per_class_iou.csv`. |
| **12.45M Pixel Entropy Metrics** | Saved Artifact | `outputs/uncertainty/uncertainty_metrics.json`. |
| **760 Total A* Planning Paths** | Saved Artifact | `outputs/risk_aware_navigation/baseline_vs_uncertainty_metrics.csv` & `stage_c_weight_sensitivity/weight_sensitivity_metrics.csv`. |
| **Master Summary Tables** | Computed Live | Executed in Notebooks 09–12 by aggregating and formatting source records into DataFrames. |
| **Visual Charts & Plots** | Rendered / Tracked | Figures in `figures/final_synthesis/` rendered directly from verified experimental data. |

---

## 6. Scientific Limitations & Likely Faculty Questions

### Q1: "Why is your path-planning success rate only 47.37%? Does the rover fail half the time?"
- **Answer:**  
  *"The 47.37% figure is strictly an **offline image-space benchmark** using fixed diagonal corner-to-corner coordinates $(5, 5) \to (250, 250)$. In 100 out of the 190 validation frames, the start or goal pixel lands directly inside boundary impassable terrain (Rock or Void). Crucially, in all 90 reachable frames, both the baseline and uncertainty-aware planners found valid paths. Furthermore, because non-obstacle costs are strictly capped at 239, adding uncertainty never falsely created impassable barriers (zero false obstacles)."*

### Q2: "Your RGB baseline was trained for 5 epochs, but the multimodal model trained for 25 epochs. Is the comparison fair?"
- **Answer:**  
  *"We explicitly disclose this training disparity in Notebook 12 and the README. The 5-epoch model was the canonical reference baseline provided in Notebook 05. The multimodal early-fusion model was trained for 25 epochs (reaching its peak validation mIoU of 0.6462 at epoch 24). The performance gain (+36.00 pp mIoU) reflects both the integration of thermal/depth sensory modalities and extended optimization. In our training logs (`training_metrics_25ep.csv`), the multimodal model surpassed the RGB baseline's final mIoU within its very first training epochs."*

### Q3: "Is your traversability map based on ground-truth rover vehicle measurements?"
- **Answer:**  
  *"No. We explicitly declare that traversability costs are **engineered heuristic proxies** mapped from semantic class IDs (e.g., Compact=10, Sand=80, Rock=255) and normalized Shannon entropy. The study does not model 3D terrain elevation, chassis ground clearance, or wheel-soil slip. It evaluates the algorithmic hypothesis that predictive entropy regularizes path selection away from uncertain decision boundaries."*

### Q4: "Why use single-pass Shannon entropy instead of Monte Carlo Dropout or Deep Ensembles?"
- **Answer:**  
  *"Single-pass normalized Shannon entropy requires zero additional inference latency, making it practical for compute-constrained flight processors on planetary rovers. Despite its simplicity, our evaluation across 12.45 million validation pixels demonstrated an error-detection AUROC of 0.8675 and a 2.54× entropy separation between correct and erroneous pixels, proving highly effective for real-time risk weighting."*

### Q5: "Have you tested this on Mars or on physical rover hardware?"
- **Answer:**  
  *"No physical rover experiments or Martian test sets were conducted. All evaluations were conducted on Earth-analog planetary surface imagery from the BASEPROD validation dataset (190 frames). This work serves as an offline algorithmic proof-of-concept for uncertainty-aware planetary rover planning."*

---

## 7. Headless Reproducibility Scripts (Command-Line Demos)

If faculty request verifying the pipeline from the terminal without opening Jupyter:
- **Reproduce Final Figures & Notebook 12:**
  ```bash
  python generate_final_results_synthesis.py
  ```
- **Run Headless Stage A (Uncertainty Analysis):**
  ```bash
  python run_uncertainty_analysis.py
  ```
- **Run Headless Stage B (Paired Path Planning):**
  ```bash
  python run_stage_b_path_planning.py
  ```
- **Run Headless Stage C (Weight Sensitivity Ablation):**
  ```bash
  python run_stage_c_weight_sensitivity.py
  ```
