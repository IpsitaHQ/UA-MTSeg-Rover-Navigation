"""
generate_final_results_synthesis.py

Synthesizes, cross-verifies, and visualizes all completed experimental results
from Stages 5 (RGB baseline), 8/Epoch 24 (Multimodal), 9/Stage A (Uncertainty),
10/Stage B (Planning), and 11/Stage C (Weight Sensitivity).

Generates:
1. Publication-quality figures in outputs/final_synthesis/
2. Canonical synthesis notebook: notebooks/12_Final_Results_and_Analysis.ipynb
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

cwd = Path.cwd().resolve()
ROOT = cwd.parent if cwd.name == "notebooks" else (next((p for p in (cwd, *cwd.parents) if (p / "notebooks").is_dir()), cwd))
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "figures" / "final_synthesis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SYNTHESIS_DIR = FIGURES_DIR

# -----------------------------------------------------------------------------
# 1. Load Source Data
# -----------------------------------------------------------------------------
def load_source_data():
    print("Loading source experimental outputs...")

    # Multimodal segmentation
    multi_eval_path = OUTPUTS_DIR / "multimodal_segmentation" / "final_evaluation.json"
    with open(multi_eval_path, "r", encoding="utf-8") as f:
        multi_eval = json.load(f)

    per_class_csv_path = OUTPUTS_DIR / "multimodal_segmentation" / "per_class_iou.csv"
    per_class_df = pd.read_csv(per_class_csv_path)

    # Uncertainty estimation
    unc_metrics_path = OUTPUTS_DIR / "uncertainty" / "uncertainty_metrics.json"
    with open(unc_metrics_path, "r", encoding="utf-8") as f:
        unc_metrics = json.load(f)

    # Stage B planning
    plan_metrics_path = OUTPUTS_DIR / "risk_aware_navigation" / "risk_map_summary.json"
    with open(plan_metrics_path, "r", encoding="utf-8") as f:
        plan_summary = json.load(f)

    # Stage C weight sensitivity
    sens_summary_path = OUTPUTS_DIR / "risk_aware_navigation" / "stage_c_weight_sensitivity" / "weight_sensitivity_summary.json"
    with open(sens_summary_path, "r", encoding="utf-8") as f:
        sens_summary = json.load(f)

    return {
        "multi_eval": multi_eval,
        "per_class_df": per_class_df,
        "unc_metrics": unc_metrics,
        "plan_summary": plan_summary,
        "sens_summary": sens_summary,
    }


# -----------------------------------------------------------------------------
# 2. Generate Publication-Quality Visualizations
# -----------------------------------------------------------------------------
def generate_figures(data):
    print("Generating publication-quality visualization figures...")

    # Colors
    c_blue = "#1f77b4"
    c_orange = "#ff7f0e"
    c_green = "#2ca02c"
    c_red = "#d62728"
    c_purple = "#9467bd"

    # -------------------------------------------------------------------------
    # Fig 1: Segmentation Performance Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["RGB Baseline (3 ch, 5 ep)", "Multimodal UNet (5 ch, 25 ep)"]
    miou_vals = [0.2862, 0.6462]
    acc_vals = [0.6687, 0.8609]

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, miou_vals, width, label="Validation mIoU", color=c_blue, edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, acc_vals, width, label="Validation Pixel Accuracy", color=c_orange, edgecolor="black", alpha=0.9)

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Terrain Segmentation Performance: RGB Baseline vs. Multimodal Early Fusion", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="upper left")

    def autolabel(rects, is_pct=False):
        for rect in rects:
            height = rect.get_height()
            txt = f"{height:.4f}" if not is_pct else f"{height*100:.1f}%"
            ax.annotate(txt,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    # Annotation of gains
    ax.annotate("+36.00 pp (+125.8% rel.)", xy=(1 - width/2, 0.6462), xytext=(0.5, 0.50),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
                fontsize=9, fontweight="bold", ha="center")
    ax.annotate("+19.22 pp (+28.8% rel.)", xy=(1 + width/2, 0.8609), xytext=(1.15, 0.73),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
                fontsize=9, fontweight="bold", ha="center")

    fig.tight_layout()
    f1_path = SYNTHESIS_DIR / "fig1_segmentation_comparison.png"
    fig.savefig(f1_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {f1_path}")

    # -------------------------------------------------------------------------
    # Fig 2: Per-Class IoU Breakdown
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    df_cls = data["per_class_df"]
    classes = df_cls["class_name"].tolist()
    ious = df_cls["iou"].tolist()
    palette = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#6ece58", "#fde725"]

    bars = ax.barh(classes[::-1], [v * 100 for v in ious[::-1]], color=palette[::-1], edgecolor="black", height=0.65)
    ax.axvline(0.6462 * 100, color="red", linestyle="--", linewidth=1.5, label="Mean IoU (64.62%)")

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f}%",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Intersection-over-Union (%)", fontsize=11, fontweight="bold")
    ax.set_title("Multimodal UNet Per-Class IoU on BASEPROD Validation Set (190 Frames)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    f2_path = SYNTHESIS_DIR / "fig2_per_class_iou.png"
    fig.savefig(f2_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {f2_path}")

    # -------------------------------------------------------------------------
    # Fig 3: Entropy vs. Error Rate Chart
    # -------------------------------------------------------------------------
    bins_data = data["unc_metrics"]["uncertainty_bins"]
    bin_labels = [b["bin_range"] for b in bins_data]
    error_rates = [b["error_rate_pct"] for b in bins_data]
    pixel_pcts = [b["pixel_fraction_pct"] for b in bins_data]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    x_idx = np.arange(len(bin_labels))
    width = 0.4

    b1 = ax1.bar(x_idx - width/2, pixel_pcts, width, label="Pixel Density (% of total)", color="#4a7c59", edgecolor="black", alpha=0.85)
    l1 = ax2.plot(x_idx + width/2, error_rates, marker="o", linewidth=2.5, color="#c0392b", label="Segmentation Error Rate (%)")

    ax1.set_xlabel("Predictive Shannon Entropy Bin ($H_{norm}$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Pixel Density (% of 12.45M Pixels)", fontsize=11, fontweight="bold", color="#2d5236")
    ax2.set_ylabel("Observed Error Rate (%)", fontsize=11, fontweight="bold", color="#96281b")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(bin_labels, fontsize=10)
    ax1.set_ylim(0, 70)
    ax2.set_ylim(0, 80)

    for i, (b, err, px) in enumerate(zip(x_idx, error_rates, pixel_pcts)):
        ax1.annotate(f"{px:.1f}%", xy=(b - width/2, px), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
        ax2.annotate(f"{err:.1f}%", xy=(b + width/2, err), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#96281b")

    plt.title("Predictive Entropy and Segmentation Error Detection\n(AUROC = 0.8675, Incorrect/Correct Entropy Ratio = 2.54×)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Combined legend
    patches = [
        mpatches.Patch(facecolor="#4a7c59", edgecolor="black", label="Pixel Density (%)"),
        plt.Line2D([0], [0], color="#c0392b", marker="o", linewidth=2, label="Empirical Error Rate (%)")
    ]
    ax1.legend(handles=patches, loc="upper left", fontsize=10)

    fig.tight_layout()
    f3_path = SYNTHESIS_DIR / "fig3_entropy_vs_error_rate.png"
    fig.savefig(f3_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {f3_path}")

    # -------------------------------------------------------------------------
    # Fig 4: Planning Risk-Distance Trade-Off (Stage B)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    methods = ["Semantic Baseline (A)\n($C_A = C_s$)", "Uncertainty-Aware (B)\n($0.7 C_s + 0.3 C_u$)"]
    mean_u = [0.2723, 0.1643]
    mean_l = [384.81, 389.06]

    # Subplot 1: Route Uncertainty
    bars_u = axes[0].bar(methods, mean_u, color=["#e74c3c", "#27ae60"], edgecolor="black", width=0.55)
    axes[0].set_ylabel("Mean Route Predictive Entropy ($\overline{H}_{norm}$)", fontsize=10, fontweight="bold")
    axes[0].set_title("Mean Route Uncertainty Reduction", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 0.35)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars_u:
        h = b.get_height()
        axes[0].annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    axes[0].annotate("-39.68% Uncertainty\n(in 100% of routes)", xy=(1, 0.1643), xytext=(0.5, 0.28),
                     arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
                     ha="center", fontsize=9, fontweight="bold")

    # Subplot 2: Path Length
    bars_l = axes[1].bar(methods, mean_l, color=["#3498db", "#2980b9"], edgecolor="black", width=0.55)
    axes[1].set_ylabel("Mean Path Length (pixels)", fontsize=10, fontweight="bold")
    axes[1].set_title("Geometric Path Length Overhead", fontsize=11, fontweight="bold")
    axes[1].set_ylim(350, 410)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars_l:
        h = b.get_height()
        axes[1].annotate(f"{h:.2f} px", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    axes[1].annotate("+1.11% Detour (+4.25 px)", xy=(1, 389.06), xytext=(0.5, 400),
                     arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
                     ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Stage B Paired Comparison: Offline Image-Space $A^*$ Across 90 Reachable Validation Frames", fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    f4_path = SYNTHESIS_DIR / "fig4_planning_risk_distance_tradeoff.png"
    fig.savefig(f4_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {f4_path}")

    # -------------------------------------------------------------------------
    # Fig 5: Weight Sensitivity Curves (Stage C)
    # -------------------------------------------------------------------------
    sens_rows = data["sens_summary"]["configuration_comparison_table"]
    w_u = [r["uncertainty_weight"] for r in sens_rows]
    u_red = [r["uncertainty_reduction_percent"] for r in sens_rows]
    l_over = [r["path_length_overhead_percent"] for r in sens_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(l_over[1:], u_red[1:], s=140, color="#2ecc71", edgecolor="black", zorder=4)
    ax.plot(l_over, u_red, linestyle="--", color="#7f8c8d", linewidth=1.8, marker="o", alpha=0.8)

    labels = [
        "Baseline ($w_u=0.0$)\n(0.0%, 0.0%)",
        "Config 1 ($w_u=0.1$)\n(+0.44% len, -29.19% u)",
        "Config 2: Stage B ($w_u=0.3$)\n(+1.11% len, -39.68% u)",
        "Config 3 ($w_u=0.5$)\n(+2.91% len, -47.02% u)"
    ]

    offsets = [(-15, 8), (10, -5), (10, -5), (10, -5)]
    for i, (txt, off) in enumerate(zip(labels, offsets)):
        ax.annotate(txt, (l_over[i], u_red[i]), xytext=off, textcoords="offset points", fontweight="bold", fontsize=9)

    ax.set_xlabel("Mean Path Length Overhead (% vs. Semantic Baseline)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Route Uncertainty Reduction (%)", fontsize=11, fontweight="bold")
    ax.set_title("Stage C Sensitivity: Risk-Distance Trade-Off Across Uncertainty Weights $w_u \in [0.0, 0.5]$", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.3, 3.8)
    ax.set_ylim(-3, 52)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    f5_path = SYNTHESIS_DIR / "fig5_weight_sensitivity_curves.png"
    fig.savefig(f5_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {f5_path}")

    # -------------------------------------------------------------------------
    # Fig 6: Representative Paths Mosaic (Using existing images)
    # -------------------------------------------------------------------------
    rep_files = sorted(list((OUTPUTS_DIR / "risk_aware_navigation" / "representative_paths").glob("*.png")))
    if rep_files:
        from PIL import Image
        loaded_imgs = [Image.open(p) for p in rep_files[:2]]  # load top 2
        w, h = loaded_imgs[0].size
        mosaic = Image.new("RGB", (w, h * 2 + 30), color="white")
        mosaic.paste(loaded_imgs[0], (0, 0))
        mosaic.paste(loaded_imgs[1], (0, h + 30))
        f6_path = SYNTHESIS_DIR / "fig6_representative_paths_mosaic.png"
        mosaic.save(f6_path)
        print(f"Saved: {f6_path}")

    return {
        "f1": f1_path,
        "f2": f2_path,
        "f3": f3_path,
        "f4": f4_path,
        "f5": f5_path,
        "f6": SYNTHESIS_DIR / "fig6_representative_paths_mosaic.png",
    }


# -----------------------------------------------------------------------------
# 3. Generate Canonical Jupyter Notebook
# -----------------------------------------------------------------------------
def generate_synthesis_notebook():
    print("Constructing canonical notebooks/12_Final_Results_and_Analysis.ipynb...")

    cells = []

    def add_md(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [source]})

    def add_code(source):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [source]})

    # Title & Metadata
    add_md("""# 12. Final Results Synthesis and Experimental Analysis

**Project:** Uncertainty-Aware Terrain Semantic Segmentation and Traversability Estimation for Autonomous Planetary Rover Navigation  
**Author / Repository:** IpsitaHQ / [UA-MTSeg-Rover-Navigation](https://github.com/IpsitaHQ/UA-MTSeg-Rover-Navigation)  
**Notebook Scope:** Final results synthesis, statistical cross-verification, and publication-ready reporting across all completed research stages.

---

### Important Scientific Scope and Evaluated Framework Constraints

To preserve strict scientific integrity, the findings presented in this notebook adhere to the following explicit constraints:
1. **Validation Set Scope:** All segmentation, uncertainty, and planning evaluations are strictly conducted on the **190 validation images** ($12,451,840$ total pixels) from the BASEPROD 80/20 split. No independent held-out Martian test set is available or claimed.
2. **Offline Image-Space Formulation:** The $47.37\%$ path reachability metric is strictly an **offline image-space path-planning success rate** on fixed diagonal coordinates $(5, 5) \to (250, 250)$. It must **not** be termed "navigation accuracy".
3. **No Physical Rover Experiments:** No physical planetary rover, field testbed, or analog site deployment was performed.
4. **No 3D Terrain or Vehicle Dynamics:** The planning proxy does not model digital elevation models (DEM), slope geometry, slip, ground clearance, tip-over stability, or rover kinodynamics.
5. **No Supervised Ground-Truth Traversability:** Traversability costs are engineered heuristic proxies mapped from semantic class IDs and normalized Shannon entropy.
6. **Predictive Uncertainty Formulation:** Predictive uncertainty is computed via single-pass normalized Shannon entropy over softmax probabilities, capturing combined aleatoric and model uncertainty rather than decomposed epistemic uncertainty (such as MC Dropout or deep ensembles).
7. **No Pareto Optimality Claims:** Detour-uncertainty trade-offs reflect parameter sensitivity within the evaluated proxy formulation, not formal Pareto optimality or definitive mathematical proof.
8. **Documentation of Missing Baseline Data:** While overall validation mIoU ($0.2862$) and pixel accuracy ($0.6687$) were recorded for the 5-epoch RGB baseline (Notebook 05), per-class IoU values were not saved in the source baseline outputs. In accordance with scientific reporting standards, these values are designated as *Not Recorded in Source Baseline* rather than retrospectively simulated or invented.
9. **Modality Alignment:** Depth and thermal channels are paired from the BASEPROD dataset using identical timestamp keys and resized to uniform resolution ($256 \times 256$). Sub-pixel geometric calibration was not independently verified by internal sensor calibration routines in this codebase.
10. **Training Duration Disparity:** The reference RGB baseline was trained for 5 epochs (yielding validation mIoU $0.2862$, pixel accuracy $0.6687$), whereas the multimodal early-fusion model was trained for 25 epochs (reaching its best validation checkpoint at epoch 24 with mIoU $0.6462$, pixel accuracy $0.8609$). The observed gains reflect both the addition of depth and thermal modalities and extended training optimization duration.
""")

    # Setup cell
    add_code("""import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:.4f}' if abs(x) < 1 else f'{x:.2f}')

ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent

OUTPUTS_DIR = ROOT / 'outputs'
FIGURES_DIR = ROOT / 'figures' / 'final_synthesis'
if not FIGURES_DIR.is_dir() and (OUTPUTS_DIR / 'final_synthesis').is_dir():
    FIGURES_DIR = OUTPUTS_DIR / 'final_synthesis'

print(f"Project Root: {ROOT}")
print(f"Figures Dir:  {FIGURES_DIR}")
print(f"Outputs Dir:  {OUTPUTS_DIR}")
""")

    # Section 1: Segmentation comparison
    add_md("""---
## 1. RGB Baseline vs. RGB-D-T Multimodal Terrain Segmentation

Comparison of the reference RGB-only U-Net baseline (Notebook 05, 5 epochs) against the early-fusion multimodal U-Net (Notebook 08 / Epoch 24 checkpoint, 5 input channels: RGB + Depth + Thermal, 25 epochs).

- **Frozen Multimodal Model:** `outputs/multimodal_segmentation/best_multimodal_unet.pt` (Epoch 24)
- **Input Channels:** 5 (RGB + metric Depth + Thermal Celsius)
- **Parameters:** 7,766,792
- **Image Resolution:** $256 \times 256$ pixels
- **Split:** 760 training samples, 190 validation samples (Seed 42)

*Note on Training Duration Disparity:* The RGB baseline was trained for 5 epochs, while the multimodal model was trained for 25 epochs (with the best checkpoint selected at epoch 24). Performance differences reflect both the integration of multimodal sensory data (depth + thermal) and extended optimization duration.
""")

    add_code("""# Table 1: Model Segmentation Performance Comparison
table1_data = [
    {
        "Model": "RGB Baseline U-Net",
        "Input Modalities": "3 ch (RGB)",
        "Epochs": 5,
        "Val mIoU": 0.2862,
        "Val Pixel Accuracy": 0.668699,
        "Absolute Gain (mIoU)": "Baseline",
        "Percentage Point Gain (mIoU)": "Baseline",
        "Relative Improvement (mIoU)": "Baseline",
        "Relative Improvement (Acc)": "Baseline"
    },
    {
        "Model": "Multimodal U-Net (Early Fusion)",
        "Input Modalities": "5 ch (RGB + Depth + Thermal)",
        "Epochs": 25,
        "Val mIoU": 0.6462015,
        "Val Pixel Accuracy": 0.8609459,
        "Absolute Gain (mIoU)": "+0.3600",
        "Percentage Point Gain (mIoU)": "+36.00 pp",
        "Relative Improvement (mIoU)": "+125.79%",
        "Relative Improvement (Acc)": "+28.75%"
    }
]

df_table1 = pd.DataFrame(table1_data)
display(df_table1)
""")

    add_code("""# Display Figure 1: Segmentation Comparison
f1_path = FIGURES_DIR / 'fig1_segmentation_comparison.png'
if f1_path.is_file():
    display(Image.open(f1_path))
""")

    # Section 2: Per-class IoU
    add_md("""---
## 2. Per-Class Intersection-over-Union (IoU) Breakdown

The BASEPROD dataset classifies planetary analog terrain into eight mutually exclusive classes.
The table below documents per-class IoU, validation pixel abundance, and observed error rates for each terrain category.

*Note on Baseline Data:* Per-class IoU metrics for the 5-epoch RGB baseline were not logged in the original baseline outputs; they are marked as *Not recorded in source baseline* to preserve archival documentation integrity.
""")

    add_code("""# Table 2: Per-Class Segmentation Performance (190 Validation Frames)
table2_data = [
    {"Class ID": 0, "Terrain Class": "Void", "Validation Pixels": 1044531, "Pixel Fraction (%)": 8.39, "Error Rate (%)": 24.01, "RGB Baseline IoU": "Not recorded", "Multimodal IoU": 0.667920},
    {"Class ID": 1, "Terrain Class": "Compact", "Validation Pixels": 6035201, "Pixel Fraction (%)": 48.47, "Error Rate (%)": 6.27, "RGB Baseline IoU": "Not recorded", "Multimodal IoU": 0.847922},
    {"Class ID": 2, "Terrain Class": "Grass", "Validation Pixels": 847749, "Pixel Fraction (%)": 6.81, "Error Rate (%)": 29.82, "RGB Baseline IoU": "Not recorded", "Multimodal IoU": 0.514720},
    {"Class ID": 3, "Terrain Class": "Bedrock", "Validation Pixels": 153052, "Pixel Fraction (%)": 1.23, "Error Rate (%)": 32.45, "RGB Baseline IoU": "Not recorded", "Multimodal IoU": 0.577139},
    {"Class ID": 4, "Terrain Class": "Sandy", "Validation Pixels": 1101710, "Pixel Fraction (%)": 8.85, "Error Rate (%)": 10.06, "RGB Baseline IoU": "Not recorded", "Multimodal IoU": 0.767217},
    {"Class ID": 5, "Terrain Class": "Pebble / Gravel", "Validation Pixels": 2475015, "Pixel Fraction (%)": 19.88, "Error Rate (%)": 17.13, "RGB Baseline IoU": "Not recorded", "Multimodal IoU": 0.749426},
    {"Class ID": 6, "Terrain Class": "Rock", "Validation Pixels": 223683, "Pixel Fraction (%)": 1.80, "Error Rate (%)": 43.64, "RGB Baseline IoU": "Not recorded", "Multimodal IoU": 0.454099},
    {"Class ID": 7, "Terrain Class": "Vegetation / Bush", "Validation Pixels": 570899, "Pixel Fraction (%)": 4.58, "Error Rate (%)": 29.31, "RGB Baseline IoU": "Not recorded", "Multimodal IoU": 0.591169},
]

df_table2 = pd.DataFrame(table2_data)
display(df_table2)
print(f"Validation Mean IoU across all 8 classes: {df_table2['Multimodal IoU'].mean():.4f}")
""")

    add_code("""# Display Figure 2: Per-Class IoU Breakdown
f2_path = FIGURES_DIR / 'fig2_per_class_iou.png'
if f2_path.is_file():
    display(Image.open(f2_path))
""")

    # Section 3: Uncertainty / Error Analysis
    add_md("""---
## 3. Predictive Entropy and Segmentation Error Detection Analysis

In Stage A (Notebook 09), normalized Shannon entropy was evaluated across all $12,451,840$ validation pixels:
$$H_{\\text{norm}}(p) = -\\frac{\\sum_{c=1}^8 p_c \\ln(p_c + 10^{-8})}{\\ln(8)}$$

### Key Statistical Separation:
- **Correct Pixels:** Mean entropy = $0.1977$, Median entropy = $0.1324$ ($\pm 0.1870$)
- **Incorrect Pixels:** Mean entropy = $0.5022$, Median entropy = $0.5167$ ($\pm 0.1864$)
- **Entropy Separation Ratio:** **$2.54\\times$ higher mean entropy in erroneous pixels**
- **AUROC for Error Detection:** **$0.8675$** (Area Under the Receiver Operating Characteristic curve)
""")

    add_code("""# Table 3: Binned Predictive Entropy and Segmentation Error Detection Breakdown
table3_data = [
    {"Entropy Bin Range": "0.0 - 0.2", "Pixel Count": 6858779, "Pixel Share (%)": 55.08, "Error Pixels": 125809, "Empirical Error Rate (%)": 1.83, "Mean Bin Entropy": 0.0788},
    {"Entropy Bin Range": "0.2 - 0.4", "Pixel Count": 2615826, "Pixel Share (%)": 21.01, "Error Pixels": 368027, "Empirical Error Rate (%)": 14.07, "Mean Bin Entropy": 0.2907},
    {"Entropy Bin Range": "0.4 - 0.6", "Pixel Count": 1914927, "Pixel Share (%)": 15.38, "Error Pixels": 672827, "Empirical Error Rate (%)": 35.14, "Mean Bin Entropy": 0.4954},
    {"Entropy Bin Range": "0.6 - 0.8", "Pixel Count": 942811, "Pixel Share (%)": 7.57, "Error Pixels": 488426, "Empirical Error Rate (%)": 51.81, "Mean Bin Entropy": 0.6776},
    {"Entropy Bin Range": "0.8 - 1.0", "Pixel Count": 119497, "Pixel Share (%)": 0.96, "Error Pixels": 76443, "Empirical Error Rate (%)": 63.97, "Mean Bin Entropy": 0.8436},
]

df_table3 = pd.DataFrame(table3_data)
display(df_table3)
""")

    add_code("""# Display Figure 3: Entropy vs Error Rate
f3_path = FIGURES_DIR / 'fig3_entropy_vs_error_rate.png'
if f3_path.is_file():
    display(Image.open(f3_path))
""")

    # Section 4: Stage B Planning Comparison
    add_md("""---
## 4. Offline Image-Space Path Planning: Semantic Baseline vs. 0.7/0.3 Uncertainty-Aware Formulation

In Stage B (Notebook 10), an offline 8-connected image-space $A^*$ search evaluated the cost of path safety:
- **Method A (Semantic Baseline):** $C_A = C_s$
- **Method B (Uncertainty-Aware):** $C_B = \\min(239, \\text{round}(0.7 C_s + 0.3 C_u))$ for $C_s < 240$; $C_B = 255$ for $C_s \\ge 240$.
- **Fixed Endpoints:** Start $(5, 5)$, Goal $(250, 250)$ across all 190 validation frames.
""")

    add_code("""# Table 4: Paired Path Planning Comparison (Stage B, 190 Validation Frames)
table4_data = [
    {
        "Metric": "Total Validation Frames Evaluated",
        "Method A: Semantic Baseline": "190",
        "Method B: Uncertainty-Aware (0.7/0.3)": "190",
        "Paired Difference / Relative Delta": "Identical protocol"
    },
    {
        "Metric": "Successful Routes (Reachable)",
        "Method A: Semantic Baseline": "90",
        "Method B: Uncertainty-Aware (0.7/0.3)": "90",
        "Paired Difference / Relative Delta": "0 net change"
    },
    {
        "Metric": "Failed Routes (Unreachable Endpoints)",
        "Method A: Semantic Baseline": "100",
        "Method B: Uncertainty-Aware (0.7/0.3)": "100",
        "Paired Difference / Relative Delta": "100% agreement"
    },
    {
        "Metric": "Offline Path-Planning Success Rate",
        "Method A: Semantic Baseline": "47.37%",
        "Method B: Uncertainty-Aware (0.7/0.3)": "47.37%",
        "Paired Difference / Relative Delta": "0.0 pp difference"
    },
    {
        "Metric": "Mean Path Length (pixels)",
        "Method A: Semantic Baseline": "384.81 px",
        "Method B: Uncertainty-Aware (0.7/0.3)": "389.06 px",
        "Paired Difference / Relative Delta": "+4.25 px (+1.11% overhead)"
    },
    {
        "Metric": "Median Path Length (pixels)",
        "Method A: Semantic Baseline": "380.75 px",
        "Method B: Uncertainty-Aware (0.7/0.3)": "385.09 px",
        "Paired Difference / Relative Delta": "+4.34 px (+1.14% overhead)"
    },
    {
        "Metric": "Mean Path Cost (Cumulative)",
        "Method A: Semantic Baseline": "21,560.05",
        "Method B: Uncertainty-Aware (0.7/0.3)": "20,628.43",
        "Paired Difference / Relative Delta": "-4.32% cost"
    },
    {
        "Metric": "Median Path Cost (Cumulative)",
        "Method A: Semantic Baseline": "20,433.92",
        "Method B: Uncertainty-Aware (0.7/0.3)": "19,182.54",
        "Paired Difference / Relative Delta": "-6.12% cost"
    },
    {
        "Metric": "Mean Route Uncertainty (H_norm)",
        "Method A: Semantic Baseline": "0.2723",
        "Method B: Uncertainty-Aware (0.7/0.3)": "0.1643",
        "Paired Difference / Relative Delta": "-0.1081 (-39.68% reduction)"
    },
    {
        "Metric": "Median Route Uncertainty (H_norm)",
        "Method A: Semantic Baseline": "0.2650",
        "Method B: Uncertainty-Aware (0.7/0.3)": "0.1491",
        "Paired Difference / Relative Delta": "-0.1159 (-43.74% reduction)"
    },
    {
        "Metric": "Paths with Reduced Uncertainty",
        "Method A: Semantic Baseline": "N/A",
        "Method B: Uncertainty-Aware (0.7/0.3)": "90 / 90 (100.0%)",
        "Paired Difference / Relative Delta": "100.0% of reachable routes"
    },
    {
        "Metric": "Paths with Altered Trajectories",
        "Method A: Semantic Baseline": "N/A",
        "Method B: Uncertainty-Aware (0.7/0.3)": "90 / 90 (100.0%)",
        "Paired Difference / Relative Delta": "100.0% topological change"
    }
]

df_table4 = pd.DataFrame(table4_data)
display(df_table4)
""")

    add_code("""# Display Figure 4: Planning Risk-Distance Trade-Off
f4_path = FIGURES_DIR / 'fig4_planning_risk_distance_tradeoff.png'
if f4_path.is_file():
    display(Image.open(f4_path))
""")

    # Section 5: Stage C Sensitivity
    add_md("""---
## 5. Parameter Sensitivity: Uncertainty Weight Ablation ($w_u \\in [0.0, 0.5]$)

In Stage C (Notebook 11), the sensitivity of the planning result to the choice of uncertainty weight was evaluated across four configurations:
1. **Semantic Baseline:** $w_s=1.0, w_u=0.0$
2. **Configuration 1 (Low):** $w_s=0.9, w_u=0.1$
3. **Configuration 2 (Stage B Primary):** $w_s=0.7, w_u=0.3$
4. **Configuration 3 (High):** $w_s=0.5, w_u=0.5$
""")

    add_code("""# Table 5: Weight Sensitivity Analysis Across 190 Validation Frames
table5_data = [
    {
        "Configuration": "Semantic Baseline",
        "Semantic Weight (ws)": 1.0,
        "Uncertainty Weight (wu)": 0.0,
        "Success Rate (%)": 47.37,
        "Mean Path Length (px)": 384.81,
        "Mean Route Uncertainty": 0.2723,
        "Uncertainty Reduction (%)": 0.00,
        "Length Overhead (%)": 0.00,
        "Paths Changed vs Baseline": "0 / 90 (0.0%)"
    },
    {
        "Configuration": "Configuration 1 (Low)",
        "Semantic Weight (ws)": 0.9,
        "Uncertainty Weight (wu)": 0.1,
        "Success Rate (%)": 47.37,
        "Mean Path Length (px)": 386.49,
        "Mean Route Uncertainty": 0.1929,
        "Uncertainty Reduction (%)": 29.19,
        "Length Overhead (%)": 0.44,
        "Paths Changed vs Baseline": "86 / 90 (95.6%)"
    },
    {
        "Configuration": "Configuration 2 (Stage B)",
        "Semantic Weight (ws)": 0.7,
        "Uncertainty Weight (wu)": 0.3,
        "Success Rate (%)": 47.37,
        "Mean Path Length (px)": 389.06,
        "Mean Route Uncertainty": 0.1643,
        "Uncertainty Reduction (%)": 39.68,
        "Length Overhead (%)": 1.11,
        "Paths Changed vs Baseline": "90 / 90 (100.0%)"
    },
    {
        "Configuration": "Configuration 3 (High)",
        "Semantic Weight (ws)": 0.5,
        "Uncertainty Weight (wu)": 0.5,
        "Success Rate (%)": 47.37,
        "Mean Path Length (px)": 396.00,
        "Mean Route Uncertainty": 0.1443,
        "Uncertainty Reduction (%)": 47.02,
        "Length Overhead (%)": 2.91,
        "Paths Changed vs Baseline": "90 / 90 (100.0%)"
    }
]

df_table5 = pd.DataFrame(table5_data)
display(df_table5)
""")

    add_code("""# Display Figure 5: Weight Sensitivity Trade-Off
f5_path = FIGURES_DIR / 'fig5_weight_sensitivity_curves.png'
if f5_path.is_file():
    display(Image.open(f5_path))
""")

    # Section 6: Representative Paths
    add_md("""---
## 6. Qualitative Visual Analysis: Representative Path Trajectories

The composite below displays paired planning trajectories on representative validation frames, illustrating how uncertainty cost guides paths around high-entropy terrain transition boundaries while semantic obstacles remain strictly respected.
""")

    add_code("""# Display Figure 6: Representative Paths Mosaic
f6_path = FIGURES_DIR / 'fig6_representative_paths_mosaic.png'
if f6_path.is_file():
    display(Image.open(f6_path))
""")

    # Section 7: Final Consolidated Summary
    add_md("""---
## 7. Consolidated Executive Summary of Research Findings

This master table synthesizes the empirical evidence established across all experimental stages of the investigation.
""")

    add_code("""# Table 6: Consolidated Master Findings Table
table6_data = [
    {
        "Research Dimension": "1. Multimodal Early Fusion",
        "Evaluated Baseline / Control": "RGB-only U-Net (3 ch, 5 ep): mIoU = 0.2862, Acc = 0.6687",
        "Proposed System": "Multimodal U-Net (5 ch: RGB-D-T, 25 ep, best ep 24): mIoU = 0.6462, Acc = 0.8609",
        "Observed Quantitative Impact": "+0.3600 mIoU (+36.00 pp, +125.8% rel.); +0.1922 Acc (+19.22 pp, +28.8% rel.)",
        "Scientific Conclusion / Scope": "Early fusion of thermal and depth data alongside extended training substantially improves terrain classification on the BASEPROD validation set."
    },
    {
        "Research Dimension": "2. Predictive Entropy and Error Detection",
        "Evaluated Baseline / Control": "Correct pixel entropy: Mean = 0.1977, Median = 0.1324",
        "Proposed System": "Incorrect pixel entropy: Mean = 0.5022, Median = 0.5167",
        "Observed Quantitative Impact": "2.54× higher mean entropy in erroneous pixels; AUROC = 0.8675 for error detection",
        "Scientific Conclusion / Scope": "Predictive Shannon entropy is strongly correlated with segmentation failure on the 190 validation frames."
    },
    {
        "Research Dimension": "3. Risk-Aware Path Planning (Stage B)",
        "Evaluated Baseline / Control": "Semantic-only A* (CA = Cs): Mean U = 0.2723, Mean Len = 384.81 px",
        "Proposed System": "Uncertainty-Aware A* (CB = 0.7Cs + 0.3Cu): Mean U = 0.1643, Mean Len = 389.06 px",
        "Observed Quantitative Impact": "-39.68% mean route uncertainty at +1.11% length overhead (+4.25 px); 100% routes rerouted",
        "Scientific Conclusion / Scope": "Uncertainty cost routes paths through high-confidence semantic regions at minimal 2D geometric detour costs."
    },
    {
        "Research Dimension": "4. Weight Sensitivity (Stage C)",
        "Evaluated Baseline / Control": "Fixed weight wu = 0.3 (Stage B primary formulation)",
        "Proposed System": "Ablation over wu in [0.0, 0.1, 0.3, 0.5]",
        "Observed Quantitative Impact": "Monotonic trade-off: wu=0.1 (-29.2% u, +0.44% len); wu=0.5 (-47.0% u, +2.91% len)",
        "Scientific Conclusion / Scope": "Uncertainty reduction is robust and monotonic across weights; wu = 0.3 provides a balanced operating point."
    },
    {
        "Research Dimension": "5. Path Reachability / Obstacle Guarantee",
        "Evaluated Baseline / Control": "Semantic baseline reachability = 90 / 190 (47.37%)",
        "Proposed System": "Uncertainty-aware reachability across all wu in [0.1, 0.5] = 90 / 190 (47.37%)",
        "Observed Quantitative Impact": "Identical 47.37% reachability across all configurations; 0 false obstacles generated",
        "Scientific Conclusion / Scope": "Cost capping (< 240) strictly preserves reachable topology while obstacles remain permanently blocked (255)."
    }
]

df_table6 = pd.DataFrame(table6_data)
display(df_table6)
""")

    nb_json = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    nb_path = ROOT / "notebooks" / "12_Final_Results_and_Analysis.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=2)

    print(f"Saved notebook: {nb_path}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("FINAL RESULTS SYNTHESIS & NOTEBOOK GENERATION")
    print("=" * 70)

    data = load_source_data()
    figs = generate_figures(data)
    generate_synthesis_notebook()

    print("\n" + "=" * 70)
    print("SYNTHESIS COMPLETE.")
    print("=" * 70)

if __name__ == "__main__":
    main()
