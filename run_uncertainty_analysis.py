"""
run_uncertainty_analysis.py
===========================
Executes Stage A: Predictive Entropy Uncertainty Estimation using the frozen
Epoch 24 MultimodalUNet model on all 190 validation images.

Tasks performed:
1. Evaluates all 190 validation images in eval mode with torch.no_grad().
2. Computes normalized predictive entropy: H_norm(p) = -sum(p_c * log(p_c)) / log(8)
3. Evaluates error vs. uncertainty across all 12.45 million validation pixels.
4. Computes mean/median uncertainty for correct vs. incorrect pixels.
5. Computes uncertainty bins ([0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]) and error rates.
6. Computes AUROC for predictive entropy identifying segmentation errors.
7. Computes per-class uncertainty breakdown across all 8 terrain classes.
8. Saves:
   - outputs/uncertainty/entropy_summary.csv
   - outputs/uncertainty/uncertainty_error_analysis.csv
   - outputs/uncertainty/uncertainty_metrics.json
   - outputs/uncertainty/uncertainty_vs_error_rate.png
   - outputs/uncertainty/predictions/ (4 5-panel validation figures)
9. Generates notebooks/09_Uncertainty_Estimation.ipynb containing the complete research methodology and results.
"""

import os
import sys
import json
import csv
import zipfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Exact Constants matching frozen model protocol
SEED = 42
TARGET_SIZE = (256, 256)
NUM_CLASSES = 8
CLASS_NAMES = [
    "Void", "Compact", "Grass", "Bedrock",
    "Sandy", "Pebble", "Rock", "Vegetation",
]

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "uncertainty"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
CANONICAL_CHECKPOINT = PROJECT_ROOT / "outputs" / "multimodal_segmentation" / "best_multimodal_unet.pt"
CACHE_DIR = PROJECT_ROOT / "outputs" / "multimodal_segmentation" / "cache"

DATASET_ROOT = Path(r"C:\Users\Anushka\data\BASEPROD_labeled")
ZIP_PATH = Path(r"C:\Users\Anushka\data\BASEPROD_labeled.zip")

# Model Definition matching frozen checkpoint
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class MultimodalUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=8):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


def get_validation_entries():
    """Reproduce exact dataset discovery and split."""
    rgb_dir = DATASET_ROOT / "BASEPROD" / "ORIGINAL" / "COLOR_IMGS"
    mask_dir = DATASET_ROOT / "BASEPROD" / "ORIGINAL" / "MASK_RAWS"
    rgb_files = sorted(rgb_dir.glob("*.png"))
    mask_files = sorted(mask_dir.glob("*_mask.png"))
    paired = {p.stem: {"rgb": p} for p in rgb_files}
    mask_keys = {m.name.replace("_mask.png", "") for m in mask_files}
    valid_keys = sorted(set(paired) & mask_keys)
    combined = {
        k: {
            "rgb": paired[k]["rgb"],
            "mask": mask_dir / f"{k}_mask.png",
        }
        for k in valid_keys
    }

    # Index ZIP
    zip_map = {"depth": {}, "thermal": {}}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in zf.namelist():
            low = name.lower()
            if low.endswith("d.csv") and "depth" in low:
                key = Path(name).stem.replace("d", "")
                zip_map["depth"][key] = name
            if low.endswith("t.csv") and "thermal" in low:
                key = Path(name).stem.replace("t", "")
                zip_map["thermal"][key] = name

    shared_keys = sorted(set(combined) & set(zip_map["depth"]) & set(zip_map["thermal"]))
    entries = []
    for key in shared_keys:
        entries.append({
            "key": key,
            "rgb_path": str(combined[key]["rgb"]),
            "mask_path": str(combined[key]["mask"]),
        })

    rng = np.random.RandomState(SEED)
    keys = np.array([e["key"] for e in entries])
    rng.shuffle(keys)
    val_size = max(1, int(round(len(keys) * 0.2)))
    val_keys = set(keys[:val_size])
    val_entries = [e for e in entries if e["key"] in val_keys]
    return val_entries


def compute_normalized_entropy(probs, eps=1e-8):
    """
    Computes Shannon entropy normalized to [0, 1] using base log(num_classes):
    H_norm(p) = -sum_c p_c * log(p_c + eps) / log(num_classes)
    """
    # probs: tensor of shape (B, C, H, W) or (C, H, W)
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=1) # shape: (B, H, W)
    max_entropy = np.log(NUM_CLASSES)
    norm_entropy = entropy / max_entropy
    return torch.clamp(norm_entropy, 0.0, 1.0)


def generate_notebook():
    """Builds notebooks/09_Uncertainty_Estimation.ipynb with complete structure."""
    import nbformat as nbf
    nb = nbf.v4.new_notebook()

    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# Notebook 09 — Predictive Entropy Uncertainty Estimation
**Project**: Uncertainty-Aware Terrain Semantic Segmentation and Traversability Estimation for Autonomous Planetary Rover Navigation  
**Stage A**: Predictive Entropy Uncertainty Analysis on Verified Epoch 24 Multimodal Model

---

## 1. Research Objectives
This notebook evaluates the hypothesis:
> *"Does predictive uncertainty identify pixels where the multimodal segmentation model is more likely to be incorrect?"*

We employ **Shannon Predictive Entropy** derived from class posterior probabilities:
$$H(p) = -\\sum_{c=0}^{C-1} p_c \\ln(p_c)$$
Normalized to $[0, 1]$ via the maximum theoretical entropy $\\ln(C) = \\ln(8) \\approx 2.0794$:
$$H_{\\text{norm}}(p) = \\frac{H(p)}{\\ln(8)} \\in [0, 1]$$

Where:
- $H_{\\text{norm}}(p) = 0.0$: Maximum confidence (model assigns probability 1.0 to a single terrain class).
- $H_{\\text{norm}}(p) = 1.0$: Complete uncertainty (uniform distribution over all 8 terrain classes)."""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path.cwd().resolve()
if (PROJECT_ROOT / "notebooks").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "uncertainty"

# Verify summary metrics
metrics_path = OUTPUT_DIR / "uncertainty_metrics.json"
with open(metrics_path, "r") as f:
    metrics = json.load(f)

print("=== Predictive Entropy Uncertainty Metrics ===")
print(f"Validation Frames Evaluated: {metrics['validation_samples']}")
print(f"Total Validation Pixels:    {metrics['total_pixels']:,}")
print(f"Mean Entropy (Correct):     {metrics['overall_uncertainty']['mean_entropy_correct']:.4f}")
print(f"Mean Entropy (Incorrect):   {metrics['overall_uncertainty']['mean_entropy_incorrect']:.4f}")
print(f"Entropy Separation Ratio:   {metrics['overall_uncertainty']['entropy_separation_ratio']:.2f}x")
print(f"AUROC (Error Detection):    {metrics['overall_uncertainty']['auroc_error_detection']:.4f}")"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Uncertainty Bins vs. Empirical Error Rate
Pixels are partitioned into 5 uncertainty intervals $[0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]$.
A reliable uncertainty metric should exhibit a strong monotonic relationship between uncertainty and segmentation error rate."""))

    cells.append(nbf.v4.new_code_cell("""df_bins = pd.read_csv(OUTPUT_DIR / "uncertainty_error_analysis.csv")
print(df_bins.to_markdown(index=False))

# Plot Error Rate vs Uncertainty Bin
fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:red'
ax1.set_xlabel('Predictive Entropy Bin', fontsize=12, fontweight='bold')
ax1.set_ylabel('Error Rate (%)', color=color, fontsize=12, fontweight='bold')
bars = ax1.bar(df_bins['bin_range'], df_bins['error_rate_pct'], color=color, alpha=0.7, edgecolor='black', width=0.5)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.title('Predictive Uncertainty vs. Segmentation Error Rate', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Per-Class Uncertainty Analysis
Examines whether difficult terrain classes (such as Rock, Bedrock, and Grass) manifest higher predictive entropy compared to expansive homogeneous terrain (Compact, Sandy)."""))

    cells.append(nbf.v4.new_code_cell("""df_class = pd.read_csv(OUTPUT_DIR / "entropy_summary.csv")
print(df_class.to_markdown(index=False))"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Qualitative Visualizations: Uncertainty & Error Maps
Each sample displays:
1. Input RGB
2. Ground-Truth Mask
3. Predicted Semantic Mask
4. Predictive Entropy Map (Normalized Shannon Entropy)
5. Prediction Error Map (Red = Segmentation Error)"""))

    cells.append(nbf.v4.new_code_cell("""pred_imgs = sorted((OUTPUT_DIR / "predictions").glob("*.png"))
for p in pred_imgs:
    print(f"Displaying: {p.name}")
    img = Image.open(p)
    plt.figure(figsize=(16, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.show()"""))

    nb.cells = cells
    nb_path = PROJECT_ROOT / "notebooks" / "09_Uncertainty_Estimation.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated notebook: {nb_path}")


def main():
    print("=" * 70)
    print("STAGE A: PREDICTIVE ENTROPY UNCERTAINTY ESTIMATION")
    print("=" * 70)

    # 1. Setup paths
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Check checkpoint existence
    if not CANONICAL_CHECKPOINT.exists():
        print(f"ERROR: Frozen checkpoint not found at {CANONICAL_CHECKPOINT}")
        sys.exit(1)

    print(f"Loading frozen Epoch 24 model: {CANONICAL_CHECKPOINT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    ckpt = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = MultimodalUNet(in_channels=5, out_channels=NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    print(f"Model successfully loaded. Best epoch: {ckpt.get('epoch')}, Best mIoU: {ckpt.get('best_val_miou'):.4f}")

    # 3. Validation Dataset Discovery
    val_entries = get_validation_entries()
    print(f"Discovered {len(val_entries)} validation images.")
    if len(val_entries) != 190:
        print(f"WARNING: Expected 190 validation entries, found {len(val_entries)}")

    transform_rgb = transforms.Compose([
        transforms.Resize(TARGET_SIZE, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    depth_cache = CACHE_DIR / "depth"
    thermal_cache = CACHE_DIR / "thermal"

    # Tracking arrays across all validation pixels
    all_entropies_correct = []
    all_entropies_incorrect = []

    # Per-class accumulators
    class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_error_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_entropy_correct_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_entropy_correct_count = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_entropy_incorrect_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_entropy_incorrect_count = np.zeros(NUM_CLASSES, dtype=np.int64)

    # Subsampled arrays for global AUROC calculation (1-in-10 subsampling for speed, 1.25M pixels is statistically exact)
    auroc_labels = []
    auroc_scores = []

    # Bins: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    bin_names = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    bin_pixel_counts = np.zeros(5, dtype=np.int64)
    bin_error_counts = np.zeros(5, dtype=np.int64)
    bin_entropy_sums = np.zeros(5, dtype=np.float64)

    # Color palette for masks
    class_colors = [
        [0.1, 0.1, 0.1],   # 0: Void (charcoal)
        [0.7, 0.55, 0.4],  # 1: Compact (tan/brown)
        [0.4, 0.8, 0.3],   # 2: Grass (light green)
        [0.5, 0.5, 0.6],   # 3: Bedrock (slate gray)
        [0.9, 0.8, 0.4],   # 4: Sandy (golden sand)
        [0.85, 0.5, 0.2],  # 5: Pebble (terracotta)
        [0.6, 0.2, 0.2],   # 6: Rock (deep red/brown)
        [0.1, 0.45, 0.15], # 7: Vegetation (forest green)
    ]
    cmap_classes = ListedColormap(class_colors)
    bounds = np.arange(NUM_CLASSES + 1) - 0.5
    norm_classes = BoundaryNorm(bounds, cmap_classes.N)

    # Select 4 representative samples for qualitative visualization (diverse terrain classes)
    vis_candidates = []
    t0 = time.time()

    print("\nRunning inference & uncertainty estimation across validation set...")
    with torch.no_grad():
        for idx, e in enumerate(val_entries, start=1):
            key = e["key"]

            # Load modalities exactly matching reference protocol
            rgb_img = Image.open(e["rgb_path"]).convert("RGB")
            rgb_t = transform_rgb(rgb_img)
            rgb_disp = np.array(rgb_img.resize(TARGET_SIZE, resample=Image.BILINEAR))

            mask_img = Image.open(e["mask_path"])
            gt = np.asarray(
                Image.fromarray(np.array(mask_img, dtype=np.uint8)).resize(TARGET_SIZE, resample=Image.NEAREST),
                dtype=np.int64
            )

            depth = np.load(depth_cache / f"{key}.npy")
            thermal = np.load(thermal_cache / f"{key}.npy")
            depth_img = Image.fromarray(depth.astype(np.float32))
            thermal_img = Image.fromarray(thermal.astype(np.float32))
            depth_t = torch.tensor(
                np.asarray(depth_img.resize(TARGET_SIZE, resample=Image.BILINEAR), dtype=np.float32)
            ).unsqueeze(0)
            thermal_t = torch.tensor(
                np.asarray(thermal_img.resize(TARGET_SIZE, resample=Image.BILINEAR), dtype=np.float32)
            ).unsqueeze(0)

            # Assemble 5-channel tensor [1, 5, 256, 256]
            x = torch.cat([rgb_t, depth_t / 255.0, thermal_t / 255.0], dim=0).unsqueeze(0).to(device)

            # Inference
            logits = model(x)
            probs = F.softmax(logits, dim=1) # [1, 8, 256, 256]
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy() # [256, 256]

            # Predictive Entropy
            entropy_norm = compute_normalized_entropy(probs).squeeze(0).cpu().numpy() # [256, 256]

            # Error identification
            is_correct = (pred == gt)
            is_error = ~is_correct

            # Flatten
            ent_flat = entropy_norm.ravel()
            corr_flat = is_correct.ravel()
            err_flat = is_error.ravel()
            gt_flat = gt.ravel()

            # Record stats
            all_entropies_correct.append(ent_flat[corr_flat])
            all_entropies_incorrect.append(ent_flat[err_flat])

            # Subsample for AUROC (take every 10th pixel)
            auroc_labels.append(err_flat[::10].astype(np.uint8))
            auroc_scores.append(ent_flat[::10].astype(np.float32))

            # Binning
            bin_idx = np.digitize(ent_flat, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, 4)
            for b in range(5):
                in_b = (bin_idx == b)
                bin_pixel_counts[b] += in_b.sum()
                bin_error_counts[b] += (in_b & err_flat).sum()
                bin_entropy_sums[b] += ent_flat[in_b].sum()

            # Per-class stats
            for c in range(NUM_CLASSES):
                c_mask = (gt_flat == c)
                c_count = c_mask.sum()
                if c_count > 0:
                    class_pixel_counts[c] += c_count
                    c_corr = c_mask & corr_flat
                    c_err = c_mask & err_flat
                    class_error_counts[c] += c_err.sum()
                    class_entropy_correct_sum[c] += ent_flat[c_corr].sum()
                    class_entropy_correct_count[c] += c_corr.sum()
                    class_entropy_incorrect_sum[c] += ent_flat[c_err].sum()
                    class_entropy_incorrect_count[c] += c_err.sum()

            # Save qualitative visualization candidates
            uniques = np.unique(gt)
            vis_candidates.append((len(uniques), err_flat.mean(), key, rgb_disp, gt, pred, entropy_norm, is_error))

            if idx % 38 == 0 or idx == len(val_entries):
                print(f"  Processed validation images: {idx}/{len(val_entries)} ({(idx/len(val_entries))*100:.1f}%)")

    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.1f}s.")

    # Concatenate correct & incorrect arrays
    all_corr_ent = np.concatenate(all_entropies_correct)
    all_err_ent = np.concatenate(all_entropies_incorrect)
    total_pixels = len(all_corr_ent) + len(all_err_ent)
    overall_acc = len(all_corr_ent) / total_pixels

    mean_corr = float(all_corr_ent.mean())
    median_corr = float(np.median(all_corr_ent))
    std_corr = float(all_corr_ent.std())

    mean_err = float(all_err_ent.mean())
    median_err = float(np.median(all_err_ent))
    std_err = float(all_err_ent.std())

    print("\n" + "=" * 70)
    print("UNCERTAINTY VS ERROR METRICS (OVERALL)")
    print("=" * 70)
    print(f"Total Pixels Evaluated:     {total_pixels:,}")
    print(f"Overall Pixel Accuracy:     {overall_acc * 100:.2f}% (Error Rate: {(1 - overall_acc) * 100:.2f}%)")
    print(f"Correct Pixels (N={len(all_corr_ent):,}):")
    print(f"  Mean Predictive Entropy:   {mean_corr:.4f} (+/- {std_corr:.4f})")
    print(f"  Median Predictive Entropy: {median_corr:.4f}")
    print(f"Incorrect Pixels (N={len(all_err_ent):,}):")
    print(f"  Mean Predictive Entropy:   {mean_err:.4f} (+/- {std_err:.4f})")
    print(f"  Median Predictive Entropy: {median_err:.4f}")
    print(f"Separation (Mean Err / Mean Corr): {mean_err / max(mean_corr, 1e-6):.2f}x higher uncertainty on errors!")

    # AUROC calculation
    auroc_y = np.concatenate(auroc_labels)
    auroc_score_vals = np.concatenate(auroc_scores)
    auroc = float(roc_auc_score(auroc_y, auroc_score_vals))
    print(f"AUROC (Predictive Entropy detecting Errors): {auroc:.4f}")

    # Build Uncertainty Bins Table
    bin_rows = []
    print("\n--- Uncertainty Bins vs. Error Rate ---")
    for b in range(5):
        cnt = int(bin_pixel_counts[b])
        errs = int(bin_error_counts[b])
        pct_err = (errs / cnt * 100.0) if cnt > 0 else 0.0
        pct_of_total_pixels = (cnt / total_pixels * 100.0)
        mean_b_ent = (bin_entropy_sums[b] / cnt) if cnt > 0 else 0.0
        bin_rows.append({
            "bin_id": b,
            "bin_range": bin_names[b],
            "pixel_count": cnt,
            "pixel_fraction_pct": round(pct_of_total_pixels, 2),
            "error_count": errs,
            "error_rate_pct": round(pct_err, 2),
            "mean_bin_entropy": round(mean_b_ent, 4),
        })
        print(f"  Bin {bin_names[b]:10s}: {cnt:>10,} px ({pct_of_total_pixels:5.1f}%) | Errors: {errs:>8,} | Error Rate: {pct_err:5.2f}%")

    df_bins = pd.DataFrame(bin_rows)
    df_bins.to_csv(OUTPUT_DIR / "uncertainty_error_analysis.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'uncertainty_error_analysis.csv'}")

    # Plot Uncertainty vs Error Rate
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bar_color = '#d9534f'
    line_color = '#0275d8'

    bars = ax1.bar(df_bins['bin_range'], df_bins['error_rate_pct'], color=bar_color, alpha=0.75, edgecolor='black', width=0.45, label='Error Rate (%)')
    lines = ax2.plot(df_bins['bin_range'], df_bins['pixel_fraction_pct'], color=line_color, marker='o', linewidth=2.5, label='Pixel Fraction (%)')

    ax1.set_xlabel('Normalized Predictive Entropy Bin', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Empirical Error Rate (%)', color=bar_color, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fraction of Total Pixels (%)', color=line_color, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=bar_color)
    ax2.tick_params(axis='y', labelcolor=line_color)
    ax1.set_ylim(0, 100)

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.title(f'Predictive Uncertainty vs. Segmentation Error Rate\n(AUROC = {auroc:.4f})', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "uncertainty_vs_error_rate.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

    # Build Per-Class Summary Table
    print("\n--- Per-Class Predictive Entropy Summary ---")
    class_rows = []
    for c in range(NUM_CLASSES):
        tot_c = int(class_pixel_counts[c])
        err_c = int(class_error_counts[c])
        err_rate_c = (err_c / tot_c * 100.0) if tot_c > 0 else 0.0

        n_corr = int(class_entropy_correct_count[c])
        mean_ent_corr = (class_entropy_correct_sum[c] / n_corr) if n_corr > 0 else 0.0

        n_err = int(class_entropy_incorrect_count[c])
        mean_ent_err = (class_entropy_incorrect_sum[c] / n_err) if n_err > 0 else 0.0

        row = {
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "total_pixels": tot_c,
            "error_pixels": err_c,
            "class_error_rate_pct": round(err_rate_c, 2),
            "mean_entropy_correct": round(mean_ent_corr, 4),
            "mean_entropy_incorrect": round(mean_ent_err, 4),
            "uncertainty_separation": round(mean_ent_err / max(mean_ent_corr, 1e-4), 2) if n_corr > 0 else 0.0
        }
        class_rows.append(row)
        print(f"  Class {c} ({CLASS_NAMES[c]:10s}): {tot_c:>9,} px | Err Rate: {err_rate_c:5.1f}% | Correct Ent: {mean_ent_corr:.4f} | Error Ent: {mean_ent_err:.4f}")

    df_class = pd.DataFrame(class_rows)
    df_class.to_csv(OUTPUT_DIR / "entropy_summary.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'entropy_summary.csv'}")

    # Save comprehensive uncertainty metrics JSON
    metrics_json = {
        "model_checkpoint": str(CANONICAL_CHECKPOINT),
        "validation_samples": len(val_entries),
        "total_pixels": int(total_pixels),
        "image_resolution": list(TARGET_SIZE),
        "entropy_formula": "H_norm(p) = -sum(p_c * log(p_c + 1e-8)) / log(8)",
        "overall_pixel_accuracy": round(overall_acc, 6),
        "overall_error_rate": round(1.0 - overall_acc, 6),
        "overall_uncertainty": {
            "mean_entropy_correct": round(mean_corr, 6),
            "std_entropy_correct": round(std_corr, 6),
            "median_entropy_correct": round(median_corr, 6),
            "mean_entropy_incorrect": round(mean_err, 6),
            "std_entropy_incorrect": round(std_err, 6),
            "median_entropy_incorrect": round(median_err, 6),
            "entropy_separation_ratio": round(mean_err / max(mean_corr, 1e-6), 4),
            "auroc_error_detection": round(auroc, 6)
        },
        "uncertainty_bins": bin_rows,
        "per_class_uncertainty": class_rows,
        "experimental_protocol": {
            "dataset_split": "80/20 train/validation split (190 validation frames)",
            "random_seed": SEED,
            "normalization": "RGB to [0,1], Depth / 255.0, Thermal / 255.0",
            "evaluation_mode": "torch.no_grad(), eval mode",
            "date_executed": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    with open(OUTPUT_DIR / "uncertainty_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'uncertainty_metrics.json'}")

    # 4. Generate 4 Representative 5-Panel Figures
    print("\nGenerating 4 representative 5-panel validation qualitative figures...")
    # Sort candidates by number of classes descending, then error rate
    vis_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected_vis = vis_candidates[:4]

    for v_idx, (num_cls, err_rate, key, rgb_disp, gt, pred, ent_map, is_err) in enumerate(selected_vis, start=1):
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

        # Panel 1: RGB Image
        axes[0].imshow(rgb_disp)
        axes[0].set_title(f"1. Input RGB\n({key})", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Ground-Truth Mask
        axes[1].imshow(gt, cmap=cmap_classes, norm=norm_classes)
        axes[1].set_title("2. Ground-Truth Mask", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Predicted Semantic Mask
        sample_acc = (1.0 - err_rate) * 100
        axes[2].imshow(pred, cmap=cmap_classes, norm=norm_classes)
        axes[2].set_title(f"3. Multimodal Pred\n(Acc: {sample_acc:.1f}%)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # Panel 4: Predictive Entropy Uncertainty Map
        im_ent = axes[3].imshow(ent_map, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[3].set_title(f"4. Predictive Entropy\n(Mean: {ent_map.mean():.3f})", fontsize=11, fontweight="bold")
        axes[3].axis("off")
        cbar = fig.colorbar(im_ent, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        # Panel 5: Prediction Error Map
        # Black = Correct, Bright Red = Error
        err_display = np.zeros((*TARGET_SIZE, 3), dtype=np.float32)
        err_display[is_err] = [0.95, 0.2, 0.2] # Bright red for errors
        err_display[~is_err] = [0.1, 0.1, 0.1] # Dark gray for correct
        axes[4].imshow(err_display)
        axes[4].set_title(f"5. Error Map\n(Red: Error, {err_rate*100:.1f}%)", fontsize=11, fontweight="bold")
        axes[4].axis("off")

        # Legend for classes
        legend_patches = [
            mpatches.Patch(color=class_colors[c], label=CLASS_NAMES[c])
            for c in range(NUM_CLASSES)
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=8,
            bbox_to_anchor=(0.5, -0.06),
            frameon=True,
            fontsize=9
        )

        fig.tight_layout()
        out_fig_path = PREDICTIONS_DIR / f"val_uncertainty_sample_{v_idx:02d}_{key}.png"
        fig.savefig(out_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_fig_path.name}")

    # Generate Notebook 09
    print("\nGenerating notebooks/09_Uncertainty_Estimation.ipynb...")
    generate_notebook()

    print("\nSTAGE A EXPERIMENT SUCCESSFULLY COMPLETED.")

if __name__ == "__main__":
    main()
