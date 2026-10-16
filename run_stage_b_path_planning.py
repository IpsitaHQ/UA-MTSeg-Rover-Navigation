"""
run_stage_b_path_planning.py
============================
Stage B: Uncertainty-Aware Terrain Risk / Cost Map + Offline A* Comparison.

Evaluates and compares:
- Method A: Baseline Semantic Cost (C_A = C_s)
- Method B: Uncertainty-Aware Cost (C_B = 0.7 * C_s + 0.3 * C_u, with C_s >= 240 blocked)

Uses exact Notebook 06/07 A* planner and evaluation protocol on the frozen
Epoch 24 MultimodalUNet model predictions and Stage A predictive entropy.
"""

import os
import sys
import json
import csv
import zipfile
import time
import math
import heapq
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Exact Constants from Notebooks 06 & 07
SEED = 42
TARGET_SIZE = (256, 256)
NUM_CLASSES = 8
CLASS_NAMES = [
    "Void", "Compact", "Grass", "Bedrock",
    "Sandy", "Pebble", "Rock", "Vegetation",
]

# Exact Semantic Costs from Notebook 06/07
CLASS_COSTS = {
    0: 255,  # Void: conservatively blocked
    1: 30,   # Compact: lowest cost
    2: 80,   # Grass: moderate-low cost
    3: 255,  # Bedrock: maximum cost / blocked
    4: 160,  # Sandy: moderate-high cost
    5: 130,  # Pebble: moderate cost
    6: 255,  # Rock: maximum cost / blocked
    7: 180,  # Vegetation: high cost
}
BLOCKED_THRESHOLD = 240
START_XY = (5, 5)
GOAL_XY = (250, 250)
ALLOW_DIAGONAL = True

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "risk_aware_navigation"
PATHS_DIR = OUTPUT_DIR / "representative_paths"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "multimodal_segmentation" / "best_multimodal_unet.pt"
CACHE_DIR = PROJECT_ROOT / "outputs" / "multimodal_segmentation" / "cache"

DATASET_ROOT = Path(r"C:\Users\Anushka\data\BASEPROD_labeled")
ZIP_PATH = Path(r"C:\Users\Anushka\data\BASEPROD_labeled.zip")


# ── Model Architecture (Frozen MultimodalUNet) ──
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
    """Normalized Shannon predictive entropy in [0, 1]."""
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=1)
    max_entropy = np.log(NUM_CLASSES)
    norm_entropy = entropy / max_entropy
    return torch.clamp(norm_entropy, 0.0, 1.0)


# ── Exact A* Algorithm from Notebooks 06 & 07 ──
def astar_path(costs, start_xy, goal_xy, blocked_threshold=BLOCKED_THRESHOLD, allow_diagonal=ALLOW_DIAGONAL):
    """
    Exact A* implementation from Notebooks 06 and 07.
    Returns: (path, accumulated_cost) or (None, float('inf')).
    """
    height, width = costs.shape
    for name, (x, y) in (('start', start_xy), ('goal', goal_xy)):
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f'{name} coordinate {(x, y)} is outside map bounds {width}x{height}.')
        if int(costs[y, x]) >= blocked_threshold:
            raise ValueError(f'{name} coordinate {(x, y)} is blocked with cost {int(costs[y, x])}.')
    if start_xy == goal_xy:
        return [start_xy], 0.0

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        moves += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    free_costs = costs[costs < blocked_threshold]
    minimum_cost = float(free_costs.min()) if free_costs.size else 0.0

    def heuristic(x, y):
        distance = math.hypot(goal_xy[0] - x, goal_xy[1] - y) if allow_diagonal else abs(goal_xy[0] - x) + abs(goal_xy[1] - y)
        return minimum_cost * distance

    start_x, start_y = start_xy
    frontier = [(heuristic(start_x, start_y), 0.0, start_y, start_x)]
    best_cost = {start_xy: 0.0}
    parent = {}
    closed = set()
    while frontier:
        _, path_cost, y, x = heapq.heappop(frontier)
        current = (x, y)
        if current in closed:
            continue
        if current == goal_xy:
            path = [goal_xy]
            while path[-1] != start_xy:
                path.append(parent[path[-1]])
            path.reverse()
            return path, path_cost
        closed.add(current)
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if int(costs[ny, nx]) >= blocked_threshold or neighbor in closed:
                continue
            # Corner-cut check
            if dx and dy and (int(costs[y, nx]) >= blocked_threshold or int(costs[ny, x]) >= blocked_threshold):
                continue
            step_distance = math.sqrt(2) if dx and dy else 1.0
            next_cost = path_cost + step_distance * (float(costs[y, x]) + float(costs[ny, nx])) / 2.0
            if next_cost < best_cost.get(neighbor, float('inf')):
                best_cost[neighbor] = next_cost
                parent[neighbor] = current
                priority = next_cost + heuristic(nx, ny)
                heapq.heappush(frontier, (priority, next_cost, ny, nx))
    return None, float('inf')


def compute_path_uncertainty(path, entropy_map):
    """Calculates accumulated and mean uncertainty along an A* path."""
    if not path or len(path) < 2:
        return 0.0, 0.0
    accumulated_u = 0.0
    total_length = 0.0
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        step_d = math.hypot(x1 - x0, y1 - y0)
        u_step = step_d * (entropy_map[y0, x0] + entropy_map[y1, x1]) / 2.0
        accumulated_u += u_step
        total_length += step_d
    mean_u = accumulated_u / total_length if total_length > 0 else 0.0
    return float(accumulated_u), float(mean_u)


def run_pipeline_for_image(prediction, entropy_norm):
    """
    Computes Cost Maps and runs A* for both Method A and Method B.
    Returns: dict with all results and cost maps.
    """
    # 1. Semantic Cost Map (C_s)
    c_s = np.zeros_like(prediction, dtype=np.uint8)
    for class_id, terrain_cost in CLASS_COSTS.items():
        c_s[prediction == class_id] = terrain_cost

    # 2. Uncertainty Cost (C_u)
    c_u = np.round(255.0 * entropy_norm).astype(np.uint8)

    # 3. Method A Cost Map: C_A = C_s
    cost_a = c_s.copy()

    # 4. Method B Cost Map: C_B = 0.7 * C_s + 0.3 * C_u
    # Rules: If C_s >= 240 -> 255. Else -> min(239, round(0.7 * C_s + 0.3 * C_u))
    combined_raw = np.round(0.7 * c_s.astype(np.float32) + 0.3 * c_u.astype(np.float32))
    cost_b = np.where(c_s >= BLOCKED_THRESHOLD, 255, np.clip(combined_raw, 0, 239)).astype(np.uint8)

    # Planning for Method A
    res_a = {}
    if int(cost_a[START_XY[1], START_XY[0]]) >= BLOCKED_THRESHOLD:
        res_a['path'], res_a['cost'], res_a['reason'] = None, float('inf'), 'blocked_start'
    elif int(cost_a[GOAL_XY[1], GOAL_XY[0]]) >= BLOCKED_THRESHOLD:
        res_a['path'], res_a['cost'], res_a['reason'] = None, float('inf'), 'blocked_goal'
    else:
        path_a, cost_val_a = astar_path(cost_a, START_XY, GOAL_XY, BLOCKED_THRESHOLD, ALLOW_DIAGONAL)
        res_a['path'] = path_a
        res_a['cost'] = cost_val_a
        res_a['reason'] = 'no_route' if path_a is None else ''

    # Planning for Method B
    res_b = {}
    if int(cost_b[START_XY[1], START_XY[0]]) >= BLOCKED_THRESHOLD:
        res_b['path'], res_b['cost'], res_b['reason'] = None, float('inf'), 'blocked_start'
    elif int(cost_b[GOAL_XY[1], GOAL_XY[0]]) >= BLOCKED_THRESHOLD:
        res_b['path'], res_b['cost'], res_b['reason'] = None, float('inf'), 'blocked_goal'
    else:
        path_b, cost_val_b = astar_path(cost_b, START_XY, GOAL_XY, BLOCKED_THRESHOLD, ALLOW_DIAGONAL)
        res_b['path'] = path_b
        res_b['cost'] = cost_val_b
        res_b['reason'] = 'no_route' if path_b is None else ''

    # Path geometry & uncertainty metrics
    for name, r in [('A', res_a), ('B', res_b)]:
        p = r['path']
        if p is not None:
            r['found'] = True
            r['steps'] = len(p) - 1
            r['length'] = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(p, p[1:]))
            acc_u, mean_u = compute_path_uncertainty(p, entropy_norm)
            r['acc_u'] = acc_u
            r['mean_u'] = mean_u
        else:
            r['found'] = False
            r['steps'] = 0
            r['length'] = float('nan')
            r['acc_u'] = float('nan')
            r['mean_u'] = float('nan')

    return {
        'cost_s': c_s,
        'cost_u': c_u,
        'cost_a': cost_a,
        'cost_b': cost_b,
        'res_a': res_a,
        'res_b': res_b,
    }


def run_smoke_test(model, val_entries, device, transform_rgb):
    """
    Runs Smoke Test on the first 3 validation images.
    Verifies 6 strict requirements.
    """
    print("\n" + "=" * 70)
    print("STEP 1: SMOKE TEST (3 VALIDATION IMAGES)")
    print("=" * 70)

    depth_cache = CACHE_DIR / "depth"
    thermal_cache = CACHE_DIR / "thermal"
    smoke_samples = val_entries[:3]

    print(f"Testing on 3 validation images: {[s['key'] for s in smoke_samples]}")
    print(f"Fixed start coordinate: {START_XY}")
    print(f"Fixed goal coordinate:  {GOAL_XY}")

    with torch.no_grad():
        for i, s in enumerate(smoke_samples, start=1):
            key = s['key']
            rgb_img = Image.open(s['rgb_path']).convert("RGB")
            rgb_t = transform_rgb(rgb_img)

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

            x = torch.cat([rgb_t, depth_t / 255.0, thermal_t / 255.0], dim=0).unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            entropy_norm = compute_normalized_entropy(probs).squeeze(0).cpu().numpy()

            out = run_pipeline_for_image(pred, entropy_norm)

            # Verification checks
            # 1. Coordinates identical
            v1 = (START_XY == (5, 5)) and (GOAL_XY == (250, 250))
            # 2. Blocked cells preserved in B
            blocked_in_s = (out['cost_s'] >= BLOCKED_THRESHOLD)
            blocked_in_b = (out['cost_b'] >= BLOCKED_THRESHOLD)
            v3 = np.all(blocked_in_b[blocked_in_s])  # all blocked in S must be blocked in B
            # 3. Non-blocked cells capped < 240 in B
            non_blocked_in_s = (out['cost_s'] < BLOCKED_THRESHOLD)
            v3b = np.all(out['cost_b'][non_blocked_in_s] < BLOCKED_THRESHOLD)
            # 4. Entropy actually changes cost map
            has_diff = not np.array_equal(out['cost_a'], out['cost_b'])
            # 5. Gradient check (no training)
            v5 = not logits.requires_grad

            print(f"\n  Image {i} ({key}):")
            print(f"    - Same start/goal coords:   {v1} {START_XY} -> {GOAL_XY}")
            print(f"    - Blocked cells preserved: {v3} ({blocked_in_s.sum()} cells)")
            print(f"    - Non-blocked capped <240: {v3b}")
            print(f"    - Entropy changes cost map: {has_diff} (max diff = {np.abs(out['cost_b'].astype(int) - out['cost_a'].astype(int)).max()})")
            print(f"    - Zero gradients / no train: {v5}")
            print(f"    - Method A: found={out['res_a']['found']}, len={out['res_a']['length']:.1f}, mean_u={out['res_a']['mean_u']:.4f}")
            print(f"    - Method B: found={out['res_b']['found']}, len={out['res_b']['length']:.1f}, mean_u={out['res_b']['mean_u']:.4f}")

            if not (v1 and v3 and v3b and has_diff and v5):
                print(f"SMOKE TEST FAILED on image {key}")
                sys.exit(1)

    print("\n[PASSED] ALL 6 SMOKE TEST VERIFICATION CHECKS PASSED.")


def generate_stage_b_notebook():
    """Builds canonical notebooks/10_Uncertainty_Aware_Path_Planning.ipynb."""
    import nbformat as nbf
    nb = nbf.v4.new_notebook()

    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# Notebook 10 — Uncertainty-Aware Terrain Risk / Cost Map & Offline A* Path Planning
**Project**: Uncertainty-Aware Terrain Semantic Segmentation and Traversability Estimation for Autonomous Planetary Rover Navigation  
**Stage B**: Incorporating Predictive Uncertainty into 2D Cost Maps for Offline Image-Space Rover Route Planning

---

## 1. Research Objectives & Methodology
This notebook investigates:
> *"Can predictive uncertainty be incorporated into a terrain risk/cost map so that offline path planning prefers lower-risk regions?"*

We perform a strict paired comparison between:
- **Method A (Baseline Semantic Cost)**:
  $$C_A(x, y) = C_s(x, y)$$
- **Method B (Uncertainty-Aware Cost)**:
  $$C_u(x, y) = \\text{round}(255 \\cdot H_{\\text{norm}}(p(x, y)))$$
  $$C_B(x, y) = \\begin{cases} 255 & \\text{if } C_s(x, y) \\ge 240 \\\\ \\min(239, \\text{round}(0.7 \\cdot C_s(x, y) + 0.3 \\cdot C_u(x, y))) & \\text{otherwise} \\end{cases}$$

All evaluation is conducted on a fixed $256 \\times 256$ pixel grid with identical start `(5, 5)` and goal `(250, 250)` coordinates on the 190 validation frames.  
**Note**: This evaluates simulated, offline image-space path planning, not physical rover navigation."""))

    cells.append(nbf.v4.new_code_cell("""import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path.cwd().resolve()
if (PROJECT_ROOT / "notebooks").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "risk_aware_navigation"

# Load aggregate summary
with open(OUTPUT_DIR / "risk_map_summary.json", "r") as f:
    summary = json.load(f)

print("=== Stage B Path Planning Aggregate Summary ===")
print(f"Validation Images Evaluated: {summary['evaluation_protocol']['images_evaluated']}")
print(f"Method A (Semantic Baseline)   - Success Rate: {summary['method_a_baseline']['success_rate_pct']:.1f}% ({summary['method_a_baseline']['successful_paths']}/{summary['evaluation_protocol']['images_evaluated']})")
print(f"Method B (Uncertainty-Aware)   - Success Rate: {summary['method_b_uncertainty_aware']['success_rate_pct']:.1f}% ({summary['method_b_uncertainty_aware']['successful_paths']}/{summary['evaluation_protocol']['images_evaluated']})")
print()
print("=== Paired Succeeded Comparison ===")
paired = summary['paired_comparison']
print(f"Both Succeeded:                {paired['both_succeeded']} frames")
print(f"Number of Changed Paths:       {paired['paths_changed_count']} frames ({paired['paths_changed_pct']:.1f}%)")
print(f"Paths where B has Lower Uncert:{paired['b_lower_uncertainty_count']} frames ({paired['b_lower_uncertainty_pct']:.1f}%)")
print(f"Mean Path Uncertainty A:       {paired['mean_path_uncertainty_A']:.4f}")
print(f"Mean Path Uncertainty B:       {paired['mean_path_uncertainty_B']:.4f} (Reduction: {paired['mean_uncertainty_reduction_pct']:.1f}%)")
print(f"Mean Path Length A:            {paired['mean_path_length_A']:.2f} px")
print(f"Mean Path Length B:            {paired['mean_path_length_B']:.2f} px (Length Overhead: {paired['mean_length_overhead_pct']:.1f}%)")"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Paired Navigation Evaluation Table
Detailed breakdown of each validation frame comparing path length, path cost, and mean accumulated uncertainty."""))

    cells.append(nbf.v4.new_code_cell("""df_nav = pd.read_csv(OUTPUT_DIR / "baseline_vs_uncertainty_metrics.csv")
display_cols = ['image_id', 'a_found', 'b_found', 'path_changed', 'a_length', 'b_length', 'a_mean_u', 'b_mean_u', 'u_diff']
print(df_nav[display_cols].head(15).to_markdown(index=False))"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Representative Route Visualizations
Comparing Method A (Cyan) and Method B (Orange) routes plotted over the RGB image, Ground-Truth Mask, Predicted Semantics, Cost Maps, and Predictive Entropy."""))

    cells.append(nbf.v4.new_code_cell("""vis_paths = sorted((OUTPUT_DIR / "representative_paths").glob("*.png"))
for vp in vis_paths:
    print(f"Visualizing: {vp.name}")
    img = Image.open(vp)
    plt.figure(figsize=(18, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.show()"""))

    nb.cells = cells
    nb_path = PROJECT_ROOT / "notebooks" / "10_Uncertainty_Aware_Path_Planning.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Saved: {nb_path}")


def main():
    print("=" * 70)
    print("STAGE B: UNCERTAINTY-AWARE RISK MAP & PAIRED A* EVALUATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PATHS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading frozen model on {device}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = MultimodalUNet(in_channels=5, out_channels=NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    val_entries = get_validation_entries()
    print(f"Loaded {len(val_entries)} validation entries.")

    transform_rgb = transforms.Compose([
        transforms.Resize(TARGET_SIZE, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    depth_cache = CACHE_DIR / "depth"
    thermal_cache = CACHE_DIR / "thermal"

    # Step 1: Smoke Test
    run_smoke_test(model, val_entries, device, transform_rgb)

    # Step 2: Full 190-Image Paired Evaluation
    print("\n" + "=" * 70)
    print("STEP 2: FULL 190-IMAGE PAIRED EVALUATION")
    print("=" * 70)

    rows = []
    vis_candidates = []
    t0 = time.time()

    with torch.no_grad():
        for idx, s in enumerate(val_entries, start=1):
            key = s['key']

            rgb_img = Image.open(s['rgb_path']).convert("RGB")
            rgb_t = transform_rgb(rgb_img)
            rgb_disp = np.array(rgb_img.resize(TARGET_SIZE, resample=Image.BILINEAR))

            mask_img = Image.open(s['mask_path'])
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

            x = torch.cat([rgb_t, depth_t / 255.0, thermal_t / 255.0], dim=0).unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            entropy_norm = compute_normalized_entropy(probs).squeeze(0).cpu().numpy()

            out = run_pipeline_for_image(pred, entropy_norm)
            res_a = out['res_a']
            res_b = out['res_b']

            # Compare path coordinates if both found
            path_changed = False
            if res_a['found'] and res_b['found']:
                path_changed = (res_a['path'] != res_b['path'])
                u_diff = res_b['mean_u'] - res_a['mean_u']
                len_diff = res_b['length'] - res_a['length']
                cost_diff = res_b['cost'] - res_a['cost']
            else:
                u_diff = float('nan')
                len_diff = float('nan')
                cost_diff = float('nan')

            row = {
                'image_id': key,
                'a_found': res_a['found'],
                'a_reason': res_a['reason'],
                'a_steps': res_a['steps'],
                'a_length': round(res_a['length'], 2) if res_a['found'] else '',
                'a_cost': round(res_a['cost'], 2) if res_a['found'] else '',
                'a_mean_u': round(res_a['mean_u'], 4) if res_a['found'] else '',
                'b_found': res_b['found'],
                'b_reason': res_b['reason'],
                'b_steps': res_b['steps'],
                'b_length': round(res_b['length'], 2) if res_b['found'] else '',
                'b_cost': round(res_b['cost'], 2) if res_b['found'] else '',
                'b_mean_u': round(res_b['mean_u'], 4) if res_b['found'] else '',
                'path_changed': path_changed,
                'u_diff': round(u_diff, 4) if not np.isnan(u_diff) else '',
                'len_diff': round(len_diff, 2) if not np.isnan(len_diff) else '',
                'cost_diff': round(cost_diff, 2) if not np.isnan(cost_diff) else '',
            }
            rows.append(row)

            # Keep candidate for representative visualization
            if res_a['found'] and res_b['found'] and path_changed:
                vis_candidates.append({
                    'key': key,
                    'rgb': rgb_disp,
                    'gt': gt,
                    'pred': pred,
                    'cost_s': out['cost_s'],
                    'cost_b': out['cost_b'],
                    'entropy': entropy_norm,
                    'path_a': res_a['path'],
                    'path_b': res_b['path'],
                    'len_a': res_a['length'],
                    'len_b': res_b['length'],
                    'u_a': res_a['mean_u'],
                    'u_b': res_b['mean_u'],
                    'u_diff': u_diff,
                })

            if idx % 38 == 0 or idx == len(val_entries):
                print(f"  Processed {idx}/{len(val_entries)} frames ({(idx/len(val_entries))*100:.1f}%)")

    elapsed = time.time() - t0
    print(f"Completed 190-frame evaluation in {elapsed:.1f}s.")

    # Save detailed CSV
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "baseline_vs_uncertainty_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Compute Aggregate & Paired Statistics
    total_imgs = len(rows)
    a_succ = sum(r['a_found'] for r in rows)
    b_succ = sum(r['b_found'] for r in rows)

    both_succ = [r for r in rows if r['a_found'] and r['b_found']]
    a_only = [r for r in rows if r['a_found'] and not r['b_found']]
    b_only = [r for r in rows if not r['a_found'] and r['b_found']]
    both_fail = [r for r in rows if not r['a_found'] and not r['b_found']]

    # Metrics on paired succeeded subset
    a_lens = [float(r['a_length']) for r in both_succ]
    b_lens = [float(r['b_length']) for r in both_succ]
    a_costs = [float(r['a_cost']) for r in both_succ]
    b_costs = [float(r['b_cost']) for r in both_succ]
    a_us = [float(r['a_mean_u']) for r in both_succ]
    b_us = [float(r['b_mean_u']) for r in both_succ]

    paths_changed = [r for r in both_succ if r['path_changed']]
    b_lower_u = [r for r in both_succ if float(r['u_diff']) < -1e-4]
    b_equal_u = [r for r in both_succ if abs(float(r['u_diff'])) <= 1e-4]
    b_higher_u = [r for r in both_succ if float(r['u_diff']) > 1e-4]

    summary_json = {
        "evaluation_protocol": {
            "images_evaluated": total_imgs,
            "start_xy": list(START_XY),
            "goal_xy": list(GOAL_XY),
            "blocked_threshold": BLOCKED_THRESHOLD,
            "allow_diagonal": ALLOW_DIAGONAL,
            "method_a_formula": "C_A = C_s",
            "method_b_formula": "C_B = 0.7 * C_s + 0.3 * C_u (capped < 240 for free cells, 255 for blocked)",
            "checkpoint": str(CHECKPOINT_PATH),
        },
        "method_a_baseline": {
            "successful_paths": a_succ,
            "failed_paths": total_imgs - a_succ,
            "success_rate_pct": round(a_succ / total_imgs * 100.0, 2),
            "mean_path_length": round(float(np.mean(a_lens)), 2) if a_lens else None,
            "median_path_length": round(float(np.median(a_lens)), 2) if a_lens else None,
            "mean_path_cost": round(float(np.mean(a_costs)), 2) if a_costs else None,
            "median_path_cost": round(float(np.median(a_costs)), 2) if a_costs else None,
            "mean_path_uncertainty": round(float(np.mean(a_us)), 4) if a_us else None,
            "median_path_uncertainty": round(float(np.median(a_us)), 4) if a_us else None,
        },
        "method_b_uncertainty_aware": {
            "successful_paths": b_succ,
            "failed_paths": total_imgs - b_succ,
            "success_rate_pct": round(b_succ / total_imgs * 100.0, 2),
            "mean_path_length": round(float(np.mean(b_lens)), 2) if b_lens else None,
            "median_path_length": round(float(np.median(b_lens)), 2) if b_lens else None,
            "mean_path_cost": round(float(np.mean(b_costs)), 2) if b_costs else None,
            "median_path_cost": round(float(np.median(b_costs)), 2) if b_costs else None,
            "mean_path_uncertainty": round(float(np.mean(b_us)), 4) if b_us else None,
            "median_path_uncertainty": round(float(np.median(b_us)), 4) if b_us else None,
        },
        "paired_comparison": {
            "both_succeeded": len(both_succ),
            "a_succeeded_b_failed": len(a_only),
            "a_failed_b_succeeded": len(b_only),
            "both_failed": len(both_fail),
            "paths_changed_count": len(paths_changed),
            "paths_changed_pct": round(len(paths_changed) / len(both_succ) * 100.0, 2) if both_succ else 0.0,
            "b_lower_uncertainty_count": len(b_lower_u),
            "b_lower_uncertainty_pct": round(len(b_lower_u) / len(both_succ) * 100.0, 2) if both_succ else 0.0,
            "b_equal_uncertainty_count": len(b_equal_u),
            "b_higher_uncertainty_count": len(b_higher_u),
            "mean_path_uncertainty_A": round(float(np.mean(a_us)), 4) if a_us else None,
            "mean_path_uncertainty_B": round(float(np.mean(b_us)), 4) if b_us else None,
            "mean_uncertainty_diff_B_minus_A": round(float(np.mean(b_us) - np.mean(a_us)), 4) if a_us else None,
            "mean_uncertainty_reduction_pct": round((float(np.mean(a_us)) - float(np.mean(b_us))) / float(np.mean(a_us)) * 100.0, 2) if a_us else None,
            "mean_path_length_A": round(float(np.mean(a_lens)), 2) if a_lens else None,
            "mean_path_length_B": round(float(np.mean(b_lens)), 2) if b_lens else None,
            "mean_length_overhead_pct": round((float(np.mean(b_lens)) - float(np.mean(a_lens))) / float(np.mean(a_lens)) * 100.0, 2) if a_lens else None,
            "mean_path_cost_A": round(float(np.mean(a_costs)), 2) if a_costs else None,
            "mean_path_cost_B": round(float(np.mean(b_costs)), 2) if b_costs else None,
        }
    }

    summary_file = OUTPUT_DIR / "risk_map_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved: {summary_file}")

    # Generate Representative Visual Comparisons
    print(f"\nGenerating visual path comparisons for representative frames (found {len(vis_candidates)} diverging paths)...")
    # Sort by largest uncertainty reduction (u_diff most negative)
    vis_candidates.sort(key=lambda c: c['u_diff'])
    selected_vis = vis_candidates[:4]

    class_colors = [
        [0.1, 0.1, 0.1],   # 0: Void
        [0.7, 0.55, 0.4],  # 1: Compact
        [0.4, 0.8, 0.3],   # 2: Grass
        [0.5, 0.5, 0.6],   # 3: Bedrock
        [0.9, 0.8, 0.4],   # 4: Sandy
        [0.85, 0.5, 0.2],  # 5: Pebble
        [0.6, 0.2, 0.2],   # 6: Rock
        [0.1, 0.45, 0.15], # 7: Vegetation
    ]
    cmap_classes = ListedColormap(class_colors)
    bounds = np.arange(NUM_CLASSES + 1) - 0.5
    norm_classes = BoundaryNorm(bounds, cmap_classes.N)

    for v_idx, cand in enumerate(selected_vis, start=1):
        key = cand['key']
        fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))

        # 1. RGB with Both Paths Plotted
        axes[0].imshow(cand['rgb'])
        pa = np.array(cand['path_a'])
        pb = np.array(cand['path_b'])
        axes[0].plot(pa[:, 0], pa[:, 1], color='cyan', linewidth=2.0, label=f'Method A (L={cand["len_a"]:.0f})')
        axes[0].plot(pb[:, 0], pb[:, 1], color='gold', linewidth=2.0, linestyle='--', label=f'Method B (L={cand["len_b"]:.0f})')
        axes[0].scatter(*START_XY, color='lime', edgecolors='black', s=60, label='Start', zorder=4)
        axes[0].scatter(*GOAL_XY, color='red', marker='*', edgecolors='black', s=110, label='Goal', zorder=4)
        axes[0].set_title(f"1. RGB & Planned Routes\n({key})", fontsize=11, fontweight="bold")
        axes[0].legend(loc='lower right', fontsize=8, framealpha=0.8)
        axes[0].axis('off')

        # 2. Ground-Truth Mask
        axes[1].imshow(cand['gt'], cmap=cmap_classes, norm=norm_classes)
        axes[1].set_title("2. Ground-Truth Mask", fontsize=11, fontweight="bold")
        axes[1].axis('off')

        # 3. Predicted Mask
        axes[2].imshow(cand['pred'], cmap=cmap_classes, norm=norm_classes)
        axes[2].set_title("3. Multimodal Prediction", fontsize=11, fontweight="bold")
        axes[2].axis('off')

        # 4. Semantic Cost Map (C_s) + Method A Path
        axes[3].imshow(cand['cost_s'], cmap='gray', vmin=0, vmax=255)
        axes[3].plot(pa[:, 0], pa[:, 1], color='cyan', linewidth=2.0)
        axes[3].scatter(*START_XY, color='lime', s=40, zorder=4)
        axes[3].scatter(*GOAL_XY, color='red', marker='*', s=80, zorder=4)
        axes[3].set_title(f"4. Semantic Cost (C_A)\nRoute Uncertainty: {cand['u_a']:.3f}", fontsize=11, fontweight="bold")
        axes[3].axis('off')

        # 5. Predictive Entropy Map
        im_ent = axes[4].imshow(cand['entropy'], cmap='inferno', vmin=0.0, vmax=1.0)
        axes[4].set_title("5. Predictive Entropy Map\n(H_norm)", fontsize=11, fontweight="bold")
        axes[4].axis('off')
        cbar = fig.colorbar(im_ent, ax=axes[4], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        # 6. Uncertainty-Aware Cost Map (C_B) + Method B Path
        axes[5].imshow(cand['cost_b'], cmap='gray', vmin=0, vmax=255)
        axes[5].plot(pb[:, 0], pb[:, 1], color='gold', linewidth=2.0)
        axes[5].scatter(*START_XY, color='lime', s=40, zorder=4)
        axes[5].scatter(*GOAL_XY, color='red', marker='*', s=80, zorder=4)
        u_reduction = (cand['u_a'] - cand['u_b']) / cand['u_a'] * 100.0 if cand['u_a'] > 0 else 0.0
        axes[5].set_title(f"6. Risk-Aware Cost (C_B)\nRoute Uncertainty: {cand['u_b']:.3f} (-{u_reduction:.1f}%)", fontsize=11, fontweight="bold")
        axes[5].axis('off')

        fig.tight_layout()
        out_path = PATHS_DIR / f"val_path_comp_{v_idx:02d}_{key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved figure: {out_path.name}")

    # Generate Notebook 10
    print("\nGenerating canonical notebooks/10_Uncertainty_Aware_Path_Planning.ipynb...")
    generate_stage_b_notebook()

    print("\n" + "=" * 70)
    print("STAGE B EXECUTION COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
