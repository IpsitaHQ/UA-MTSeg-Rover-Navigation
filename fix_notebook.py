import json
import shutil
import os
from pathlib import Path

# Read the notebook
with open('notebooks/01_Dataset_Explorer.ipynb', 'r') as f:
    nb = json.load(f)

# Update cell 1 (index 1) - install dependencies
nb['cells'][1]['source'] = ["%pip install -U datasets huggingface_hub"]
nb['cells'][1]['outputs'] = []
nb['cells'][1]['execution_count'] = None

# Update cell 3 (index 2) - load dataset with cache cleanup
new_cell_3 = """from datasets import load_dataset
import shutil
from pathlib import Path

# Clear corrupted cache if it exists
cache_path = Path.home() / ".cache" / "huggingface" / "datasets" / "hassanjbara___baseprod"
if cache_path.exists():
    incomplete_dir = list(cache_path.rglob("*.incomplete"))
    if incomplete_dir:
        print("Found corrupted cache, clearing it...")
        shutil.rmtree(cache_path, ignore_errors=True)
        print("Cache cleared. Re-downloading dataset...")

try:
    ds = load_dataset(
        "hassanjbara/BASEPROD",
        split="train",
        streaming=True
    )
    print(ds)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("\\nSOLUTION:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new API token (or use an existing valid one)")
    print("3. Update your HF_TOKEN environment variable with the new token")
    print("4. Restart the kernel and re-run this cell")
    ds = None"""

nb['cells'][2]['source'] = [line + '\n' for line in new_cell_3.split('\n')]
nb['cells'][2]['source'][-1] = nb['cells'][2]['source'][-1].rstrip('\n')
nb['cells'][2]['outputs'] = []
nb['cells'][2]['execution_count'] = None

# Add cell 4 - display first 10 rows (only if not already present)
if len(nb['cells']) > 3 and 'Display first 10' in ''.join(nb['cells'][3]['source']):
    print('[OK] Display cell already exists, skipping append.')
else:
    new_cell_4 = """# Display first 10 rows of the dataset
if ds is not None:
    print("First 10 samples from the dataset:\\n")
    for i, sample in enumerate(ds.take(10)):
        print(f"Sample {i+1}:")
        print(sample)
        print("-" * 80)
else:
    print("Dataset not loaded. Please fix the token issue in the previous cell.")"""

    new_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'id': 'display-10-rows',
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in new_cell_4.split('\n')]
    }
    new_cell['source'][-1] = new_cell['source'][-1].rstrip('\n')
    nb['cells'].append(new_cell)

# Write back
with open('notebooks/01_Dataset_Explorer.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("[OK] Notebook cells updated successfully!")
