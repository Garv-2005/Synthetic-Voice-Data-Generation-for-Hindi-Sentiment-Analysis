"""Debug script to find where generation script hangs"""
import sys
from pathlib import Path

print("DEBUG 1: Script started")

# Add paths
base_dir = Path(__file__).resolve().parent.parent.parent
print(f"DEBUG 2: base_dir = {base_dir}")

sys.path.insert(0, str(base_dir))
print("DEBUG 3: sys.path updated")

try:
    from baseline.extract_baseline_features import main as baseline_extract_main
    print("DEBUG 4: baseline_extract_main imported")
except Exception as e:
    print(f"DEBUG 4 ERROR: {e}")

try:
    from augmentation.transformer.train_mae import train_mae
    print("DEBUG 5: train_mae imported")
except Exception as e:
    print(f"DEBUG 5 ERROR: {e}")

try:
    from augmentation.transformer.build_transformer_augmented_features import build_transformer_augmented_features
    print("DEBUG 6: build_transformer_augmented_features imported")
except Exception as e:
    print(f"DEBUG 6 ERROR: {e}")

# Get paths
capstone_root = Path(__file__).resolve().parent.parent.parent.parent  # Go up 4 levels
print(f"DEBUG 7: capstone_root = {capstone_root}")

data_dir = capstone_root / 'research_pipeline' / "data"
print(f"DEBUG 8: data_dir = {data_dir}")

baseline_npz = data_dir / "baseline_features.npz"
print(f"DEBUG 9: baseline_npz exists = {baseline_npz.exists()}")

print("\nDEBUG: All imports and path setup successful!")
