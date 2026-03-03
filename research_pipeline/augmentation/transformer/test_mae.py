"""Minimal test to see if train_mae works"""
import sys
from pathlib import Path
import os

print("Starting minimal MAE test", flush=True)

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_dir))

print("Imports path set", flush=True)

from augmentation.transformer.train_mae import train_mae

print("train_mae imported successfully", flush=True)

# Test parameters
base_dir_path = base_dir
mae_weights_dir = base_dir_path / "results" / "augmentation" / "transformer" / "mae_weights"
mae_weights_dir.mkdir(parents=True, exist_ok=True)

print(f"Calling train_mae with base_dir={base_dir_path}, save_dir={mae_weights_dir}", flush=True)

train_mae(
    base_dir=base_dir_path,
    save_dir=str(mae_weights_dir),
    epochs=2,  # Just 2 epochs for testing
    batch_size=32,
    random_seed=42,
)

print("train_mae completed successfully", flush=True)
