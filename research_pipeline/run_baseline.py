"""
Run Baseline Pipeline from Research Pipeline Root
Can be executed from any directory
"""

import os
import sys
from pathlib import Path

# Get the script directory
script_dir = Path(__file__).parent
baseline_dir = script_dir / 'baseline'

# Change to baseline directory and run
os.chdir(baseline_dir)
sys.path.insert(0, str(baseline_dir))

# Import and run
from run_baseline_pipeline import run_baseline_pipeline

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run baseline pipeline for Hindi SER'
    )
    parser.add_argument('--data_dir', type=str,
                       default='../../Dataset/my Dataset',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    run_baseline_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.seed
    )
