"""
Main Baseline Pipeline Runner
Runs complete baseline experiment: feature extraction -> training -> evaluation
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from extract_baseline_features import main as extract_features
from train_baseline_model import train_baseline_model


def run_baseline_pipeline(data_dir='../../Dataset/my Dataset',
                         epochs=100, batch_size=32, random_seed=42):
    """
    Run complete baseline pipeline
    
    Args:
        data_dir: Path to dataset directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        random_seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("BASELINE PIPELINE: HINDI SPEECH EMOTION RECOGNITION")
    print("=" * 70)
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_output_dir = base_dir / 'data'
    results_dir = base_dir / 'results' / 'baseline'
    
    # Step 1: Feature Extraction
    print("\n" + "=" * 70)
    print("STEP 1: FEATURE EXTRACTION")
    print("=" * 70)
    
    # Change to baseline directory for feature extraction
    original_dir = os.getcwd()
    os.chdir(Path(__file__).parent)
    
    try:
        feature_path, analysis = extract_features()
        feature_path = Path(feature_path).resolve()
    finally:
        os.chdir(original_dir)
    
    print(f"\n[OK] Feature extraction complete")
    print(f"  Features saved to: {feature_path}")
    print(f"  Total samples: {analysis['total_samples']}")
    print(f"  Emotion classes: {analysis['num_classes']}")
    
    # Step 2: Model Training
    print("\n" + "=" * 70)
    print("STEP 2: MODEL TRAINING")
    print("=" * 70)
    
    results, model = train_baseline_model(
        data_path=str(feature_path),
        results_dir=str(results_dir),
        epochs=epochs,
        batch_size=batch_size,
        random_seed=random_seed
    )
    
    print(f"\n[OK] Model training complete")
    print(f"  Test Accuracy: {results['metrics']['overall']['accuracy']:.4f}")
    print(f"  Macro F1-Score: {results['metrics']['overall']['macro_f1']:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("BASELINE PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults Location: {results_dir}")
    print(f"\nGenerated Files:")
    print(f"  1. Dataset analysis: {results_dir}/dataset_analysis.json")
    print(f"  2. Baseline results: {results_dir}/baseline_results.json")
    print(f"  3. Trained model: {results_dir}/best_baseline_model.h5")
    print(f"  4. Visualizations:")
    print(f"     - Dataset distribution: {results_dir}/dataset_distribution.png")
    print(f"     - Sample spectrograms: {results_dir}/sample_spectrograms.png")
    print(f"     - Training history: {results_dir}/training_history.png")
    print(f"     - Confusion matrix: {results_dir}/confusion_matrix.png")
    print(f"     - Per-class metrics: {results_dir}/per_class_metrics.png")
    
    return results, model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run complete baseline pipeline for Hindi SER'
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
