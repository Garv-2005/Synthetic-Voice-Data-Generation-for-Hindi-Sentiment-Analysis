"""
Model Comparison Framework
Trains multiple models and generates comprehensive comparison
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import CNNModel, LSTMModel, CNNLSTMModel, ResNetModel, TransformerModel, SVMModel
from comparison.train_model import train_model


def compare_all_models(data_path, results_base_dir='../results',
                       epochs=100, batch_size=32, random_seed=42, 
                       skip_existing=False, force_retrain=False):
    """
    Train and compare all model architectures
    
    Args:
        data_path: Path to .npz file with features
        results_base_dir: Base directory for results
        epochs: Number of training epochs
        batch_size: Batch size
        random_seed: Random seed for reproducibility
        skip_existing: Skip models that have already been trained
        force_retrain: Force retrain even if model exists
    """
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    
    # Check GPU status (will be shown by train_model, but show summary here)
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"\n✓ GPU Available: {len(gpus)} GPU(s) will be used for training")
        else:
            print(f"\n⚠ No GPU detected. Training will use CPU (slower).")
    except:
        pass
    
    if skip_existing and not force_retrain:
        print(f"\n📋 Resume Mode: Will skip already-trained models")
    elif force_retrain:
        print(f"\n🔄 Force Mode: Will retrain all models")
    
    # Define all models to train
    models_to_train = [
        (CNNModel, 'CNN'),
        (LSTMModel, 'LSTM'),
        (CNNLSTMModel, 'CNN-LSTM'),
        (ResNetModel, 'ResNet'),
        (TransformerModel, 'Transformer'),
        (SVMModel, 'SVM')
    ]
    
    all_results = []
    all_histories = []
    model_names = []
    
    # Train each model with progress tracking
    print(f"\nTraining {len(models_to_train)} models...")
    with tqdm(total=len(models_to_train), desc="Overall Progress", ncols=100, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', 
              position=0, leave=True) as overall_pbar:
        
        for idx, (model_class, model_name) in enumerate(models_to_train, 1):
            overall_pbar.set_description(f"Training {model_name}")
            overall_pbar.set_postfix_str(f"Model {idx}/{len(models_to_train)}")
            
            print(f"\n{'='*80}")
            print(f"Training {model_name} Model ({idx}/{len(models_to_train)})")
            print(f"{'='*80}\n")
            
            results_dir = os.path.join(results_base_dir, model_name.lower().replace('-', '_'))
            os.makedirs(results_dir, exist_ok=True)
            
            # Check if model already exists
            model_ext = '.pkl' if model_name.upper() == 'SVM' else '.h5'
            model_path = os.path.join(results_dir, f'best_{model_name.lower()}_model{model_ext}')
            results_path = os.path.join(results_dir, f'{model_name.lower()}_results.json')
            
            skip_this_model = skip_existing and not force_retrain and os.path.exists(model_path) and os.path.exists(results_path)
            
            if skip_this_model:
                print(f"  → {model_name} already trained. Loading existing results...")
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    all_results.append(results)
                    all_histories.append(results.get('training_history', {}))
                    model_names.append(model_name)
                    print(f"  ✓ Loaded {model_name} results (Accuracy: {results['metrics']['overall']['accuracy']:.4f})")
                    overall_pbar.update(1)
                    continue
                except Exception as e:
                    print(f"  ⚠ Error loading {model_name} results: {e}. Retraining...")
            
            try:
                results, model = train_model(
                    model_class=model_class,
                    data_path=data_path,
                    results_dir=results_dir,
                    model_name=model_name,
                    epochs=epochs,
                    batch_size=batch_size,
                    random_seed=random_seed,
                    skip_if_exists=False  # We handle skipping at this level
                )
                
                all_results.append(results)
                all_histories.append(results['training_history'])
                model_names.append(model_name)
                
                print(f"\n✓ {model_name} training completed successfully")
                overall_pbar.update(1)
                
            except Exception as e:
                print(f"\n✗ {model_name} training failed: {str(e)}")
                import traceback
                traceback.print_exc()
                overall_pbar.update(1)
                continue
    
    if not all_results:
        print("\nNo models were successfully trained!")
        return
    
    # Generate comparison visualizations
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    comparison_dir = os.path.join(results_base_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Lazy import visualization functions
    from utils.visualization import (
        plot_model_comparison,
        plot_training_curves_comparison,
        plot_metrics_radar_chart
    )
    
    viz_tasks = [
        ('Model comparison charts', lambda: plot_model_comparison(
            all_results,
            save_path=os.path.join(comparison_dir, 'model_comparison.png')
        )),
        ('Training curves comparison', lambda: plot_training_curves_comparison(
            all_histories,
            model_names,
            save_path=os.path.join(comparison_dir, 'training_curves_comparison.png')
        )),
        ('Metrics radar chart', lambda: plot_metrics_radar_chart(
            all_results,
            save_path=os.path.join(comparison_dir, 'metrics_radar_chart.png')
        ))
    ]
    
    with tqdm(total=len(viz_tasks), desc="Generating visualizations", ncols=100,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for task_name, task_func in viz_tasks:
            task_func()
            pbar.set_postfix_str(task_name)
            pbar.update(1)
    
    # Save comprehensive comparison results
    comparison_results = {
        'timestamp': results['timestamp'],
        'models_trained': model_names,
        'configuration': {
            'epochs': epochs,
            'batch_size': batch_size,
            'random_seed': random_seed
        },
        'results': all_results,
        'summary': {
            model_name: {
                'accuracy': r['metrics']['overall']['accuracy'],
                'macro_f1': r['metrics']['overall']['macro_f1'],
                'weighted_f1': r['metrics']['overall']['weighted_f1']
            }
            for model_name, r in zip(model_names, all_results)
        }
    }
    
    comparison_path = os.path.join(comparison_dir, 'comparison_results.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<15} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print("-" * 80)
    
    for model_name, r in zip(model_names, all_results):
        acc = r['metrics']['overall']['accuracy']
        macro_f1 = r['metrics']['overall']['macro_f1']
        weighted_f1 = r['metrics']['overall']['weighted_f1']
        print(f"{model_name:<15} {acc:<12.4f} {macro_f1:<12.4f} {weighted_f1:<12.4f}")
    
    # Find best model
    best_model_idx = np.argmax([r['metrics']['overall']['macro_f1'] for r in all_results])
    best_model_name = model_names[best_model_idx]
    best_f1 = all_results[best_model_idx]['metrics']['overall']['macro_f1']
    
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_model_name} (Macro F1: {best_f1:.4f})")
    print("=" * 80)
    
    print(f"\nComparison results saved to: {comparison_path}")
    print(f"Visualizations saved to: {comparison_dir}")
    
    return comparison_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare multiple model architectures')
    parser.add_argument('--data_path', type=str, 
                       default='../data/baseline_features.npz',
                       help='Path to feature file')
    parser.add_argument('--results_dir', type=str,
                       default='../results',
                       help='Base directory for results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip models that have already been trained')
    parser.add_argument('--force', action='store_true',
                       help='Force retrain even if model exists (overrides --skip_existing)')
    
    args = parser.parse_args()
    
    compare_all_models(
        data_path=args.data_path,
        results_base_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.seed,
        skip_existing=args.skip_existing,
        force_retrain=args.force
    )
