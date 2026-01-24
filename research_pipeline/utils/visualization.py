"""
Visualization utilities for dataset analysis and model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os


def plot_dataset_distribution(labels, emotion_map, save_path=None):
    """Plot distribution of samples across emotion classes"""
    emotion_names = {v: k for k, v in emotion_map.items()}
    label_counts = Counter(labels)
    
    emotions = [emotion_names[i] for i in sorted(label_counts.keys())]
    counts = [label_counts[i] for i in sorted(label_counts.keys())]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(emotions, counts, color=plt.cm.Set3(np.linspace(0, 1, len(emotions))))
    
    plt.xlabel('Emotion Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.title('Dataset Distribution Across Emotion Classes', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/sum(counts)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Distribution plot saved: {save_path}")
    
    plt.close()


def plot_sample_spectrograms(features, labels, emotion_map, n_samples=None, save_path=None):
    """Plot sample spectrograms for each emotion class"""
    emotion_names = {v: k for k, v in emotion_map.items()}
    n_classes = len(emotion_map)
    
    if n_samples is None:
        n_samples = min(len(features), n_classes)
    
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (feature, label) in enumerate(zip(features[:n_samples], labels[:n_samples])):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Remove channel dimension if present
        if len(feature.shape) == 3:
            feature = feature[:, :, 0]
        
        im = ax.imshow(feature, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'{emotion_names[label]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Frame', fontsize=9)
        ax.set_ylabel('Mel Frequency', fontsize=9)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Sample Mel-Spectrograms by Emotion Class', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Sample spectrograms saved: {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """Plot training and validation accuracy/loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Loss', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Training history plot saved: {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, emotion_map, save_path=None, normalize=False):
    """Plot confusion matrix with emotion labels"""
    from sklearn.metrics import confusion_matrix
    
    emotion_names = {v: k for k, v in emotion_map.items()}
    sorted_labels = sorted(emotion_map.values())
    label_names = [emotion_names[i] for i in sorted_labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=sorted_labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                linewidths=0.5, linecolor='gray')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Confusion matrix saved: {save_path}")
    
    plt.close()
    return cm


def plot_per_class_metrics(y_true, y_pred, emotion_map, save_path=None):
    """Plot per-class precision, recall, and F1 scores"""
    from sklearn.metrics import precision_recall_fscore_support
    
    emotion_names = {v: k for k, v in emotion_map.items()}
    sorted_labels = sorted(emotion_map.values())
    label_names = [emotion_names[i] for i in sorted_labels]
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=sorted_labels, zero_division=0
    )
    
    x = np.arange(len(label_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Emotion Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Per-class metrics plot saved: {save_path}")
    
    plt.close()
    
    return precision, recall, f1, support


def plot_comprehensive_training_history(history, save_path=None, model_name="Model"):
    """Plot comprehensive training history including F1 scores"""
    from sklearn.metrics import f1_score
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy plot
    axes[0, 0].plot(history['accuracy'], label='Training Accuracy', linewidth=2, marker='o', markersize=3)
    axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'{model_name} - Accuracy', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[0, 1].plot(history['loss'], label='Training Loss', linewidth=2, marker='o', markersize=3)
    axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=3)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'{model_name} - Loss', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score plot (if available)
    if 'f1' in history:
        axes[1, 0].plot(history['f1'], label='Training F1', linewidth=2, marker='o', markersize=3)
        axes[1, 0].plot(history['val_f1'], label='Validation F1', linewidth=2, marker='s', markersize=3)
        axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        axes[1, 0].set_title(f'{model_name} - F1 Score', fontsize=13, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'F1 scores not available', 
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_title(f'{model_name} - F1 Score', fontsize=13, fontweight='bold')
    
    # Learning rate plot (if available)
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], label='Learning Rate', linewidth=2, marker='o', markersize=3)
        axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        axes[1, 1].set_title(f'{model_name} - Learning Rate', fontsize=13, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning rate not tracked', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title(f'{model_name} - Learning Rate', fontsize=13, fontweight='bold')
    
    plt.suptitle(f'{model_name} - Comprehensive Training History', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Comprehensive training history saved: {save_path}")
    
    plt.close()


def plot_model_comparison(all_results, save_path=None):
    """Plot comparison of multiple models"""
    model_names = [r['model_name'] for r in all_results]
    
    # Extract metrics
    accuracies = [r['metrics']['overall']['accuracy'] for r in all_results]
    macro_f1s = [r['metrics']['overall']['macro_f1'] for r in all_results]
    weighted_f1s = [r['metrics']['overall']['weighted_f1'] for r in all_results]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(model_names))
    width = 0.6
    
    # Accuracy comparison
    bars1 = axes[0].bar(x, accuracies, width, alpha=0.8, color='steelblue')
    axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1.1])
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Macro F1 comparison
    bars2 = axes[1].bar(x, macro_f1s, width, alpha=0.8, color='coral')
    axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Macro F1-Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Macro F1-Score Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1.1])
    
    # Add value labels
    for bar, f1 in zip(bars2, macro_f1s):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Weighted F1 comparison
    bars3 = axes[2].bar(x, weighted_f1s, width, alpha=0.8, color='mediumseagreen')
    axes[2].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Weighted F1-Score', fontsize=12, fontweight='bold')
    axes[2].set_title('Weighted F1-Score Comparison', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_ylim([0, 1.1])
    
    # Add value labels
    for bar, f1 in zip(bars3, weighted_f1s):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Model comparison plot saved: {save_path}")
    
    plt.close()


def plot_training_curves_comparison(all_histories, model_names, save_path=None):
    """Plot training curves for multiple models side by side.
    Skips models whose history lacks 'accuracy', 'val_accuracy', 'loss', 'val_loss'
    (e.g. SVM and other non-iterative models).
    """
    required = {'accuracy', 'val_accuracy', 'loss', 'val_loss'}
    curves_histories = []
    curves_names = []
    for h, n in zip(all_histories, model_names):
        if isinstance(h, dict) and required.issubset(h.keys()):
            curves_histories.append(h)
            curves_names.append(n)

    if not curves_histories:
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print("  Training curves comparison skipped: no models with iterative training history (e.g. SVM has none).")
        return

    n_models = len(curves_histories)
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))

    if n_models == 1:
        axes = axes.reshape(2, 1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for idx, (history, name, color) in enumerate(zip(curves_histories, curves_names, colors)):
        # Accuracy plot
        axes[0, idx].plot(history['accuracy'], label='Train', linewidth=2, color=color, linestyle='-')
        axes[0, idx].plot(history['val_accuracy'], label='Val', linewidth=2, color=color, linestyle='--')
        axes[0, idx].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[0, idx].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        axes[0, idx].set_title(f'{name} - Accuracy', fontsize=12, fontweight='bold')
        axes[0, idx].legend(fontsize=9)
        axes[0, idx].grid(True, alpha=0.3)

        # Loss plot
        axes[1, idx].plot(history['loss'], label='Train', linewidth=2, color=color, linestyle='-')
        axes[1, idx].plot(history['val_loss'], label='Val', linewidth=2, color=color, linestyle='--')
        axes[1, idx].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1, idx].set_ylabel('Loss', fontsize=11, fontweight='bold')
        axes[1, idx].set_title(f'{name} - Loss', fontsize=12, fontweight='bold')
        axes[1, idx].legend(fontsize=9)
        axes[1, idx].grid(True, alpha=0.3)

    plt.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Training curves comparison saved: {save_path}")

    plt.close()


def plot_metrics_radar_chart(all_results, save_path=None):
    """Create radar chart comparing multiple models"""
    model_names = [r['model_name'] for r in all_results]
    
    # Extract metrics
    metrics_data = {
        'Accuracy': [r['metrics']['overall']['accuracy'] for r in all_results],
        'Macro F1': [r['metrics']['overall']['macro_f1'] for r in all_results],
        'Weighted F1': [r['metrics']['overall']['weighted_f1'] for r in all_results],
    }
    
    # Calculate average per-class F1
    avg_f1s = []
    for r in all_results:
        per_class_f1 = [v['f1-score'] for k, v in r['metrics']['per_class'].items() 
                       if k not in ['accuracy', 'macro avg', 'weighted avg']]
        avg_f1s.append(np.mean(per_class_f1))
    metrics_data['Avg Per-Class F1'] = avg_f1s
    
    categories = list(metrics_data.keys())
    n_categories = len(categories)
    
    # Compute angle for each category
    angles = [n / float(n_categories) * 2 * np.pi for n in range(n_categories)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for idx, (name, color) in enumerate(zip(model_names, colors)):
        values = [metrics_data[cat][idx] for cat in categories]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Radar chart saved: {save_path}")
    
    plt.close()
