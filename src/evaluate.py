"""
Evaluation Module for Emotion Classification.

Provides comprehensive evaluation metrics and visualization.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)


LABEL_NAMES = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']


def calculate_metrics(y_true: List[int], y_pred: List[int], 
                      label_names: List[str] = None) -> Dict:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Names for labels
        
    Returns:
        Dictionary with all metrics
    """
    if label_names is None:
        label_names = LABEL_NAMES
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics dictionary
    per_class = {}
    for i, name in enumerate(label_names):
        if i < len(precision):
            per_class[name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1-score': float(f1[i]),
                'support': int(support[i])
            }
    
    return {
        'accuracy': float(accuracy),
        'macro_avg': {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1-score': float(f1_macro)
        },
        'weighted_avg': {
            'precision': float(precision_weighted),
            'recall': float(recall_weighted),
            'f1-score': float(f1_weighted)
        },
        'per_class': per_class,
        'confusion_matrix': cm.tolist()
    }


def plot_confusion_matrix(y_true: List[int], y_pred: List[int],
                          label_names: List[str] = None,
                          normalize: bool = True,
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Names for labels
        normalize: Whether to normalize
        figsize: Figure size
        save_path: Path to save figure
    """
    if label_names is None:
        label_names = LABEL_NAMES
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_model_comparison(results: Dict[str, Dict],
                          metric: str = 'f1-score',
                          figsize: Tuple[int, int] = (12, 6),
                          save_path: str = None):
    """
    Plot comparison of multiple models.
    
    Args:
        results: Dictionary with model results
        metric: Metric to compare
        figsize: Figure size
        save_path: Path to save figure
    """
    models = []
    macro_scores = []
    weighted_scores = []
    
    for model_name, result in results.items():
        if 'error' not in result:
            models.append(model_name)
            if 'test_metrics' in result:
                metrics = result['test_metrics']
            else:
                metrics = result
            
            if 'classification_report' in metrics:
                macro_scores.append(metrics['classification_report']['macro avg'][metric])
                weighted_scores.append(metrics['classification_report']['weighted avg'][metric])
            elif 'macro_avg' in metrics:
                macro_scores.append(metrics['macro_avg'][metric])
                weighted_scores.append(metrics['weighted_avg'][metric])
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width/2, macro_scores, width, label='Macro Avg', color='steelblue')
    bars2 = ax.bar(x + width/2, weighted_scores, width, label='Weighted Avg', color='darkorange')
    
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.replace('-', ' ').title())
    ax.set_title(f'Model Comparison - {metric.replace("-", " ").title()}')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_per_class_metrics(results: Dict[str, Dict],
                           figsize: Tuple[int, int] = (14, 8),
                           save_path: str = None):
    """
    Plot per-class F1 scores for all models.
    
    Args:
        results: Dictionary with model results
        figsize: Figure size
        save_path: Path to save figure
    """
    # Collect data
    data = []
    for model_name, result in results.items():
        if 'error' not in result:
            if 'test_metrics' in result:
                metrics = result['test_metrics']
            else:
                metrics = result
            
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                for label in LABEL_NAMES:
                    if label in report:
                        data.append({
                            'Model': model_name,
                            'Label': label,
                            'F1-Score': report[label]['f1-score']
                        })
            elif 'per_class' in metrics:
                for label, values in metrics['per_class'].items():
                    data.append({
                        'Model': model_name,
                        'Label': label,
                        'F1-Score': values['f1-score']
                    })
    
    if not data:
        print("No data to plot")
        return
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=df, x='Label', y='F1-Score', hue='Model')
    plt.title('Per-Class F1 Scores by Model')
    plt.xlabel('Emotion Class')
    plt.ylabel('F1-Score')
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def error_analysis(y_true: List[int], y_pred: List[int], 
                   texts: List[str] = None,
                   label_names: List[str] = None,
                   top_n: int = 20) -> pd.DataFrame:
    """
    Analyze misclassified samples.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        texts: Original texts
        label_names: Names for labels
        top_n: Number of samples to return
        
    Returns:
        DataFrame with misclassified samples
    """
    if label_names is None:
        label_names = LABEL_NAMES
    
    # Find misclassified
    errors = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            error = {
                'index': i,
                'true_label': true,
                'pred_label': pred,
                'true_name': label_names[true] if true < len(label_names) else str(true),
                'pred_name': label_names[pred] if pred < len(label_names) else str(pred),
                'error_distance': abs(true - pred)
            }
            if texts and i < len(texts):
                error['text'] = texts[i][:200]
            errors.append(error)
    
    df = pd.DataFrame(errors)
    
    if len(df) > 0:
        # Sort by error distance (larger mistakes first)
        df = df.sort_values('error_distance', ascending=False)
    
    return df.head(top_n)


def generate_latex_table(results: Dict[str, Dict], 
                         caption: str = "Model Comparison Results",
                         label: str = "tab:results") -> str:
    """
    Generate LaTeX table for paper.
    
    Args:
        results: Dictionary with model results
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Model & Accuracy & Precision & Recall & F1-macro & F1-weighted \\\\",
        "\\midrule"
    ]
    
    for model_name, result in results.items():
        if 'error' not in result:
            if 'test_metrics' in result:
                metrics = result['test_metrics']
            else:
                metrics = result
            
            acc = metrics.get('accuracy', 0)
            
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                prec = report['macro avg']['precision']
                rec = report['macro avg']['recall']
                f1_macro = report['macro avg']['f1-score']
                f1_weighted = report['weighted avg']['f1-score']
            else:
                prec = metrics.get('macro_avg', {}).get('precision', 0)
                rec = metrics.get('macro_avg', {}).get('recall', 0)
                f1_macro = metrics.get('macro_avg', {}).get('f1-score', 0)
                f1_weighted = metrics.get('weighted_avg', {}).get('f1-score', 0)
            
            # Format model name
            display_name = model_name.replace('_', ' ').title()
            
            lines.append(
                f"{display_name} & {acc:.4f} & {prec:.4f} & {rec:.4f} & {f1_macro:.4f} & {f1_weighted:.4f} \\\\"
            )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return '\n'.join(lines)


def main():
    """Test evaluation functions."""
    # Mock data
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.1, 0.1, 0.2, 0.3, 0.3])
    y_pred = y_true.copy()
    
    # Add some errors
    error_indices = np.random.choice(n_samples, size=20, replace=False)
    y_pred[error_indices] = np.random.choice([0, 1, 2, 3, 4], size=20)
    
    texts = [f"Sample review text {i}" for i in range(n_samples)]
    
    print("=" * 60)
    print("Evaluation Test")
    print("=" * 60)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true.tolist(), y_pred.tolist())
    
    print("\n=== Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_avg']['f1-score']:.4f}")
    print(f"Weighted F1: {metrics['weighted_avg']['f1-score']:.4f}")
    
    print("\n=== Per-Class Metrics ===")
    for label, values in metrics['per_class'].items():
        print(f"  {label}: P={values['precision']:.3f}, R={values['recall']:.3f}, F1={values['f1-score']:.3f}")
    
    # Error analysis
    print("\n=== Error Analysis ===")
    errors = error_analysis(y_true.tolist(), y_pred.tolist(), texts)
    print(f"Total errors: {len(errors)}")
    if len(errors) > 0:
        print(errors[['true_name', 'pred_name', 'error_distance']].head(5))
    
    # Generate LaTeX
    print("\n=== LaTeX Table ===")
    mock_results = {
        'naive_bayes': metrics,
        'svm': metrics,
        'random_forest': metrics
    }
    latex = generate_latex_table(mock_results)
    print(latex)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
