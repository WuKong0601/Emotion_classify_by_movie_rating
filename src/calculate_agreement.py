"""
Inter-Annotator Agreement Calculator.

Calculates Cohen's Kappa, Fleiss' Kappa, and other agreement metrics
for emotion annotation task.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict
from pathlib import Path


def cohens_kappa(labels1: List[int], labels2: List[int]) -> float:
    """
    Calculate Cohen's Kappa for two annotators.
    
    Args:
        labels1: Labels from annotator 1
        labels2: Labels from annotator 2
        
    Returns:
        Cohen's Kappa coefficient
    """
    assert len(labels1) == len(labels2), "Label lists must have same length"
    
    n = len(labels1)
    if n == 0:
        return 0.0
    
    # Get all unique labels
    all_labels = list(set(labels1) | set(labels2))
    k = len(all_labels)
    
    # Build confusion matrix
    confusion = np.zeros((k, k))
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    for l1, l2 in zip(labels1, labels2):
        i, j = label_to_idx[l1], label_to_idx[l2]
        confusion[i, j] += 1
    
    # Observed agreement (p_o)
    p_o = np.trace(confusion) / n
    
    # Expected agreement (p_e)
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    p_e = np.sum(row_sums * col_sums) / (n * n)
    
    # Cohen's Kappa
    if p_e == 1.0:
        return 1.0
    kappa = (p_o - p_e) / (1 - p_e)
    
    return kappa


def percent_agreement(labels1: List[int], labels2: List[int]) -> float:
    """
    Calculate simple percent agreement.
    
    Args:
        labels1: Labels from annotator 1
        labels2: Labels from annotator 2
        
    Returns:
        Percent agreement (0-1)
    """
    assert len(labels1) == len(labels2), "Label lists must have same length"
    
    if len(labels1) == 0:
        return 0.0
    
    agreements = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2)
    return agreements / len(labels1)


def confusion_matrix_analysis(labels1: List[int], labels2: List[int], 
                               label_names: List[str] = None) -> pd.DataFrame:
    """
    Build confusion matrix between two annotators.
    
    Args:
        labels1: Labels from annotator 1
        labels2: Labels from annotator 2
        label_names: Optional names for labels
        
    Returns:
        Confusion matrix as DataFrame
    """
    all_labels = sorted(list(set(labels1) | set(labels2)))
    k = len(all_labels)
    
    if label_names is None:
        label_names = [str(l) for l in all_labels]
    
    # Build matrix
    matrix = np.zeros((k, k), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    for l1, l2 in zip(labels1, labels2):
        i, j = label_to_idx[l1], label_to_idx[l2]
        matrix[i, j] += 1
    
    df = pd.DataFrame(matrix, index=label_names, columns=label_names)
    df.index.name = 'Annotator 1'
    df.columns.name = 'Annotator 2'
    
    return df


def find_disagreements(df: pd.DataFrame, col1: str = 'label_A', 
                        col2: str = 'label_B') -> pd.DataFrame:
    """
    Find samples where annotators disagree.
    
    Args:
        df: DataFrame with annotations
        col1: Column name for annotator 1
        col2: Column name for annotator 2
        
    Returns:
        DataFrame with disagreement samples
    """
    disagreements = df[df[col1] != df[col2]].copy()
    disagreements['difference'] = abs(disagreements[col1] - disagreements[col2])
    return disagreements.sort_values('difference', ascending=False)


def interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's Kappa value."""
    if kappa < 0:
        return "Poor (worse than chance)"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def calculate_agreement_report(labels1: List[int], labels2: List[int],
                                label_names: List[str] = None) -> Dict:
    """
    Generate comprehensive agreement report.
    
    Args:
        labels1: Labels from annotator 1
        labels2: Labels from annotator 2
        label_names: Optional names for labels
        
    Returns:
        Dictionary with all metrics
    """
    kappa = cohens_kappa(labels1, labels2)
    pct_agree = percent_agreement(labels1, labels2)
    
    # Per-class agreement
    all_labels = sorted(list(set(labels1) | set(labels2)))
    per_class = {}
    
    for label in all_labels:
        mask1 = [l == label for l in labels1]
        mask2 = [l == label for l in labels2]
        
        # Both agree on this label
        both_agree = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2 == label)
        either_has = sum(1 for m1, m2 in zip(mask1, mask2) if m1 or m2)
        
        if either_has > 0:
            per_class[label] = both_agree / either_has
        else:
            per_class[label] = 0.0
    
    return {
        'cohens_kappa': kappa,
        'kappa_interpretation': interpret_kappa(kappa),
        'percent_agreement': pct_agree,
        'n_samples': len(labels1),
        'n_disagreements': sum(1 for l1, l2 in zip(labels1, labels2) if l1 != l2),
        'per_class_agreement': per_class
    }


def main():
    """Main function to test with mock data."""
    # Mock data for testing
    # In real usage, load from annotation files
    
    print("=" * 60)
    print("Inter-Annotator Agreement Calculator")
    print("=" * 60)
    
    # Simulate two annotators
    np.random.seed(42)
    n_samples = 100
    
    # Annotator 1 labels
    labels1 = np.random.choice([0, 1, 2, 3, 4], size=n_samples, 
                                p=[0.1, 0.15, 0.15, 0.3, 0.3])
    
    # Annotator 2: agrees 80% of time, otherwise random
    labels2 = []
    for l in labels1:
        if np.random.random() < 0.8:
            labels2.append(l)
        else:
            labels2.append(np.random.choice([0, 1, 2, 3, 4]))
    
    labels1 = list(labels1)
    labels2 = list(labels2)
    
    label_names = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    
    # Calculate metrics
    report = calculate_agreement_report(labels1, labels2, label_names)
    
    print(f"\n=== Agreement Metrics ===")
    print(f"Cohen's Kappa: {report['cohens_kappa']:.4f} ({report['kappa_interpretation']})")
    print(f"Percent Agreement: {report['percent_agreement']*100:.1f}%")
    print(f"Total Samples: {report['n_samples']}")
    print(f"Disagreements: {report['n_disagreements']}")
    
    print(f"\n=== Per-Class Agreement ===")
    for label, name in enumerate(label_names):
        if label in report['per_class_agreement']:
            pct = report['per_class_agreement'][label] * 100
            print(f"  {name}: {pct:.1f}%")
    
    print(f"\n=== Confusion Matrix ===")
    cm = confusion_matrix_analysis(labels1, labels2, label_names)
    print(cm)
    
    print("\n" + "=" * 60)


def load_and_calculate(annotator_a_path: str, annotator_b_path: str,
                       id_col: str = 'id', label_col: str = 'label'):
    """
    Load annotation files and calculate agreement.
    
    Args:
        annotator_a_path: Path to annotator A's CSV file
        annotator_b_path: Path to annotator B's CSV file
        id_col: Column name for sample ID
        label_col: Column name for label
    """
    # Load files
    df_a = pd.read_csv(annotator_a_path)
    df_b = pd.read_csv(annotator_b_path)
    
    # Merge on ID
    merged = df_a.merge(df_b, on=id_col, suffixes=('_A', '_B'))
    
    labels1 = merged[f'{label_col}_A'].tolist()
    labels2 = merged[f'{label_col}_B'].tolist()
    
    label_names = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    
    # Calculate
    report = calculate_agreement_report(labels1, labels2, label_names)
    
    print(f"\n=== Inter-Annotator Agreement Report ===")
    print(f"Files: {annotator_a_path} vs {annotator_b_path}")
    print(f"Cohen's Kappa: {report['cohens_kappa']:.4f} ({report['kappa_interpretation']})")
    print(f"Percent Agreement: {report['percent_agreement']*100:.1f}%")
    
    # Find disagreements
    merged['label_A'] = merged[f'{label_col}_A']
    merged['label_B'] = merged[f'{label_col}_B']
    disagreements = find_disagreements(merged)
    
    if len(disagreements) > 0:
        print(f"\n=== Top Disagreements ===")
        for _, row in disagreements.head(10).iterrows():
            print(f"ID {row['id']}: A={row['label_A']}, B={row['label_B']} (diff={row['difference']})")
    
    return report, disagreements


if __name__ == "__main__":
    main()
