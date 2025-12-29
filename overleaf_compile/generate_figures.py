"""
Generate all figures for the emotion classification paper.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# Output directory
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# Load data
data_path = Path("../data")
results_path = Path("../experiments/results")
phobert_path = Path("../phobert_results")

# 1. Label Distribution
print("Generating label distribution figure...")
with open(data_path / "annotation_agreement.json", 'r') as f:
    agreement_data = json.load(f)

labels = ['Negative', 'Neutral', 'Positive']
counts = [agreement_data['label_distribution'][label] for label in labels]
colors = ['#e74c3c', '#95a5a6', '#2ecc71']

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Number of Samples', fontweight='bold')
ax.set_xlabel('Sentiment Class', fontweight='bold')
ax.set_title('Label Distribution (3-Class Sentiment)', fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({count/sum(counts)*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "label_distribution.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "label_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Model Comparison
print("Generating model comparison figure...")
with open(results_path / "training_results.json", 'r') as f:
    ml_results = json.load(f)

with open(phobert_path / "phobert_3class_results.json", 'r') as f:
    phobert_results = json.load(f)

models = ['Naive Bayes', 'SVM', 'Random Forest', 'Logistic\nRegression', 'PhoBERT']
auc_scores = [
    ml_results['models']['naive_bayes']['test_accuracy'],
    ml_results['models']['svm']['test_accuracy'],
    ml_results['models']['random_forest']['test_accuracy'],
    ml_results['models']['logistic_regression']['test_accuracy'],
    phobert_results['test_metrics']['accuracy']
]
f1_macro = [
    ml_results['models']['naive_bayes']['test_f1_macro'],
    ml_results['models']['svm']['test_f1_macro'],
    ml_results['models']['random_forest']['test_f1_macro'],
    ml_results['models']['logistic_regression']['test_f1_macro'],
    phobert_results['test_metrics']['f1_macro']
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Accuracy comparison
x = np.arange(len(models))
bars1 = ax1.bar(x, auc_scores, color=['#3498db']*4 + ['#e74c3c'], alpha=0.8, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Accuracy', fontweight='bold')
ax1.set_xlabel('Model', fontweight='bold')
ax1.set_title('Test Accuracy Comparison', fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15, ha='right')
ax1.set_ylim([0.3, 1.0])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)

for bar, score in zip(bars1, auc_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# F1-macro comparison
bars2 = ax2.bar(x, f1_macro, color=['#3498db']*4 + ['#e74c3c'], alpha=0.8, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('F1-Macro Score', fontweight='bold')
ax2.set_xlabel('Model', fontweight='bold')
ax2.set_title('Test F1-Macro Comparison', fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=15, ha='right')
ax2.set_ylim([0.3, 1.0])
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, linewidth=1)

for bar, score in zip(bars2, f1_macro):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "model_comparison.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Confusion Matrices
print("Generating confusion matrices...")
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

model_names = ['Naive Bayes', 'SVM', 'Random Forest', 'Logistic Regression', 'PhoBERT']
model_keys = ['naive_bayes', 'svm', 'random_forest', 'logistic_regression']

for idx, (model_name, model_key) in enumerate(zip(model_names[:4], model_keys)):
    cm = np.array(ml_results['models'][model_key]['confusion_matrix'])
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Normalized'}, ax=axes[idx],
                vmin=0, vmax=1, linewidths=0.5, linecolor='gray')
    axes[idx].set_title(model_name, fontweight='bold', pad=10)
    axes[idx].set_ylabel('True Label', fontweight='bold')
    axes[idx].set_xlabel('Predicted Label', fontweight='bold')

# PhoBERT confusion matrix
cm = np.array(phobert_results['confusion_matrix'])
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Reds', 
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Normalized'}, ax=axes[4],
            vmin=0, vmax=1, linewidths=0.5, linecolor='gray')
axes[4].set_title('PhoBERT', fontweight='bold', pad=10)
axes[4].set_ylabel('True Label', fontweight='bold')
axes[4].set_xlabel('Predicted Label', fontweight='bold')

# Hide the last subplot
axes[5].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "confusion_matrices.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Per-class Performance
print("Generating per-class performance figure...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Prepare data
metrics_data = []
for model_name, model_key in zip(['NB', 'SVM', 'RF', 'LR'], model_keys):
    report = ml_results['models'][model_key]['test_classification_report']
    for label in labels:
        metrics_data.append({
            'Model': model_name,
            'Class': label,
            'Precision': report[label]['precision'],
            'Recall': report[label]['recall'],
            'F1-Score': report[label]['f1-score']
        })

# PhoBERT
phobert_report = phobert_results['classification_report']
for label in labels:
    metrics_data.append({
        'Model': 'PhoBERT',
        'Class': label,
        'Precision': phobert_report[label]['precision'],
        'Recall': phobert_report[label]['recall'],
        'F1-Score': phobert_report[label]['f1-score']
    })

df_metrics = pd.DataFrame(metrics_data)

# Precision by class
df_pivot = df_metrics.pivot(index='Class', columns='Model', values='Precision')
df_pivot[['NB', 'SVM', 'RF', 'LR', 'PhoBERT']].plot(kind='bar', ax=ax1, width=0.8, edgecolor='black', linewidth=0.8)
ax1.set_ylabel('Precision', fontweight='bold')
ax1.set_xlabel('Sentiment Class', fontweight='bold')
ax1.set_title('Precision by Class', fontweight='bold', pad=15)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.legend(title='Model', loc='lower right', framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([0, 1.1])

# F1-Score by class
df_pivot = df_metrics.pivot(index='Class', columns='Model', values='F1-Score')
df_pivot[['NB', 'SVM', 'RF', 'LR', 'PhoBERT']].plot(kind='bar', ax=ax2, width=0.8, edgecolor='black', linewidth=0.8)
ax2.set_ylabel('F1-Score', fontweight='bold')
ax2.set_xlabel('Sentiment Class', fontweight='bold')
ax2.set_title('F1-Score by Class', fontweight='bold', pad=15)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.legend(title='Model', loc='lower right', framealpha=0.9)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig(output_dir / "per_class_performance.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "per_class_performance.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Inter-Annotator Agreement
print("Generating IAA visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Kappa score
kappa = agreement_data['cohens_kappa']
agreement_pct = agreement_data['percent_agreement']

categories = ['Cohen\'s Kappa', 'Percent\nAgreement']
scores = [kappa, agreement_pct]
colors_iaa = ['#3498db', '#2ecc71']

bars = ax1.bar(categories, scores, color=colors_iaa, alpha=0.8, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('Inter-Annotator Agreement', fontweight='bold', pad=15)
ax1.set_ylim([0, 1.1])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Substantial (0.8)')
ax1.legend()

for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Agreement interpretation
interpretations = ['Poor\n(<0.2)', 'Fair\n(0.2-0.4)', 'Moderate\n(0.4-0.6)', 
                   'Substantial\n(0.6-0.8)', 'Almost Perfect\n(0.8-1.0)']
ranges = [0.2, 0.4, 0.6, 0.8, 1.0]
colors_range = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71']

bars = ax2.barh(interpretations, ranges, color=colors_range, alpha=0.6, edgecolor='black', linewidth=1)
ax2.axvline(x=kappa, color='red', linestyle='--', linewidth=2, label=f'Our Kappa: {kappa:.3f}')
ax2.set_xlabel('Cohen\'s Kappa Range', fontweight='bold')
ax2.set_title('Kappa Interpretation Scale', fontweight='bold', pad=15)
ax2.set_xlim([0, 1.0])
ax2.legend(loc='lower right')
ax2.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / "iaa_metrics.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "iaa_metrics.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ All figures generated successfully in {output_dir}/")
print(f"  - label_distribution.pdf/png")
print(f"  - model_comparison.pdf/png")
print(f"  - confusion_matrices.pdf/png")
print(f"  - per_class_performance.pdf/png")
print(f"  - iaa_metrics.pdf/png")
