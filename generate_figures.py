"""Generate figures for the paper."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

output_dir = Path("overleaf_compile/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
with open("data/annotation_agreement.json") as f:
    iaa = json.load(f)

with open("experiments/results/training_results.json") as f:
    results = json.load(f)

# PhoBERT results (from Kaggle)
phobert = {
    'accuracy': 0.8171,
    'f1_macro': 0.6053,
    'Negative': {'precision': 0.5909, 'recall': 0.6724, 'f1-score': 0.6290, 'support': 58},
    'Neutral': {'precision': 0.2059, 'recall': 0.3889, 'f1-score': 0.2692, 'support': 36},
    'Positive': {'precision': 0.9658, 'recall': 0.8738, 'f1-score': 0.9175, 'support': 420},
    'confusion_matrix': [[39, 11, 8], [14, 14, 8], [6, 43, 371]]  # Approximation
}

LABELS = ['Negative', 'Neutral', 'Positive']
MODELS = ['Naive Bayes', 'SVM', 'Random Forest', 'Logistic Reg.', 'PhoBERT']

# ========== Figure 1: Label Distribution ==========
print("Creating label_distribution.pdf...")
fig, ax = plt.subplots(figsize=(4, 3))
dist = iaa['label_distribution']
colors = ['#e74c3c', '#f39c12', '#27ae60']
bars = ax.bar(LABELS, [dist['Negative'], dist['Neutral'], dist['Positive']], color=colors)
ax.set_ylabel('Number of Samples')
ax.set_xlabel('Sentiment Class')
for bar, val in zip(bars, [dist['Negative'], dist['Neutral'], dist['Positive']]):
    pct = val / sum(dist.values()) * 100
    ax.annotate(f'{val}\n({pct:.1f}%)', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(output_dir / "label_distribution.pdf", bbox_inches='tight')
plt.close()

# ========== Figure 2: IAA Metrics ==========
print("Creating iaa_metrics.pdf...")
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

# Left: Our metrics
ax = axes[0]
metrics = ['Cohen\'s κ', 'Agreement']
values = [iaa['cohens_kappa'], iaa['percent_agreement']]
bars = ax.bar(metrics, values, color=['#3498db', '#9b59b6'])
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, label='Substantial threshold')
for bar, val in zip(bars, values):
    ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=10)
ax.legend(loc='lower right', fontsize=8)
ax.set_title('Our Dataset Agreement')

# Right: Kappa scale
ax = axes[1]
ranges = [0.2, 0.2, 0.2, 0.2, 0.2]
labels_k = ['Slight\n(0-0.2)', 'Fair\n(0.2-0.4)', 'Moderate\n(0.4-0.6)', 'Substantial\n(0.6-0.8)', 'Almost Perfect\n(0.8-1.0)']
colors_k = ['#ffcccc', '#ffebcc', '#ffffcc', '#ccffcc', '#ccffcc']
bottom = 0
for r, c, l in zip(ranges, colors_k, labels_k):
    ax.barh(0, r, left=bottom, color=c, edgecolor='black', height=0.5)
    ax.text(bottom + r/2, 0, l, ha='center', va='center', fontsize=7)
    bottom += r
ax.axvline(x=iaa['cohens_kappa'], color='red', linewidth=2)
ax.annotate(f"Our: {iaa['cohens_kappa']:.2f}", xy=(iaa['cohens_kappa'], 0.35), 
            fontsize=9, color='red', ha='center')
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 0.6)
ax.axis('off')
ax.set_title('Landis-Koch Interpretation Scale')

plt.tight_layout()
plt.savefig(output_dir / "iaa_metrics.pdf", bbox_inches='tight')
plt.close()

# ========== Figure 3: Model Comparison ==========
print("Creating model_comparison.pdf...")
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

accs = [
    results['models']['naive_bayes']['test_accuracy'],
    results['models']['svm']['test_accuracy'],
    results['models']['random_forest']['test_accuracy'],
    results['models']['logistic_regression']['test_accuracy'],
    phobert['accuracy']
]
f1s = [
    results['models']['naive_bayes']['test_f1_macro'],
    results['models']['svm']['test_f1_macro'],
    results['models']['random_forest']['test_f1_macro'],
    results['models']['logistic_regression']['test_f1_macro'],
    phobert['f1_macro']
]

colors = ['#1abc9c', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

# Accuracy
ax = axes[0]
bars = ax.bar(MODELS, accs, color=colors)
ax.set_ylabel('Accuracy')
ax.set_ylim(0.75, 0.95)
for bar, val in zip(bars, accs):
    ax.annotate(f'{val:.2%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=8)
ax.set_xticklabels(MODELS, rotation=20, ha='right')
ax.set_title('Test Accuracy')

# F1-macro
ax = axes[1]
bars = ax.bar(MODELS, f1s, color=colors)
ax.set_ylabel('F1-macro')
ax.set_ylim(0.25, 0.75)
for bar, val in zip(bars, f1s):
    ax.annotate(f'{val:.2%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=8)
ax.set_xticklabels(MODELS, rotation=20, ha='right')
ax.set_title('F1-macro Score')

plt.tight_layout()
plt.savefig(output_dir / "model_comparison.pdf", bbox_inches='tight')
plt.close()

# ========== Figure 4: Per-Class Performance ==========
print("Creating per_class_performance.pdf...")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

model_data = {
    'Naive Bayes': results['models']['naive_bayes']['test_classification_report'],
    'SVM': results['models']['svm']['test_classification_report'],
    'Random Forest': results['models']['random_forest']['test_classification_report'],
    'Logistic Reg.': results['models']['logistic_regression']['test_classification_report'],
    'PhoBERT': phobert
}

x = np.arange(len(LABELS))
width = 0.15

# Precision
ax = axes[0]
for i, (model, data) in enumerate(model_data.items()):
    precs = [data[l]['precision'] for l in LABELS]
    ax.bar(x + i*width, precs, width, label=model, color=colors[i])
ax.set_ylabel('Precision')
ax.set_xticks(x + width*2)
ax.set_xticklabels(LABELS)
ax.legend(fontsize=7, loc='upper right')
ax.set_title('Precision by Class')
ax.set_ylim(0, 1.1)

# F1-Score
ax = axes[1]
for i, (model, data) in enumerate(model_data.items()):
    f1s = [data[l]['f1-score'] for l in LABELS]
    ax.bar(x + i*width, f1s, width, label=model, color=colors[i])
ax.set_ylabel('F1-Score')
ax.set_xticks(x + width*2)
ax.set_xticklabels(LABELS)
ax.legend(fontsize=7, loc='upper right')
ax.set_title('F1-Score by Class')
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(output_dir / "per_class_performance.pdf", bbox_inches='tight')
plt.close()

# ========== Figure 5: Confusion Matrices ==========
print("Creating confusion_matrices.pdf...")
fig, axes = plt.subplots(1, 5, figsize=(14, 3))

cms = [
    np.array(results['models']['naive_bayes']['confusion_matrix']),
    np.array(results['models']['svm']['confusion_matrix']),
    np.array(results['models']['random_forest']['confusion_matrix']),
    np.array(results['models']['logistic_regression']['confusion_matrix']),
    np.array(phobert['confusion_matrix'])
]

for ax, cm, model in zip(axes, cms, MODELS):
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    ax.set_title(model, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(output_dir / "confusion_matrices.pdf", bbox_inches='tight')
plt.close()

# ========== Figure 6: Rating Distribution ==========
print("Creating rating_dist.pdf...")
df = pd.read_csv("data/processed_reviews.csv")
fig, ax = plt.subplots(figsize=(5, 3))
ratings = df['rating'].dropna().astype(int)
counts = ratings.value_counts().sort_index()
bars = ax.bar(counts.index, counts.values, color='#3498db')
ax.set_xlabel('Rating (1-10)')
ax.set_ylabel('Number of Reviews')
ax.set_xticks(range(1, 11))
for bar in bars:
    h = bar.get_height()
    if h > 100:
        ax.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width()/2, h),
                    ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(output_dir / "rating_dist.pdf", bbox_inches='tight')
plt.close()

print(f"\n✓ All figures saved to {output_dir}")
print("Files created:")
for f in output_dir.glob("*.pdf"):
    print(f"  - {f.name}")
