"""
So sánh hiệu quả của Class Weighting trong việc xử lý class imbalance.

Script này:
- Train models với và không có class_weight='balanced'
- So sánh kết quả trên development set
- Phân tích tác động của class weighting lên từng class
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import TextPreprocessor, DataLoader
from feature_extraction import FeatureExtractor
from models.traditional_ml import BaseModel, get_model
from evaluate import calculate_metrics

# Label configuration
LABEL_NAMES = ['Negative', 'Neutral', 'Positive']
NUM_CLASSES = 3


class ClassWeightComparator:
    """So sánh models với và không có class weighting."""
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "experiments/class_weight_comparison",
                 random_state: int = 42):
        """
        Initialize comparator.
        
        Args:
            data_path: Path to labeled data CSV
            output_dir: Directory to save comparison results
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        self.preprocessor = TextPreprocessor(use_word_segmentation=False)
        self.feature_extractor = None
        
        # Data splits
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        self.results = {}
        
    def load_data(self, text_col: str = 'processed_text', label_col: str = 'final_label'):
        """Load và split data."""
        print("=" * 70)
        print("LOADING DATA")
        print("=" * 70)
        
        df = pd.read_csv(self.data_path)
        
        # Check for processed_text column
        if text_col not in df.columns:
            if 'review_text' in df.columns:
                print("Preprocessing review_text...")
                df['processed_text'] = df['review_text'].fillna('').apply(
                    self.preprocessor.preprocess
                )
                text_col = 'processed_text'
            else:
                raise ValueError(f"Column '{text_col}' not found")
        
        # Handle labels
        if label_col not in df.columns:
            if 'reference_label' in df.columns:
                label_col = 'reference_label'
                print(f"Using {label_col} as label column")
            else:
                raise ValueError(f"Column '{label_col}' not found")
        else:
            print(f"Using {label_col} as label column")
        
        # Filter valid samples
        df = df[df[text_col].str.len() > 0].copy()
        df = df[df[label_col].notna()].copy()
        
        texts = df[text_col].tolist()
        labels = df[label_col].astype(int).tolist()
        
        print(f"Loaded {len(texts)} samples")
        
        # Show class distribution
        label_dist = pd.Series(labels).value_counts().sort_index()
        print(f"\nClass distribution:")
        for label_id, name in enumerate(LABEL_NAMES):
            count = label_dist.get(label_id, 0)
            pct = count / len(labels) * 100
            print(f"  {label_id} ({name}): {count} ({pct:.1f}%)")
        
        return texts, labels
    
    def split_data(self, texts: List[str], labels: List[int],
                   test_size: float = 0.15, val_size: float = 0.15):
        """Split data giống như train.py."""
        print("\nSplitting data...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        # Second split: separate validation set
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Store splits
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_features(self, feature_method: str = 'tfidf'):
        """Chuẩn bị features."""
        print("\n" + "=" * 70)
        print("PREPARING FEATURES")
        print("=" * 70)
        
        self.feature_extractor = FeatureExtractor(method=feature_method)
        
        # Fit on training data
        X_train_feat = self.feature_extractor.fit_transform(self.X_train)
        X_val_feat = self.feature_extractor.transform(self.X_val)
        
        print(f"Training features shape: {X_train_feat.shape}")
        print(f"Validation features shape: {X_val_feat.shape}")
        
        return X_train_feat, X_val_feat
    
    def train_model_with_class_weight(self, model_name: str, X_train, y_train, 
                                     use_class_weight: bool = True):
        """Train model với hoặc không có class_weight."""
        if model_name == 'naive_bayes':
            # Naive Bayes doesn't support class_weight
            model = get_model(model_name)
            model.build()
            model.fit(X_train, y_train)
            return model
        
        model = get_model(model_name)
        
        if use_class_weight:
            # Use balanced class weights
            if model_name == 'svm':
                model.build(class_weight='balanced')
            elif model_name == 'random_forest':
                model.build(class_weight='balanced')
            elif model_name == 'logistic_regression':
                model.build(class_weight='balanced')
        else:
            # No class weighting
            if model_name == 'svm':
                model.build(class_weight=None)
            elif model_name == 'random_forest':
                model.build(class_weight=None)
            elif model_name == 'logistic_regression':
                model.build(class_weight=None)
        
        model.fit(X_train, y_train)
        return model
    
    def evaluate_all_models(self, X_train_feat, X_val_feat, y_train, y_val,
                            model_names: List[str] = None):
        """Train và đánh giá models với và không có class_weight."""
        print("\n" + "=" * 70)
        print("TRAINING AND EVALUATING MODELS")
        print("=" * 70)
        
        if model_names is None:
            model_names = ['svm', 'random_forest', 'logistic_regression']
        
        results = {}
        
        for model_name in model_names:
            print(f"\n{'='*60}")
            print(f"MODEL: {model_name.upper()}")
            print('='*60)
            
            # Train without class_weight
            print("\n[1] Training WITHOUT class_weight...")
            model_no_weight = self.train_model_with_class_weight(
                model_name, X_train_feat, y_train, use_class_weight=False
            )
            
            y_pred_no_weight = model_no_weight.predict(X_val_feat)
            metrics_no_weight = calculate_metrics(y_val, y_pred_no_weight.tolist(), LABEL_NAMES)
            
            print(f"  Accuracy: {metrics_no_weight['accuracy']:.4f}")
            print(f"  F1-macro: {metrics_no_weight['macro_avg']['f1-score']:.4f}")
            print(f"  F1-weighted: {metrics_no_weight['weighted_avg']['f1-score']:.4f}")
            print("  Per-class F1:")
            for label_name, values in metrics_no_weight['per_class'].items():
                print(f"    {label_name}: {values['f1-score']:.4f} (support: {values['support']})")
            
            # Train with class_weight
            print("\n[2] Training WITH class_weight='balanced'...")
            model_with_weight = self.train_model_with_class_weight(
                model_name, X_train_feat, y_train, use_class_weight=True
            )
            
            y_pred_with_weight = model_with_weight.predict(X_val_feat)
            metrics_with_weight = calculate_metrics(y_val, y_pred_with_weight.tolist(), LABEL_NAMES)
            
            print(f"  Accuracy: {metrics_with_weight['accuracy']:.4f}")
            print(f"  F1-macro: {metrics_with_weight['macro_avg']['f1-score']:.4f}")
            print(f"  F1-weighted: {metrics_with_weight['weighted_avg']['f1-score']:.4f}")
            print("  Per-class F1:")
            for label_name, values in metrics_with_weight['per_class'].items():
                print(f"    {label_name}: {values['f1-score']:.4f} (support: {values['support']})")
            
            # Calculate improvement
            print("\n[3] Improvement:")
            acc_improve = metrics_with_weight['accuracy'] - metrics_no_weight['accuracy']
            f1_macro_improve = metrics_with_weight['macro_avg']['f1-score'] - metrics_no_weight['macro_avg']['f1-score']
            f1_weighted_improve = metrics_with_weight['weighted_avg']['f1-score'] - metrics_no_weight['weighted_avg']['f1-score']
            
            print(f"  Accuracy: {acc_improve:+.4f} ({acc_improve/metrics_no_weight['accuracy']*100:+.2f}%)")
            print(f"  F1-macro: {f1_macro_improve:+.4f} ({f1_macro_improve/metrics_no_weight['macro_avg']['f1-score']*100:+.2f}%)")
            print(f"  F1-weighted: {f1_weighted_improve:+.4f} ({f1_weighted_improve/metrics_no_weight['weighted_avg']['f1-score']*100:+.2f}%)")
            
            # Store results
            results[model_name] = {
                'without_class_weight': {
                    'metrics': metrics_no_weight,
                    'predictions': y_pred_no_weight.tolist()
                },
                'with_class_weight': {
                    'metrics': metrics_with_weight,
                    'predictions': y_pred_with_weight.tolist()
                },
                'improvement': {
                    'accuracy': acc_improve,
                    'f1_macro': f1_macro_improve,
                    'f1_weighted': f1_weighted_improve
                }
            }
        
        self.results = results
        return results
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Tạo bảng so sánh."""
        data = []
        
        for model_name, result in self.results.items():
            no_weight = result['without_class_weight']['metrics']
            with_weight = result['with_class_weight']['metrics']
            improve = result['improvement']
            
            data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Acc (No Weight)': no_weight['accuracy'],
                'Acc (With Weight)': with_weight['accuracy'],
                'Acc Improvement': improve['accuracy'],
                'F1-Macro (No Weight)': no_weight['macro_avg']['f1-score'],
                'F1-Macro (With Weight)': with_weight['macro_avg']['f1-score'],
                'F1-Macro Improvement': improve['f1_macro'],
                'F1-Weighted (No Weight)': no_weight['weighted_avg']['f1-score'],
                'F1-Weighted (With Weight)': with_weight['weighted_avg']['f1-score'],
                'F1-Weighted Improvement': improve['f1_weighted'],
            })
            
            # Add per-class improvements
            for label_name in LABEL_NAMES:
                f1_no = no_weight['per_class'][label_name]['f1-score']
                f1_with = with_weight['per_class'][label_name]['f1-score']
                f1_improve = f1_with - f1_no
                data[-1][f'F1-{label_name} (No)'] = f1_no
                data[-1][f'F1-{label_name} (With)'] = f1_with
                data[-1][f'F1-{label_name} (Improve)'] = f1_improve
        
        df = pd.DataFrame(data)
        return df
    
    def save_results(self):
        """Lưu kết quả so sánh."""
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        
        # Save JSON
        json_path = self.output_dir / "class_weight_comparison.json"
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_path': str(self.data_path),
            'validation_set_size': len(self.y_val),
            'models': {}
        }
        
        for model_name, result in self.results.items():
            summary['models'][model_name] = {
                'without_class_weight': result['without_class_weight']['metrics'],
                'with_class_weight': result['with_class_weight']['metrics'],
                'improvement': result['improvement']
            }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved JSON to {json_path}")
        
        # Save CSV
        comparison_df = self.create_comparison_table()
        csv_path = self.output_dir / "class_weight_comparison.csv"
        comparison_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"[OK] Saved CSV to {csv_path}")
        
        # Print table
        print("\n" + "=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)
        print(comparison_df.to_string(index=False))
        
        return summary, comparison_df
    
    def create_visualizations(self):
        """Tạo visualizations so sánh."""
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)
        
        # 1. Comparison bar chart: F1-macro improvement
        print("\n1. Creating F1-macro improvement chart...")
        models = []
        f1_no_weight = []
        f1_with_weight = []
        
        for model_name, result in self.results.items():
            models.append(model_name.replace('_', ' ').title())
            f1_no_weight.append(result['without_class_weight']['metrics']['macro_avg']['f1-score'])
            f1_with_weight.append(result['with_class_weight']['metrics']['macro_avg']['f1-score'])
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, f1_no_weight, width, label='Without Class Weight', color='lightcoral', alpha=0.8)
        bars2 = ax.bar(x + width/2, f1_with_weight, width, label='With Class Weight', color='steelblue', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('F1-Score (Macro)', fontsize=12)
        ax.set_title('F1-Macro Comparison: With vs Without Class Weight', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        fig_path = self.output_dir / "f1_macro_comparison.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved to {fig_path}")
        
        # 2. Per-class F1 improvement
        print("2. Creating per-class F1 improvement chart...")
        data = []
        for model_name, result in self.results.items():
            for label_name in LABEL_NAMES:
                f1_no = result['without_class_weight']['metrics']['per_class'][label_name]['f1-score']
                f1_with = result['with_class_weight']['metrics']['per_class'][label_name]['f1-score']
                data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Class': label_name,
                    'Without Class Weight': f1_no,
                    'With Class Weight': f1_with,
                    'Improvement': f1_with - f1_no
                })
        
        df_plot = pd.DataFrame(data)
        
        fig, axes = plt.subplots(1, len(LABEL_NAMES), figsize=(6*len(LABEL_NAMES), 6))
        if len(LABEL_NAMES) == 1:
            axes = [axes]
        
        for idx, label_name in enumerate(LABEL_NAMES):
            df_class = df_plot[df_plot['Class'] == label_name]
            
            x = np.arange(len(df_class))
            width = 0.35
            
            axes[idx].bar(x - width/2, df_class['Without Class Weight'], width, 
                         label='Without', color='lightcoral', alpha=0.8)
            axes[idx].bar(x + width/2, df_class['With Class Weight'], width,
                         label='With', color='steelblue', alpha=0.8)
            
            axes[idx].set_xlabel('Model', fontsize=11)
            axes[idx].set_ylabel('F1-Score', fontsize=11)
            axes[idx].set_title(f'{label_name} Class', fontsize=12, fontweight='bold')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(df_class['Model'], rotation=45, ha='right')
            axes[idx].legend()
            axes[idx].set_ylim(0, 1)
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.output_dir / "per_class_f1_comparison.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved to {fig_path}")
        
        # 3. Improvement heatmap
        print("3. Creating improvement heatmap...")
        improvement_data = []
        for model_name, result in self.results.items():
            row = {'Model': model_name.replace('_', ' ').title()}
            for label_name in LABEL_NAMES:
                f1_no = result['without_class_weight']['metrics']['per_class'][label_name]['f1-score']
                f1_with = result['with_class_weight']['metrics']['per_class'][label_name]['f1-score']
                row[label_name] = f1_with - f1_no
            improvement_data.append(row)
        
        df_heatmap = pd.DataFrame(improvement_data).set_index('Model')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'F1-Score Improvement'}, vmin=-0.5, vmax=0.5)
        plt.title('F1-Score Improvement by Class and Model\n(With Class Weight - Without Class Weight)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Model', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.tight_layout()
        fig_path = self.output_dir / "improvement_heatmap.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved to {fig_path}")
        
        print("\n[OK] All visualizations saved!")
    
    def run(self, model_names: Optional[List[str]] = None, 
            feature_method: str = 'tfidf'):
        """Chạy toàn bộ pipeline so sánh."""
        # Load data
        texts, labels = self.load_data()
        
        # Split data
        self.split_data(texts, labels)
        
        # Prepare features
        X_train_feat, X_val_feat = self.prepare_features(feature_method)
        
        # Train and evaluate
        self.evaluate_all_models(X_train_feat, X_val_feat, self.y_train, self.y_val, model_names)
        
        # Save results
        summary, comparison_df = self.save_results()
        
        # Create visualizations
        self.create_visualizations()
        
        print("\n" + "=" * 70)
        print("COMPARISON COMPLETE!")
        print("=" * 70)
        print(f"Results saved to: {self.output_dir}")
        
        return summary, comparison_df


def main():
    parser = argparse.ArgumentParser(
        description="So sánh hiệu quả của Class Weighting"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/labeled_reviews.csv",
        help="Path to labeled data CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/class_weight_comparison",
        help="Output directory for comparison results"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=['svm', 'random_forest', 'logistic_regression'],
        help="Models to compare (default: svm, random_forest, logistic_regression)"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="tfidf",
        choices=['tfidf', 'bow'],
        help="Feature extraction method"
    )
    
    args = parser.parse_args()
    
    comparator = ClassWeightComparator(
        data_path=args.data,
        output_dir=args.output,
        random_state=42
    )
    
    comparator.run(
        model_names=args.models,
        feature_method=args.features
    )


if __name__ == "__main__":
    main()

