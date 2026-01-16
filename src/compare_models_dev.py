"""
So sánh các mô hình trên tập Development (Validation Set).

Script này:
- Load tất cả các models đã train
- Đánh giá trên tập validation (dev set)
- So sánh metrics giữa các models
- Tạo visualizations và báo cáo
"""

import argparse
import json
import pickle
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
from evaluate import (
    calculate_metrics, 
    plot_model_comparison, plot_per_class_metrics,
    generate_latex_table
)

# Label configuration
LABEL_NAMES = ['Negative', 'Neutral', 'Positive']
NUM_CLASSES = 3


class ModelComparator:
    """So sánh các mô hình trên tập development."""
    
    def __init__(self, 
                 data_path: str,
                 models_dir: str = "experiments/results",
                 output_dir: str = "experiments/comparison",
                 random_state: int = 42):
        """
        Initialize model comparator.
        
        Args:
            data_path: Path to labeled data CSV
            models_dir: Directory containing trained models
            output_dir: Directory to save comparison results
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        self.preprocessor = TextPreprocessor(use_word_segmentation=False)
        self.feature_extractor = None
        self.models = {}
        self.results = {}
        
        # Data splits
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_data(self, text_col: str = 'processed_text', label_col: str = 'final_label'):
        """Load và split data giống như train.py."""
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
        print(f"Label distribution: {pd.Series(labels).value_counts().sort_index().to_dict()}")
        
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
    
    def load_models(self, model_names: Optional[List[str]] = None):
        """Load các models đã train."""
        print("\n" + "=" * 70)
        print("LOADING MODELS")
        print("=" * 70)
        
        if model_names is None:
            # Auto-detect available models
            model_files = {
                'naive_bayes': 'naive_bayes_model.pkl',
                'svm': 'svm_model.pkl',
                'random_forest': 'random_forest_model.pkl',
                'logistic_regression': 'logistic_regression_model.pkl'
            }
        else:
            model_files = {name: f"{name}_model.pkl" for name in model_names}
        
        loaded_models = {}
        for name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    model = get_model(name)
                    model.load(str(model_path))
                    loaded_models[name] = model
                    print(f"[OK] Loaded {name}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {name}: {e}")
            else:
                print(f"[SKIP] Model file not found: {model_path}")
        
        self.models = loaded_models
        print(f"\nLoaded {len(loaded_models)} models: {list(loaded_models.keys())}")
        
        return loaded_models
    
    def prepare_features(self, feature_method: str = 'tfidf'):
        """Chuẩn bị features cho validation set."""
        print("\n" + "=" * 70)
        print("PREPARING FEATURES")
        print("=" * 70)
        
        # Load vectorizer if exists, otherwise create new one
        vectorizer_path = self.models_dir / "tfidf_vectorizer.pkl"
        
        if vectorizer_path.exists():
            print(f"Loading vectorizer from {vectorizer_path}")
            self.feature_extractor = FeatureExtractor(method=feature_method)
            if feature_method == 'tfidf':
                self.feature_extractor.extractor.load(str(vectorizer_path))
        else:
            print("Creating new vectorizer...")
            self.feature_extractor = FeatureExtractor(method=feature_method)
            # Fit on training data
            X_train_feat = self.feature_extractor.fit_transform(self.X_train)
            # Save vectorizer
            if feature_method == 'tfidf':
                self.feature_extractor.extractor.save(str(vectorizer_path))
                print(f"Saved vectorizer to {vectorizer_path}")
        
        # Transform validation set
        X_val_feat = self.feature_extractor.transform(self.X_val)
        print(f"Validation features shape: {X_val_feat.shape}")
        
        return X_val_feat
    
    def evaluate_all_models(self, X_val_feat, y_val):
        """Đánh giá tất cả models trên validation set."""
        print("\n" + "=" * 70)
        print("EVALUATING MODELS ON DEVELOPMENT SET")
        print("=" * 70)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name.upper()}")
            print('='*50)
            
            try:
                # Predict
                y_pred = model.predict(X_val_feat)
                
                # Calculate metrics
                metrics = calculate_metrics(y_val, y_pred.tolist(), LABEL_NAMES)
                
                # Store results
                results[model_name] = {
                    'metrics': metrics,
                    'predictions': y_pred.tolist(),
                    'true_labels': y_val
                }
                
                # Print summary
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"F1-macro: {metrics['macro_avg']['f1-score']:.4f}")
                print(f"F1-weighted: {metrics['weighted_avg']['f1-score']:.4f}")
                
                print("\nPer-class F1 scores:")
                for label_name, values in metrics['per_class'].items():
                    print(f"  {label_name}: {values['f1-score']:.4f} (support: {values['support']})")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Tạo bảng so sánh metrics."""
        data = []
        
        for model_name, result in self.results.items():
            if 'error' not in result:
                metrics = result['metrics']
                data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': metrics['accuracy'],
                    'Precision (Macro)': metrics['macro_avg']['precision'],
                    'Recall (Macro)': metrics['macro_avg']['recall'],
                    'F1-Score (Macro)': metrics['macro_avg']['f1-score'],
                    'Precision (Weighted)': metrics['weighted_avg']['precision'],
                    'Recall (Weighted)': metrics['weighted_avg']['recall'],
                    'F1-Score (Weighted)': metrics['weighted_avg']['f1-score'],
                })
                
                # Add per-class F1
                for label_name, values in metrics['per_class'].items():
                    data[-1][f'F1-{label_name}'] = values['f1-score']
        
        df = pd.DataFrame(data)
        return df
    
    def save_results(self):
        """Lưu kết quả so sánh."""
        print("\n" + "=" * 70)
        print("SAVING RESULTS")
        print("=" * 70)
        
        # Save JSON
        json_path = self.output_dir / "dev_set_comparison.json"
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_path': str(self.data_path),
            'validation_set_size': len(self.y_val),
            'models': {}
        }
        
        for model_name, result in self.results.items():
            if 'error' not in result:
                summary['models'][model_name] = result['metrics']
            else:
                summary['models'][model_name] = {'error': result['error']}
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved JSON to {json_path}")
        
        # Save CSV comparison table
        comparison_df = self.create_comparison_table()
        csv_path = self.output_dir / "dev_set_comparison.csv"
        comparison_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"[OK] Saved CSV to {csv_path}")
        
        # Print table
        print("\n" + "=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)
        print(comparison_df.to_string(index=False))
        
        # Save LaTeX table
        latex_path = self.output_dir / "dev_set_comparison.tex"
        # Convert results format for LaTeX generator
        latex_results = {}
        for model_name, result in self.results.items():
            if 'error' not in result:
                # Convert to format expected by generate_latex_table
                latex_results[model_name] = {
                    'accuracy': result['metrics']['accuracy'],
                    'classification_report': {
                        'macro avg': {
                            'precision': result['metrics']['macro_avg']['precision'],
                            'recall': result['metrics']['macro_avg']['recall'],
                            'f1-score': result['metrics']['macro_avg']['f1-score']
                        },
                        'weighted avg': {
                            'precision': result['metrics']['weighted_avg']['precision'],
                            'recall': result['metrics']['weighted_avg']['recall'],
                            'f1-score': result['metrics']['weighted_avg']['f1-score']
                        }
                    }
                }
        
        latex_table = generate_latex_table(
            latex_results,
            caption="Model Comparison on Development Set",
            label="tab:dev_comparison"
        )
        
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        print(f"[OK] Saved LaTeX table to {latex_path}")
        
        return summary, comparison_df
    
    def create_visualizations(self):
        """Tạo các biểu đồ so sánh."""
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)
        
        # Convert results format for plotting functions
        plot_results = {}
        for model_name, result in self.results.items():
            if 'error' not in result:
                # Convert to format expected by plotting functions
                plot_results[model_name] = {
                    'accuracy': result['metrics']['accuracy'],
                    'classification_report': {
                        'macro avg': {
                            'precision': result['metrics']['macro_avg']['precision'],
                            'recall': result['metrics']['macro_avg']['recall'],
                            'f1-score': result['metrics']['macro_avg']['f1-score']
                        },
                        'weighted avg': {
                            'precision': result['metrics']['weighted_avg']['precision'],
                            'recall': result['metrics']['weighted_avg']['recall'],
                            'f1-score': result['metrics']['weighted_avg']['f1-score']
                        }
                    }
                }
                # Add per-class metrics
                for label_name, values in result['metrics']['per_class'].items():
                    if label_name not in plot_results[model_name]['classification_report']:
                        plot_results[model_name]['classification_report'][label_name] = {}
                    plot_results[model_name]['classification_report'][label_name] = {
                        'f1-score': values['f1-score'],
                        'precision': values['precision'],
                        'recall': values['recall']
                    }
        
        # 1. Model comparison (F1-score)
        print("\n1. Creating F1-score comparison plot...")
        fig_path = self.output_dir / "dev_f1_comparison.png"
        plot_model_comparison(
            plot_results,
            metric='f1-score',
            figsize=(12, 6),
            save_path=str(fig_path)
        )
        plt.close()
        
        # 2. Per-class F1 scores
        print("2. Creating per-class F1 scores plot...")
        fig_path = self.output_dir / "dev_per_class_f1.png"
        plot_per_class_metrics(
            plot_results,
            figsize=(14, 8),
            save_path=str(fig_path)
        )
        plt.close()
        
        # 3. Confusion matrices for each model
        print("3. Creating confusion matrices...")
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        
        n_models = len([r for r in self.results.values() if 'error' not in r])
        if n_models > 0:
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
            if n_models == 1:
                axes = [axes]
            
            for idx, (model_name, result) in enumerate(self.results.items()):
                if 'error' not in result:
                    y_true = result['true_labels']
                    y_pred = result['predictions']
                    
                    # Create confusion matrix
                    cm = sk_confusion_matrix(y_true, y_pred)
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    
                    # Plot
                    sns.heatmap(
                        cm_normalized, 
                        annot=True, 
                        fmt='.2f', 
                        cmap='Blues',
                        xticklabels=LABEL_NAMES, 
                        yticklabels=LABEL_NAMES,
                        ax=axes[idx]
                    )
                    axes[idx].set_title(f"{model_name.replace('_', ' ').title()}")
                    axes[idx].set_ylabel('True Label')
                    axes[idx].set_xlabel('Predicted Label')
            
            plt.tight_layout()
            fig_path = self.output_dir / "dev_confusion_matrices.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved confusion matrices to {fig_path}")
        
        print("\n[OK] All visualizations saved!")
    
    def run(self, model_names: Optional[List[str]] = None, 
            feature_method: str = 'tfidf'):
        """
        Chạy toàn bộ pipeline so sánh.
        
        Args:
            model_names: List of model names to compare (None = auto-detect)
            feature_method: Feature extraction method
        """
        # Load data
        texts, labels = self.load_data()
        
        # Split data
        self.split_data(texts, labels)
        
        # Load models
        self.load_models(model_names)
        
        if len(self.models) == 0:
            print("\n[ERROR] No models loaded. Exiting.")
            return
        
        # Prepare features
        X_val_feat = self.prepare_features(feature_method)
        
        # Evaluate all models
        self.evaluate_all_models(X_val_feat, self.y_val)
        
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
        description="So sánh các mô hình trên tập Development (Validation Set)"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/labeled_reviews.csv",
        help="Path to labeled data CSV"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="experiments/results",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/comparison",
        help="Output directory for comparison results"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specific models to compare (default: all available)"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="tfidf",
        choices=['tfidf', 'bow'],
        help="Feature extraction method"
    )
    
    args = parser.parse_args()
    
    comparator = ModelComparator(
        data_path=args.data,
        models_dir=args.models_dir,
        output_dir=args.output,
        random_state=42
    )
    
    comparator.run(
        model_names=args.models,
        feature_method=args.features
    )


if __name__ == "__main__":
    main()

