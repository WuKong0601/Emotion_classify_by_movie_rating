"""
Training Pipeline for Emotion Classification Models.

This script handles:
- Data loading and preprocessing
- Feature extraction
- Model training with hyperparameter tuning
- Evaluation and result saving
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import TextPreprocessor, DataLoader
from feature_extraction import TFIDFExtractor, FeatureExtractor
from models.traditional_ml import (
    get_model, evaluate_model, 
    NaiveBayesModel, SVMModel, RandomForestModel, LogisticRegressionModel
)


# Label configuration
LABEL_NAMES = ['Negative', 'Neutral', 'Positive']
NUM_CLASSES = 3


class TrainingPipeline:
    """End-to-end training pipeline."""
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "experiments/results",
                 random_state: int = 42):
        """
        Initialize training pipeline.
        
        Args:
            data_path: Path to annotated data (CSV with 'text' and 'label' columns)
            output_dir: Directory to save results
            random_state: Random seed
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        self.preprocessor = TextPreprocessor(use_word_segmentation=False)
        self.feature_extractor = None
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        self.results = {}
        
    def load_data(self, text_col: str = 'processed_text', label_col: str = 'final_label'):
        """Load and prepare data."""
        print("Loading data...")
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
        
        # Handle labels - prefer final_label from auto-labeling
        if label_col not in df.columns:
            if 'reference_label' in df.columns:
                label_col = 'reference_label'
                print(f"Using {label_col} as label column")
            else:
                raise ValueError(f"Column '{label_col}' not found")
        else:
            print(f"Using {label_col} as label column (from auto-labeling)")
        
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
        """Split data into train/val/test sets."""
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
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def extract_features(self, X_train: List[str], X_val: List[str], X_test: List[str],
                         method: str = 'tfidf', **kwargs):
        """Extract features from texts."""
        print(f"Extracting features using {method}...")
        
        self.feature_extractor = FeatureExtractor(method=method, **kwargs)
        
        # Fit on training data, transform all
        X_train_feat = self.feature_extractor.fit_transform(X_train)
        X_val_feat = self.feature_extractor.transform(X_val)
        X_test_feat = self.feature_extractor.transform(X_test)
        
        print(f"Feature shape: {X_train_feat.shape}")
        
        return X_train_feat, X_val_feat, X_test_feat
    
    def train_model(self, model_name: str, X_train, y_train, 
                    tune_hyperparams: bool = True) -> Dict:
        """Train a single model."""
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()}")
        print('='*50)
        
        model = get_model(model_name)
        
        if tune_hyperparams:
            print("Tuning hyperparameters...")
            tune_result = model.tune_hyperparameters(X_train, y_train)
            print(f"Best params: {tune_result['best_params']}")
            print(f"CV Score: {tune_result['best_score']:.4f}")
        else:
            model.build()
            model.fit(X_train, y_train)
        
        return model
    
    def evaluate(self, model, X_test, y_test) -> Dict:
        """Evaluate model on test set."""
        return evaluate_model(model, X_test, y_test, LABEL_NAMES)
    
    def run(self, models: List[str] = None, 
            feature_method: str = 'tfidf',
            tune_hyperparams: bool = True):
        """
        Run full training pipeline.
        
        Args:
            models: List of model names to train
            feature_method: Feature extraction method
            tune_hyperparams: Whether to tune hyperparameters
        """
        if models is None:
            models = ['naive_bayes', 'svm', 'random_forest', 'logistic_regression']
        
        # Load and prepare data
        texts, labels = self.load_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(texts, labels)
        
        # Extract features
        X_train_feat, X_val_feat, X_test_feat = self.extract_features(
            X_train, X_val, X_test, method=feature_method
        )
        
        # Store for later use
        self.X_train, self.X_val, self.X_test = X_train_feat, X_val_feat, X_test_feat
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        # Train and evaluate each model
        self.results = {}
        
        for model_name in models:
            try:
                # Train
                model = self.train_model(
                    model_name, X_train_feat, y_train, 
                    tune_hyperparams=tune_hyperparams
                )
                
                # Evaluate on validation
                val_metrics = self.evaluate(model, X_val_feat, y_val)
                print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
                
                # Evaluate on test
                test_metrics = self.evaluate(model, X_test_feat, y_test)
                print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"Test F1-macro: {test_metrics['classification_report']['macro avg']['f1-score']:.4f}")
                
                # Store results
                self.results[model_name] = {
                    'model': model,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'best_params': model.best_params
                }
                
                # Save model
                model_path = self.output_dir / f"{model_name}_model.pkl"
                model.save(str(model_path))
                print(f"Model saved to {model_path}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                self.results[model_name] = {'error': str(e)}
        
        # Save summary
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save training results to JSON."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_path': str(self.data_path),
            'models': {}
        }
        
        for model_name, result in self.results.items():
            if 'error' in result:
                summary['models'][model_name] = {'error': result['error']}
            else:
                summary['models'][model_name] = {
                    'best_params': result['best_params'],
                    'val_accuracy': result['val_metrics']['accuracy'],
                    'test_accuracy': result['test_metrics']['accuracy'],
                    'test_f1_macro': result['test_metrics']['classification_report']['macro avg']['f1-score'],
                    'test_classification_report': result['test_metrics']['classification_report'],
                    'confusion_matrix': result['test_metrics']['confusion_matrix']
                }
        
        output_path = self.output_dir / "training_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_path}")
        
        # Print summary table
        self.print_summary()
    
    def print_summary(self):
        """Print summary table of results."""
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"{'Model':<25} {'Test Acc':<12} {'F1-macro':<12} {'F1-weighted':<12}")
        print("-" * 70)
        
        for model_name, result in self.results.items():
            if 'error' in result:
                print(f"{model_name:<25} ERROR: {result['error'][:40]}")
            else:
                acc = result['test_metrics']['accuracy']
                f1_macro = result['test_metrics']['classification_report']['macro avg']['f1-score']
                f1_weighted = result['test_metrics']['classification_report']['weighted avg']['f1-score']
                print(f"{model_name:<25} {acc:<12.4f} {f1_macro:<12.4f} {f1_weighted:<12.4f}")
        
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train emotion classification models")
    parser.add_argument("--data", type=str, default="data/processed_reviews.csv",
                        help="Path to annotated data CSV")
    parser.add_argument("--output", type=str, default="experiments/results",
                        help="Output directory")
    parser.add_argument("--models", type=str, nargs="+",
                        default=['naive_bayes', 'svm', 'random_forest', 'logistic_regression'],
                        help="Models to train")
    parser.add_argument("--features", type=str, default="tfidf",
                        choices=['tfidf', 'bow'],
                        help="Feature extraction method")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter tuning")
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline(
        data_path=args.data,
        output_dir=args.output
    )
    
    pipeline.run(
        models=args.models,
        feature_method=args.features,
        tune_hyperparams=not args.no_tune
    )


if __name__ == "__main__":
    main()
