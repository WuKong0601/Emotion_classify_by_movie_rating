"""
Machine Learning Models for Emotion Classification.

This module implements:
- Naive Bayes
- SVM (Support Vector Machine)
- Random Forest
- Logistic Regression

All models follow sklearn interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.best_params = None
        self.is_fitted = False
        
    @abstractmethod
    def build(self, **kwargs):
        """Build the model with specified parameters."""
        pass
    
    def fit(self, X, y):
        """Train the model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X) -> np.ndarray:
        """Predict labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.name} doesn't support predict_proba")
    
    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'best_params': self.best_params,
                'name': self.name
            }, f)
    
    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.best_params = data.get('best_params')
            self.name = data.get('name', self.name)
        self.is_fitted = True
    
    def cross_validate(self, X, y, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation."""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1_macro')
        return {
            'mean_f1_macro': scores.mean(),
            'std_f1_macro': scores.std(),
            'scores': scores.tolist()
        }


class NaiveBayesModel(BaseModel):
    """Multinomial Naive Bayes classifier."""
    
    def __init__(self):
        super().__init__("Naive Bayes")
        
    def build(self, alpha: float = 1.0, **kwargs):
        """
        Build Naive Bayes model.
        
        Args:
            alpha: Additive smoothing parameter
        """
        self.model = MultinomialNB(alpha=alpha)
        return self
    
    def tune_hyperparameters(self, X, y, cv: int = 5) -> Dict:
        """Grid search for best parameters."""
        param_grid = {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
        }
        
        grid = GridSearchCV(
            MultinomialNB(),
            param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        grid.fit(X, y)
        
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.is_fitted = True
        
        return {
            'best_params': self.best_params,
            'best_score': grid.best_score_
        }


class SVMModel(BaseModel):
    """Support Vector Machine classifier."""
    
    def __init__(self):
        super().__init__("SVM")
        
    def build(self, C: float = 1.0, kernel: str = 'rbf', 
              gamma: str = 'scale', class_weight: str = 'balanced', **kwargs):
        """
        Build SVM model.
        
        Args:
            C: Regularization parameter
            kernel: Kernel type
            gamma: Kernel coefficient
            class_weight: Class weighting strategy
        """
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=True,
            random_state=42
        )
        return self
    
    def tune_hyperparameters(self, X, y, cv: int = 5) -> Dict:
        """Grid search for best parameters."""
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        
        grid = GridSearchCV(
            SVC(class_weight='balanced', probability=True, random_state=42),
            param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        grid.fit(X, y)
        
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.is_fitted = True
        
        return {
            'best_params': self.best_params,
            'best_score': grid.best_score_
        }


class RandomForestModel(BaseModel):
    """Random Forest classifier."""
    
    def __init__(self):
        super().__init__("Random Forest")
        
    def build(self, n_estimators: int = 100, max_depth: Optional[int] = None,
              min_samples_split: int = 2, class_weight: str = 'balanced', **kwargs):
        """
        Build Random Forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            class_weight: Class weighting strategy
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        return self
    
    def tune_hyperparameters(self, X, y, cv: int = 5) -> Dict:
        """Grid search for best parameters."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        grid = GridSearchCV(
            RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
            param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        grid.fit(X, y)
        
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.is_fitted = True
        
        return {
            'best_params': self.best_params,
            'best_score': grid.best_score_
        }
    
    def get_feature_importance(self, feature_names: List[str] = None, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n)


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier."""
    
    def __init__(self):
        super().__init__("Logistic Regression")
        
    def build(self, C: float = 1.0, max_iter: int = 1000, 
              class_weight: str = 'balanced', **kwargs):
        """
        Build Logistic Regression model.
        
        Args:
            C: Inverse regularization strength
            max_iter: Maximum iterations
            class_weight: Class weighting strategy
        """
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        return self
    
    def tune_hyperparameters(self, X, y, cv: int = 5) -> Dict:
        """Grid search for best parameters."""
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'saga']
        }
        
        grid = GridSearchCV(
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        grid.fit(X, y)
        
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.is_fitted = True
        
        return {
            'best_params': self.best_params,
            'best_score': grid.best_score_
        }


def get_model(model_name: str) -> BaseModel:
    """Factory function to get model by name."""
    models = {
        'naive_bayes': NaiveBayesModel,
        'svm': SVMModel,
        'random_forest': RandomForestModel,
        'logistic_regression': LogisticRegressionModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]()


def evaluate_model(model: BaseModel, X_test, y_test, 
                   label_names: List[str] = None) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        label_names: Names for labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Classification report
    report = classification_report(y_test, y_pred, 
                                   target_names=label_names,
                                   output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist()
    }


def main():
    """Test models with sample data."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("Model Testing")
    print("=" * 60)
    
    # Generate sample data
    X, y = make_classification(
        n_samples=500,
        n_features=100,
        n_informative=20,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Make features non-negative for Naive Bayes
    X = X - X.min()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    label_names = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    
    models_to_test = ['naive_bayes', 'svm', 'random_forest', 'logistic_regression']
    
    results = {}
    for model_name in models_to_test:
        print(f"\n=== {model_name.upper()} ===")
        
        model = get_model(model_name)
        model.build()
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_test, y_test, label_names)
        results[model_name] = metrics
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-macro: {metrics['classification_report']['macro avg']['f1-score']:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1-macro':<12}")
    print("-" * 50)
    for name, res in results.items():
        acc = res['accuracy']
        f1 = res['classification_report']['macro avg']['f1-score']
        print(f"{name:<25} {acc:<12.4f} {f1:<12.4f}")


if __name__ == "__main__":
    main()
