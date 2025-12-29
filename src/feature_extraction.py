"""
Feature Extraction Module for Emotion Classification.

Supports:
- TF-IDF features
- Word embeddings (fastText, PhoBERT)
- Combined features
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TFIDFExtractor:
    """TF-IDF feature extractor."""
    
    def __init__(self, 
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.95):
        """
        Initialize TF-IDF extractor.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams (min, max)
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True
        )
        self.is_fitted = False
        
    def fit(self, texts: List[str]):
        """Fit on training data."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF features."""
        if not self.is_fitted:
            raise ValueError("Extractor not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        result = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.vectorizer.get_feature_names_out().tolist()
    
    def save(self, path: str):
        """Save fitted vectorizer."""
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load(self, path: str):
        """Load fitted vectorizer."""
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True


class BOWExtractor:
    """Bag of Words feature extractor."""
    
    def __init__(self,
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 1),
                 binary: bool = False):
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            binary=binary
        )
        self.is_fitted = False
        
    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Extractor not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        result = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return result


class PhoBERTExtractor:
    """PhoBERT embedding extractor using transformers."""
    
    def __init__(self, model_name: str = "vinai/phobert-base", max_length: int = 256):
        """
        Initialize PhoBERT extractor.
        
        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        
    def load_model(self):
        """Load PhoBERT model and tokenizer."""
        if self._is_loaded:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
            self._is_loaded = True
            print(f"Model loaded on {self.device}")
            
        except ImportError:
            raise ImportError("transformers and torch required. Install with: pip install transformers torch")
    
    def extract(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings from texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for inference
            
        Returns:
            Numpy array of shape (n_samples, hidden_size)
        """
        import torch
        
        self.load_model()
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use [CLS] token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)


class FeatureExtractor:
    """Combined feature extractor supporting multiple methods."""
    
    def __init__(self, method: str = 'tfidf', **kwargs):
        """
        Initialize feature extractor.
        
        Args:
            method: Feature extraction method ('tfidf', 'bow', 'phobert')
            **kwargs: Additional arguments for specific extractor
        """
        self.method = method
        
        if method == 'tfidf':
            self.extractor = TFIDFExtractor(**kwargs)
        elif method == 'bow':
            self.extractor = BOWExtractor(**kwargs)
        elif method == 'phobert':
            self.extractor = PhoBERTExtractor(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts."""
        if self.method in ['tfidf', 'bow']:
            return self.extractor.fit_transform(texts)
        else:
            return self.extractor.extract(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts."""
        if self.method in ['tfidf', 'bow']:
            return self.extractor.transform(texts)
        else:
            return self.extractor.extract(texts)


def main():
    """Test feature extraction."""
    # Sample texts
    texts = [
        "phim hay quá đỉnh luôn",
        "không thích phim này dở quá",
        "bình thường không có gì đặc biệt",
        "tuyệt vời xuất sắc recommend",
        "thất vọng không như kỳ vọng"
    ]
    
    print("=" * 60)
    print("Feature Extraction Test")
    print("=" * 60)
    
    # Test TF-IDF
    print("\n=== TF-IDF ===")
    tfidf = TFIDFExtractor(max_features=100)
    features = tfidf.fit_transform(texts)
    print(f"Feature shape: {features.shape}")
    print(f"Top features: {tfidf.get_feature_names()[:10]}")
    
    # Test BOW
    print("\n=== Bag of Words ===")
    bow = BOWExtractor(max_features=100)
    features = bow.fit_transform(texts)
    print(f"Feature shape: {features.shape}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
