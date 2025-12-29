"""
Deep Learning Models for Emotion Classification.

This module implements:
- LSTM / BiLSTM
- PhoBERT fine-tuning

Requires: torch, transformers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Deep learning models disabled.")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import Trainer, TrainingArguments
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. PhoBERT model disabled.")


# ===================== Dataset Classes =====================

if TORCH_AVAILABLE:
    class TextDataset(Dataset):
        """Dataset for text classification."""
        
        def __init__(self, texts: List[str], labels: List[int], 
                     tokenizer, max_length: int = 256):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            }


    class LSTMClassifier(nn.Module):
        """LSTM-based text classifier."""
        
        def __init__(self, 
                     vocab_size: int,
                     embedding_dim: int = 128,
                     hidden_dim: int = 256,
                     num_layers: int = 2,
                     num_classes: int = 5,
                     bidirectional: bool = True,
                     dropout: float = 0.3):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
            
            lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(lstm_output_dim, num_classes)
            
        def forward(self, input_ids, attention_mask=None):
            # Embedding
            embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
            
            # LSTM
            lstm_out, (hidden, cell) = self.lstm(embedded)
            
            # Use last hidden state
            if self.lstm.bidirectional:
                hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            else:
                hidden = hidden[-1,:,:]
            
            # Classification
            out = self.dropout(hidden)
            out = self.fc(out)
            
            return out


# ===================== PhoBERT Model =====================

class PhoBERTClassifier:
    """PhoBERT fine-tuning for Vietnamese text classification."""
    
    def __init__(self, 
                 model_name: str = "vinai/phobert-base",
                 num_labels: int = 5,
                 max_length: int = 256):
        """
        Initialize PhoBERT classifier.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of output classes
            max_length: Maximum sequence length
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers")
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model(self):
        """Load pre-trained model and tokenizer."""
        print(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        print("Model loaded.")
        
    def prepare_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        """Prepare dataset for training."""
        if self.tokenizer is None:
            self.load_model()
        return TextDataset(texts, labels, self.tokenizer, self.max_length)
    
    def train(self, 
              train_texts: List[str], 
              train_labels: List[int],
              val_texts: List[str] = None,
              val_labels: List[int] = None,
              output_dir: str = "experiments/phobert",
              num_epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              warmup_steps: int = 500):
        """
        Fine-tune PhoBERT on training data.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            output_dir: Output directory for checkpoints
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler
        """
        if self.model is None:
            self.load_model()
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        val_dataset = None
        if val_texts and val_labels:
            val_dataset = self.prepare_dataset(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            learning_rate=learning_rate,
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Train
        print("Starting training...")
        self.trainer.train()
        print("Training complete.")
        
    def predict(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for texts.
        
        Args:
            texts: Input texts
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")
        
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
        all_preds = []
        all_probs = []
        
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
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)
    
    def save(self, path: str):
        """Save model and tokenizer."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load saved model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        print(f"Model loaded from {path}")


def main():
    """Test deep learning models."""
    print("=" * 60)
    print("Deep Learning Models Test")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping tests.")
        return
    
    # Test PhoBERT initialization
    if TRANSFORMERS_AVAILABLE:
        print("\n=== PhoBERT ===")
        print("PhoBERT classifier available.")
        print("To use: classifier = PhoBERTClassifier()")
        print("        classifier.train(train_texts, train_labels)")
    else:
        print("\nTransformers not available for PhoBERT.")
    
    # Test LSTM
    print("\n=== LSTM ===")
    model = LSTMClassifier(vocab_size=10000, num_classes=5)
    print(f"LSTM model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Dummy forward pass
    dummy_input = torch.randint(0, 10000, (4, 100))  # batch=4, seq_len=100
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be (4, 5)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
