"""
PhoBERT Training Script for 3-Class Sentiment Classification.

Labels:
- 0: Negative
- 1: Neutral  
- 2: Positive
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# 3-Class Configuration
LABEL_NAMES = ['Negative', 'Neutral', 'Positive']
NUM_CLASSES = 3
MODEL_NAME = "vinai/phobert-base"


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class WeightedTrainer(Trainer):
    """Trainer with class weights for imbalanced data."""
    
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    f1_weighted = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )[2]
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall
    }


def train_phobert(
    data_path: str = "data/labeled_reviews.csv",
    output_dir: str = "experiments/phobert_results",
    max_length: int = 256,
    batch_size: int = 16,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    seed: int = 42
):
    print("=" * 60)
    print("PHOBERT 3-CLASS TRAINING")
    print("=" * 60)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print(f"\nLoading {data_path}...")
    df = pd.read_csv(data_path)
    
    text_col = 'processed_text' if 'processed_text' in df.columns else 'review_text'
    df = df[df[text_col].notna() & df['final_label'].notna()]
    
    texts = df[text_col].tolist()
    labels = df['final_label'].astype(int).tolist()
    
    print(f"Samples: {len(texts)}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {labels.count(i)} ({labels.count(i)/len(labels)*100:.1f}%)")
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {dict(zip(LABEL_NAMES, class_weights.cpu().numpy()))}")
    
    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
    model.to(device)
    
    # Datasets
    train_dataset = EmotionDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = EmotionDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = EmotionDataset(X_test, y_test, tokenizer, max_length)
    
    # Training
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available()
    )
    
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    trainer.train()
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    test_results = trainer.evaluate(test_dataset)
    for k, v in test_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, digits=4))
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Save
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': MODEL_NAME,
        'num_classes': NUM_CLASSES,
        'test_metrics': {
            'accuracy': float(test_results['eval_accuracy']),
            'f1_macro': float(test_results['eval_f1_macro']),
            'f1_weighted': float(test_results['eval_f1_weighted'])
        },
        'classification_report': classification_report(y_test, y_pred, target_names=LABEL_NAMES, output_dict=True),
        'confusion_matrix': cm.tolist()
    }
    
    with open(output_path / "phobert_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    trainer.save_model(str(output_path / "model"))
    tokenizer.save_pretrained(str(output_path / "model"))
    
    print(f"\n✓ Results saved to {output_path}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/labeled_reviews.csv")
    parser.add_argument("--output", default="experiments/phobert_results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    
    train_phobert(
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
