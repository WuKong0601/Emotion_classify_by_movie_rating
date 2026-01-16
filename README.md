# Emotion Classification for Vietnamese Movie Reviews

Xây dựng dữ liệu và mô hình phân loại cảm xúc trong các bài đánh giá phim tiếng Việt.

## 📋 Project Overview

- **Dataset**: 3,648 reviews từ Moveek.com
- **Task**: 5-class emotion classification
- **Labels**: Very Negative, Negative, Neutral, Positive, Very Positive
- **Models**: Naive Bayes, SVM, Random Forest, Logistic Regression, LSTM, PhoBERT

## 📁 Project Structure

```
Emotion_research/
├── crawler/                    # Data crawler
│   ├── data/
│   │   ├── raw/               # Raw crawled data
│   │   └── processed/         # Processed data
│   ├── config.py              # Crawler configuration
│   └── moveek_crawler.py      # Moveek crawler
├── src/                        # Source code
│   ├── preprocessing.py       # Text preprocessing
│   ├── feature_extraction.py  # TF-IDF, PhoBERT features
│   ├── calculate_agreement.py # Inter-annotator agreement
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation metrics
│   └── models/
│       ├── traditional_ml.py  # NB, SVM, RF, LR
│       └── deep_learning.py   # LSTM, PhoBERT
├── data/                       # Data directory
│   └── annotations/           # Annotation files
├── experiments/                # Experiment results
│   └── results/
├── docs/                       # Documentation
│   ├── annotation_guidelines.md
│   └── paper1.md
└── requirements.txt           # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# For deep learning models
pip install torch transformers
pip install underthesea  # Vietnamese word segmentation
```

### 2. Preprocess Data

```bash
python src/preprocessing.py
```

This will:
- Clean and normalize text
- Create `data/processed_reviews.csv`
- Generate annotation file

### 3. Annotation Workflow

1. Read guidelines: `docs/annotation_guidelines.md`
2. Two annotators label data independently
3. Calculate agreement:
   ```bash
   python src/calculate_agreement.py
   ```
4. Resolve disagreements

### 4. Train Models

```bash
# Train all traditional ML models
python src/train.py --data data/processed_reviews.csv

# Train specific models
python src/train.py --models naive_bayes svm random_forest

# With specific features
python src/train.py --features tfidf --models svm
```

### 5. Evaluate

Results are saved to `experiments/results/training_results.json`

### 6. Compare Models on Development Set

So sánh các mô hình đã train trên tập validation (development set):

```bash
# So sánh tất cả models có sẵn
python src/compare_models_dev.py

# So sánh các models cụ thể
python src/compare_models_dev.py --models naive_bayes svm logistic_regression

# Chỉ định đường dẫn data và models
python src/compare_models_dev.py --data data/labeled_reviews.csv --models-dir experiments/results --output experiments/comparison
```

Kết quả sẽ được lưu trong `experiments/comparison/`:
- `dev_set_comparison.json`: Kết quả chi tiết dạng JSON
- `dev_set_comparison.csv`: Bảng so sánh dạng CSV
- `dev_set_comparison.tex`: Bảng LaTeX cho paper
- `dev_f1_comparison.png`: Biểu đồ so sánh F1-score
- `dev_per_class_f1.png`: Biểu đồ F1-score theo từng class
- `dev_confusion_matrices.png`: Confusion matrices cho tất cả models

## 📊 Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-macro**: Unweighted mean of per-class F1
- **F1-weighted**: Weighted mean by class support

## 📝 Label Schema

| ID | Label | Vietnamese | Rating Range |
|----|-------|------------|--------------|
| 0 | Very Negative | Rất tiêu cực | 1-2 |
| 1 | Negative | Tiêu cực | 3-4 |
| 2 | Neutral | Trung lập | 5-6 |
| 3 | Positive | Tích cực | 7-8 |
| 4 | Very Positive | Rất tích cực | 9-10 |

## 📄 Paper

Paper follows ACL format. See `docs/paper1.md` for draft.

## 📚 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- transformers, torch (for deep learning)
- underthesea (for Vietnamese NLP)

## 👥 Authors

[Your Name]

## 📜 License

MIT License
