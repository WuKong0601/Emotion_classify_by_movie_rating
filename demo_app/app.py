"""
Demo Web Application for Vietnamese Movie Review Sentiment Classification.

This Flask app demonstrates the emotion classification models with:
- Movie listing interface
- Review submission form with rating
- Model selection dropdown
- Real-time sentiment prediction
"""

import os
import sys
import json
import pickle
import re
import unicodedata
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "experiments" / "results"
DATA_DIR = BASE_DIR

# Label configuration
LABEL_NAMES = ['Negative', 'Neutral', 'Positive']
LABEL_COLORS = {
    'Negative': '#e74c3c',
    'Neutral': '#95a5a6', 
    'Positive': '#27ae60'
}
LABEL_EMOJI = {
    'Negative': '😞',
    'Neutral': '😐',
    'Positive': '😊'
}

# Global model storage
models = {}
vectorizer = None


class TextPreprocessor:
    """Simplified text preprocessor for Vietnamese text."""
    
    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize('NFKC', text)
    
    def remove_html_tags(self, text: str) -> str:
        return re.sub(r'<[^>]+>', '', text)
    
    def remove_urls(self, text: str) -> str:
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    
    def normalize_whitespace(self, text: str) -> str:
        return ' '.join(text.split())
    
    def remove_special_chars(self, text: str) -> str:
        pattern = r'[^\w\s.,!?;:\-\'\"àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ]'
        return re.sub(pattern, ' ', text)
    
    def normalize_vietnamese_chars(self, text: str) -> str:
        replacements = {
            'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',
            'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',
            'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def preprocess(self, text: str) -> str:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""
        
        text = self.normalize_unicode(text)
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.normalize_vietnamese_chars(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        text = text.lower()
        
        return text.strip()


preprocessor = TextPreprocessor()


def load_models():
    """Load all trained models and vectorizer."""
    global models, vectorizer
    
    model_files = {
        'naive_bayes': 'naive_bayes_model.pkl',
        'svm': 'svm_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'logistic_regression': 'logistic_regression_model.pkl'
    }
    
    for name, filename in model_files.items():
        model_path = MODELS_DIR / filename
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    models[name] = data['model']
                    print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
    
    # Load or create TF-IDF vectorizer
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    if vectorizer_path.exists():
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            print(f"✓ Loaded TF-IDF vectorizer ({len(vectorizer.vocabulary_)} features)")
    else:
        # Create new vectorizer and fit on training data
        # IMPORTANT: Must use same split as training to get same features!
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        import pandas as pd
        
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Load training data to fit vectorizer
        data_path = PROJECT_ROOT / "data" / "labeled_reviews.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            
            # Use processed_text column if exists
            text_col = 'processed_text' if 'processed_text' in df.columns else 'review_text'
            df = df[df[text_col].notna() & df['final_label'].notna()]
            
            texts = df[text_col].fillna('').tolist()
            labels = df['final_label'].astype(int).tolist()
            
            # Split EXACTLY like train.py: 15% test, 15% val = 70% train
            X_temp, _, y_temp, _ = train_test_split(
                texts, labels,
                test_size=0.15,
                random_state=42,
                stratify=labels
            )
            val_ratio = 0.15 / 0.85
            X_train, _, _, _ = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio,
                random_state=42,
                stratify=y_temp
            )
            
            vectorizer.fit(X_train)
            
            # Save for future use
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            print(f"✓ Created and saved TF-IDF vectorizer ({len(vectorizer.vocabulary_)} features)")
        else:
            print("✗ Training data not found for vectorizer")


def load_movies():
    """Load sample movies data."""
    movies_path = DATA_DIR / "sample_movies.json"
    if movies_path.exists():
        with open(movies_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def predict_sentiment(text: str, model_name: str):
    """Predict sentiment for given text using specified model."""
    if model_name not in models:
        return None, None, "Model not found"
    
    if vectorizer is None:
        return None, None, "Vectorizer not loaded"
    
    # Preprocess text
    processed_text = preprocessor.preprocess(text)
    
    if len(processed_text) == 0:
        return None, None, "Text is empty after preprocessing"
    
    # Transform to features
    features = vectorizer.transform([processed_text])
    
    # Predict
    model = models[model_name]
    prediction = model.predict(features)[0]
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(features)[0]
        except:
            pass
    
    return prediction, probabilities, None


@app.route('/')
def index():
    """Main page with movie listing and review form."""
    movies = load_movies()
    available_models = list(models.keys())
    return render_template('index.html', 
                         movies=movies, 
                         models=available_models,
                         label_names=LABEL_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for sentiment prediction."""
    data = request.json
    
    review_text = data.get('review', '')
    model_name = data.get('model', 'svm')
    movie_title = data.get('movie', '')
    rating = data.get('rating', 5)
    
    if not review_text.strip():
        return jsonify({
            'success': False,
            'error': 'Vui lòng nhập nội dung đánh giá!'
        })
    
    prediction, probabilities, error = predict_sentiment(review_text, model_name)
    
    if error:
        return jsonify({
            'success': False,
            'error': error
        })
    
    label = LABEL_NAMES[prediction]
    
    result = {
        'success': True,
        'prediction': int(prediction),
        'label': label,
        'color': LABEL_COLORS[label],
        'emoji': LABEL_EMOJI[label],
        'movie': movie_title,
        'rating': rating,
        'model': model_name,
        'processed_text': preprocessor.preprocess(review_text)
    }
    
    if probabilities is not None:
        result['probabilities'] = {
            LABEL_NAMES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probabilities)
        }
    
    return jsonify(result)


@app.route('/api/models')
def get_models():
    """Get list of available models."""
    return jsonify({
        'models': list(models.keys()),
        'labels': LABEL_NAMES
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🎬 Vietnamese Movie Review Sentiment Demo")
    print("=" * 60)
    
    # Load models
    print("\nLoading models...")
    load_models()
    
    print(f"\n✓ Loaded {len(models)} models: {list(models.keys())}")
    print("\n" + "=" * 60)
    print("Starting server at http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, port=5000)
