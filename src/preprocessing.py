"""
Data Preprocessing Module for Vietnamese Movie Review Emotion Classification.

This module handles:
- Text cleaning and normalization
- Vietnamese word segmentation
- Data loading and splitting
"""

import re
import unicodedata
from typing import List, Tuple, Optional
import pandas as pd
from pathlib import Path


class TextPreprocessor:
    """Preprocessor for Vietnamese movie review text."""
    
    def __init__(self, use_word_segmentation: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            use_word_segmentation: Whether to use Vietnamese word segmentation
        """
        self.use_word_segmentation = use_word_segmentation
        self._segmenter = None
        
    @property
    def segmenter(self):
        """Lazy load word segmenter."""
        if self._segmenter is None and self.use_word_segmentation:
            try:
                from underthesea import word_tokenize
                self._segmenter = word_tokenize
            except ImportError:
                print("Warning: underthesea not installed. Using space-based tokenization.")
                self._segmenter = lambda x: x
        return self._segmenter
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode to NFKC form."""
        return unicodedata.normalize('NFKC', text)
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r'<[^>]+>', '', text)
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r'\S+@\S+\.\S+'
        return re.sub(email_pattern, '', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize multiple whitespaces to single space."""
        return ' '.join(text.split())
    
    def remove_special_chars(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters while optionally keeping punctuation."""
        if keep_punctuation:
            # Keep Vietnamese characters, digits, and common punctuation
            pattern = r'[^\w\s.,!?;:\-\'\"àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ]'
        else:
            pattern = r'[^\w\s]'
        return re.sub(pattern, ' ', text)
    
    def normalize_vietnamese_chars(self, text: str) -> str:
        """Normalize Vietnamese character variations."""
        # Some common normalization rules
        replacements = {
            'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',
            'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',
            'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""
        
        # Step 1: Unicode normalization
        text = self.normalize_unicode(text)
        
        # Step 2: Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Step 3: Remove URLs and emails
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        
        # Step 4: Normalize Vietnamese characters
        text = self.normalize_vietnamese_chars(text)
        
        # Step 5: Remove special characters
        text = self.remove_special_chars(text)
        
        # Step 6: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 7: Lowercase
        text = text.lower()
        
        return text.strip()
    
    def segment(self, text: str) -> str:
        """
        Perform Vietnamese word segmentation.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Word-segmented text
        """
        if self.use_word_segmentation and self.segmenter:
            return self.segmenter(text)
        return text


class DataLoader:
    """Data loader for movie review dataset."""
    
    def __init__(self, raw_data_path: str, preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize data loader.
        
        Args:
            raw_data_path: Path to raw CSV data
            preprocessor: TextPreprocessor instance
        """
        self.raw_data_path = Path(raw_data_path)
        self.preprocessor = preprocessor or TextPreprocessor()
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV."""
        df = pd.read_csv(self.raw_data_path)
        print(f"Loaded {len(df)} reviews")
        return df
    
    def clean_data(self, df: pd.DataFrame, min_text_length: int = 10) -> pd.DataFrame:
        """
        Clean and filter data.
        
        Args:
            df: Raw dataframe
            min_text_length: Minimum text length to keep
            
        Returns:
            Cleaned dataframe
        """
        # Fill NaN values
        df['review_text'] = df['review_text'].fillna('')
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['review_text'], keep='first')
        print(f"Removed {initial_count - len(df)} duplicates")
        
        # Filter by text length
        df = df[df['review_text'].str.len() >= min_text_length].copy()
        print(f"After filtering short reviews: {len(df)} reviews")
        
        return df.reset_index(drop=True)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to text column.
        
        Args:
            df: Dataframe with review_text column
            
        Returns:
            Dataframe with processed_text column
        """
        df = df.copy()
        df['processed_text'] = df['review_text'].apply(self.preprocessor.preprocess)
        return df
    
    def prepare_for_annotation(self, df: pd.DataFrame, output_path: str):
        """
        Prepare data for annotation.
        
        Args:
            df: Preprocessed dataframe
            output_path: Path to save annotation file
        """
        annotation_df = df[['movie_title', 'review_text', 'rating']].copy()
        annotation_df.insert(0, 'id', range(1, len(annotation_df) + 1))
        annotation_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved {len(annotation_df)} reviews to {output_path}")


def rating_to_label(rating: int) -> int:
    """
    Convert rating (1-10) to 5-class label for reference.
    
    This provides an initial label based on rating, but manual annotation
    should be the final source of truth.
    
    Args:
        rating: Rating from 1-10
        
    Returns:
        Label from 0-4 (Very Negative to Very Positive)
    """
    if rating <= 2:
        return 0  # Very Negative
    elif rating <= 4:
        return 1  # Negative
    elif rating <= 6:
        return 2  # Neutral
    elif rating <= 8:
        return 3  # Positive
    else:
        return 4  # Very Positive


def main():
    """Main function to test preprocessing."""
    import os
    
    # Paths
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / "crawler" / "data" / "raw" / "moveek_reviews.csv"
    output_path = project_root / "data" / "annotations" / "to_annotate.csv"
    processed_path = project_root / "data" / "processed_reviews.csv"
    
    # Initialize
    preprocessor = TextPreprocessor(use_word_segmentation=False)  # Disable for now
    loader = DataLoader(raw_data_path, preprocessor)
    
    # Process
    print("=" * 60)
    print("Loading raw data...")
    df = loader.load_raw_data()
    
    print("\nCleaning data...")
    df = loader.clean_data(df)
    
    print("\nPreprocessing text...")
    df = loader.preprocess_data(df)
    
    # Add reference labels from rating
    df['reference_label'] = df['rating'].apply(lambda x: rating_to_label(x) if pd.notna(x) else 2)
    
    # Show sample
    print("\n=== Sample Preprocessed Data ===")
    for i, row in df.head(5).iterrows():
        print(f"\nOriginal: {row['review_text'][:100]}...")
        print(f"Processed: {row['processed_text'][:100]}...")
        print(f"Rating: {row['rating']} -> Reference Label: {row['reference_label']}")
    
    # Save processed data
    df.to_csv(processed_path, index=False, encoding='utf-8')
    print(f"\nSaved processed data to {processed_path}")
    
    # Prepare for annotation
    print("\nPreparing annotation file...")
    loader.prepare_for_annotation(df, output_path)
    
    # Statistics
    print("\n=== Label Distribution (from rating) ===")
    label_names = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    for label_id, name in enumerate(label_names):
        count = (df['reference_label'] == label_id).sum()
        pct = count / len(df) * 100
        print(f"  {label_id} ({name}): {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
