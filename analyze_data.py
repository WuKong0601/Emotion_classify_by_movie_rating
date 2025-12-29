"""Analyze crawled movie review data."""
import pandas as pd

# Load data
df = pd.read_csv('crawler/data/raw/moveek_reviews.csv')

print("=" * 60)
print("DATA ANALYSIS")
print("=" * 60)

print(f"\nTotal rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\n=== Rating Distribution ===")
print(df['rating'].value_counts().sort_index())
print(f"\nRatings with NaN: {df['rating'].isna().sum()}")

print("\n=== Reviews with Content ===")
df['text_len'] = df['review_text'].fillna('').str.len()
valid = df[df['text_len'] > 10]
print(f"Reviews with text (>10 chars): {len(valid)}")

print("\n=== Sample Reviews ===")
for i, row in valid.head(15).iterrows():
    text = str(row['review_text'])[:120].replace('\n', ' ')
    print(f"Rating: {row['rating']:>5} | {row['movie_title'][:25]:<25} | {text}...")

print("\n=== Text Length Statistics ===")
print(valid['text_len'].describe())

print("\n=== Movies with Most Reviews ===")
print(df.groupby('movie_title').size().sort_values(ascending=False).head(20))
