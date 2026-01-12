import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

print("🚀 Preparing film data for quantum analysis...")

# Load your data
df = pd.read_csv('inkrealm_clean.csv')
print(f"📊 Loaded {len(df)} film titles")

# Step 1: Basic text features
df['title_length'] = df['TITLE'].str.len()
df['word_count'] = df['TITLE'].str.split().str.len()
df['has_year'] = df['TITLE'].str.contains(r'\d{4}').astype(int)

print("✅ Added basic features: title_length, word_count, has_year")

# Step 2: Bag-of-words encoding
try:
    vectorizer = CountVectorizer(max_features=20, stop_words='english')
    title_matrix = vectorizer.fit_transform(df['TITLE']).toarray()
    print(f"📝 Created text features from {title_matrix.shape[1]} common words")
except Exception as e:
    print(f"⚠️ Could not create text features: {e}")
    title_matrix = np.zeros((len(df), 20))

# Combine features
X_basic = df[['title_length', 'word_count', 'has_year']].values
X = np.hstack([X_basic, title_matrix])
print(f"🧮 Combined feature matrix shape: {X.shape}")

# Step 3: Create a simple classification task
df['contains_love'] = df['TITLE'].str.contains('love', case=False).astype(int)
y = df['contains_love'].values
print(f"🎯 Target: {sum(y)} titles contain 'love' ({(sum(y)/len(y)*100):.1f}%)")

# Save for quantum analysis
np.save('film_features.npy', X)
np.save('film_labels.npy', y)
df.to_csv('enriched_films.csv', index=False)

print("\n✅ Data preparation complete!")
print("   - Saved: film_features.npy (features for QNN)")
print("   - Saved: film_labels.npy (classification targets)")
print("   - Saved: enriched_films.csv (enhanced dataset)")
