# Task 2: TF-IDF Vectorizer for Food Delivery Reviews

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------
# 1. Create a corpus of 8 food delivery reviews
# (Positive, Negative, Neutral)
# ---------------------------------------------------

reviews = [
    "The pizza was delicious and delivered hot.",                 # Positive
    "Burger arrived cold and fries were soggy.",                  # Negative
    "The biryani was average and nothing special.",               # Neutral
    "Amazing pasta with fresh vegetables and great taste.",       # Positive
    "Delivery was late and the food was cold.",                   # Negative
    "Loved the butter chicken and garlic naan.",                  # Positive
    "The app was easy to use and delivery was okay.",             # Neutral
    "Dessert was tasty but the packaging was damaged."            # Neutral
]

# ---------------------------------------------------
# 2. Transform the corpus into a TF-IDF matrix
# ---------------------------------------------------

vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(reviews)

feature_names = vectorizer.get_feature_names_out()

# ---------------------------------------------------
# 3. Display TF-IDF Matrix as a Pandas DataFrame
# ---------------------------------------------------

review_labels = [f"Review {i+1}" for i in range(len(reviews))]

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    index=review_labels,
    columns=feature_names
)

print("=" * 80)
print("FULL TF-IDF MATRIX")
print("=" * 80)
print(tfidf_df.round(2))

# ---------------------------------------------------
# 4. Print Top 3 TF-IDF Words for Each Review
# ---------------------------------------------------

print("\n" + "=" * 80)
print("TOP 3 WORDS FOR EACH REVIEW")
print("=" * 80)

for i in range(len(reviews)):

    row = tfidf_df.iloc[i]

    top3 = row.sort_values(ascending=False).head(3)

    print(f"\n{review_labels[i]}")
    print("Review:", reviews[i])

    for word, score in top3.items():
        print(f"{word:<12} : {score:.2f}")