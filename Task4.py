# Task 4 - Sentiment Analysis Model Comparison

import pandas as pd
from review_preprocessor import preprocess_review

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------------------------------------------------------
# 1. Labelled Dataset (30 Reviews)
# --------------------------------------------------------

reviews = [

# Positive Reviews (15)

("The pizza was delicious and hot.", "Positive"),
("Amazing burger and crispy fries.", "Positive"),
("Loved the butter chicken.", "Positive"),
("Delivery was very fast.", "Positive"),
("Fresh sushi and excellent taste.", "Positive"),
("Food quality was outstanding.", "Positive"),
("The dessert was amazing.", "Positive"),
("Everything arrived on time.", "Positive"),
("Chicken biryani was flavorful.", "Positive"),
("Great packaging and fresh food.", "Positive"),
("The app was easy to use.", "Positive"),
("Fantastic customer service.", "Positive"),
("Best pasta ever.", "Positive"),
("Loved the garlic naan.", "Positive"),
("Highly recommend this restaurant.", "Positive"),

# Negative Reviews (15)

("Food arrived cold.", "Negative"),
("Delivery was two hours late.", "Negative"),
("Burger was soggy.", "Negative"),
("Chicken was undercooked.", "Negative"),
("Soup tasted bad.", "Negative"),
("App crashed during payment.", "Negative"),
("Payment failed.", "Negative"),
("Order never arrived.", "Negative"),
("Pizza was burnt.", "Negative"),
("Packaging was damaged.", "Negative"),
("Very poor customer service.", "Negative"),
("Rice was overcooked.", "Negative"),
("Food had no taste.", "Negative"),
("Driver delivered to wrong address.", "Negative"),
("Very disappointing experience.", "Negative")

]

# --------------------------------------------------------
# 2. Create DataFrame
# --------------------------------------------------------

df = pd.DataFrame(reviews, columns=["Review", "Sentiment"])

# Preprocess Reviews using Task 1
df["Processed"] = df["Review"].apply(preprocess_review)

# --------------------------------------------------------
# 3. Train-Test Split (80% Train / 20% Test)
# --------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["Processed"],
    df["Sentiment"],
    test_size=0.20,
    random_state=42,
    stratify=df["Sentiment"]
)

# --------------------------------------------------------
# 4. Build Two Pipelines
# --------------------------------------------------------

nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# --------------------------------------------------------
# 5. Train Both Models
# --------------------------------------------------------

nb_pipeline.fit(X_train, y_train)
lr_pipeline.fit(X_train, y_train)

# Predictions
nb_pred = nb_pipeline.predict(X_test)
lr_pred = lr_pipeline.predict(X_test)

# --------------------------------------------------------
# 6. Evaluation Function
# --------------------------------------------------------

def evaluate_model(y_true, y_pred):

    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, pos_label="Positive", zero_division=0),
        recall_score(y_true, y_pred, pos_label="Positive", zero_division=0),
        f1_score(y_true, y_pred, pos_label="Positive", zero_division=0)
    ]

nb_results = evaluate_model(y_test, nb_pred)
lr_results = evaluate_model(y_test, lr_pred)

# --------------------------------------------------------
# 7. Side-by-Side Comparison Table
# --------------------------------------------------------

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "MultinomialNB": nb_results,
    "LogisticRegression": lr_results
})

print("=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)
print(comparison.round(2))

# --------------------------------------------------------
# 8. Deployment Decision
# --------------------------------------------------------

# I would deploy Logistic Regression because it generally provides better overall performance and generalises better on TF-IDF text features.

if lr_results[3] >= nb_results[3]:
    print("\nRecommended Model: Logistic Regression")
else:
    print("\nRecommended Model: Multinomial Naive Bayes")