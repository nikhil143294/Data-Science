# Food Delivery Sentiment Analysis using Logistic Regression

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords (only first time)
nltk.download("stopwords")

# -----------------------------------------
# Preprocessing Function
# -----------------------------------------

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)

# -----------------------------------------
# Dataset (20 Reviews)
# -----------------------------------------

reviews = [

("The pizza was delicious and hot", "Positive"),
("Amazing burger and fries", "Positive"),
("Loved the butter chicken", "Positive"),
("Delivery was very fast", "Positive"),
("Food quality was excellent", "Positive"),
("Fresh sushi and tasty food", "Positive"),
("Best pasta ever", "Positive"),
("Dessert was amazing", "Positive"),
("Everything arrived on time", "Positive"),
("Great customer service", "Positive"),

("Food arrived cold", "Negative"),
("Delivery was two hours late", "Negative"),
("Burger was soggy", "Negative"),
("Chicken was undercooked", "Negative"),
("App crashed during payment", "Negative"),
("Order never arrived", "Negative"),
("Pizza was burnt", "Negative"),
("Packaging was damaged", "Negative"),
("Very disappointing service", "Negative"),
("Food had no taste", "Negative")

]

# -----------------------------------------
# Prepare Data
# -----------------------------------------

texts = [preprocess(review) for review, label in reviews]
labels = [label for review, label in reviews]

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42
)

# Logistic Regression Model
model = LogisticRegression()

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)

print("="*50)
print("Model Accuracy:", round(accuracy_score(y_test, y_pred)*100,2), "%")
print("="*50)

# -----------------------------------------
# Predict 3 New Reviews
# -----------------------------------------

print("\nEnter 3 New Reviews")

for i in range(3):

    review = input(f"\nReview {i+1}: ")

    review = preprocess(review)

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)

    print("Predicted Sentiment:", prediction[0])