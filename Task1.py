# Food Review Text Preprocessor

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resource
nltk.download('stopwords')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# -----------------------------------------
# Text Preprocessing Function
# -----------------------------------------
def preprocess_review(review):

    # 1. Convert to lowercase
    review = review.lower()

    # 2. Remove punctuation and numbers
    review = re.sub(r'[^a-z\s]', '', review)

    # 3. Remove extra whitespace
    review = re.sub(r'\s+', ' ', review).strip()

    # 4. Tokenize
    words = review.split()

    # 5. Remove stopwords
    words = [word for word in words if word not in stop_words]

    # 6. Apply stemming
    words = [stemmer.stem(word) for word in words]

    # 7. Return processed string
    return " ".join(words)

# -----------------------------------------
# Test the Function
# -----------------------------------------

review1 = "The food arrived COLD!!! and 45 minutes late."
review2 = "Absolutely loved the butter chicken. Delivery was super fast!"
review3 = "The app crashed while placing my order, very disappointing."

reviews = [review1, review2, review3]

for i, review in enumerate(reviews, start=1):
    print("=" * 60)
    print(f"Review {i}")
    print("Original Review:")
    print(review)
    print("\nProcessed Review:")
    print(preprocess_review(review))