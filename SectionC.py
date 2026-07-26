import sys
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from review_preprocessor import preprocess_review 
# Minimum number of training samples required for each class
MIN_SAMPLES_PER_CLASS = 2

SENTIMENT_TRAINING_DATA = [
    ("The food arrived hot and fresh, absolutely loved the flavors!", "Positive"),
    ("Best butter chicken I've had in months, so creamy and rich.", "Positive"),
    ("Delivery was super fast and the driver was really friendly.", "Positive"),
    ("Everything was packed neatly and the portions were generous.", "Positive"),
    ("Amazing pizza, the crust was perfectly crispy and cheese was fresh.", "Positive"),
    ("Great experience overall, will definitely order again soon.", "Positive"),
    ("The sushi was fresh and beautifully presented, highly recommend.", "Positive"),
    ("Loved the spicy ramen, the broth was so flavorful.", "Positive"),
    ("Excellent service, my order arrived earlier than expected.", "Positive"),
    ("The tacos were delicious, fresh salsa and warm tortillas.", "Positive"),
    ("Fantastic biryani, perfectly spiced and the rice was fluffy.", "Positive"),
    ("The app made ordering so easy and checkout was smooth.", "Positive"),
    ("Food quality was outstanding, definitely worth the price.", "Positive"),
    ("Really impressed with the packaging, nothing spilled at all.", "Positive"),
    ("The dessert was heavenly, brownie was warm and gooey.", "Positive"),
    ("The food arrived cold and the packaging had leaked everywhere.", "Negative"),
    ("Terrible experience, my order was over an hour late.", "Negative"),
    ("The burger was soggy and completely fell apart.", "Negative"),
    ("Delivery driver couldn't find my address and cancelled the order.", "Negative"),
    ("Pizza was cold and the cheese was hard as rubber.", "Negative"),
    ("Very disappointing meal, the chicken tasted undercooked.", "Negative"),
    ("The app crashed twice and I couldn't complete my order.", "Negative"),
    ("Portion size was tiny and way overpriced for what we got.", "Negative"),
    ("Food was bland and clearly not fresh at all.", "Negative"),
    ("Worst delivery experience, the driver was extremely rude.", "Negative"),
    ("The soup was lukewarm and tasted completely watered down.", "Negative"),
    ("My payment failed three times but I was still charged.", "Negative"),
    ("Sushi rice was mushy and the fish smelled off.", "Negative"),
    ("Order was missing half the items I paid for.", "Negative"),
    ("Never ordering from here again, absolutely awful service.", "Negative"),
]

CATEGORY_TRAINING_DATA = [
    # ---- Delivery ----
    ("The delivery guy took over 90 minutes and never called me.", "Delivery"),
    ("My order arrived 2 hours late and the driver couldn't find my address.", "Delivery"),
    ("Rider left the food at the wrong door, I never received my package.", "Delivery"),
    ("Delivery was supposed to arrive by 8pm but showed up at 10pm.", "Delivery"),
    ("The courier cancelled my order without any notice or reason.", "Delivery"),
    ("Tracking said out for delivery for three hours with no update.", "Delivery"),
    ("Driver dropped off someone else's order at my house by mistake.", "Delivery"),
    ("Extremely slow delivery, the estimated time was way off again.", "Delivery"),
    # ---- Food Quality ----
    ("The pizza was cold and the cheese had completely hardened.", "Food Quality"),
    ("My burger was soggy and the bun fell apart immediately.", "Food Quality"),
    ("The chicken tasted undercooked and honestly a bit off.", "Food Quality"),
    ("Food was bland and clearly not fresh, very disappointing meal.", "Food Quality"),
    ("The salad had wilted lettuce and tasted stale.", "Food Quality"),
    ("Portion size was tiny and the curry had almost no flavor.", "Food Quality"),
    ("Sushi rice was mushy and the fish smelled off.", "Food Quality"),
    ("The soup arrived lukewarm and tasted watered down.", "Food Quality"),
    # ---- App ----
    ("The app crashed twice while I was trying to check out.", "App"),
    ("I could not apply my discount code, the app kept throwing an error.", "App"),
    ("Payment failed three times but the app still charged my card.", "App"),
    ("The app froze on the order tracking screen and wouldn't refresh.", "App"),
    ("Login page keeps timing out, I cannot even open the app.", "App"),
    ("The app showed the wrong restaurant menu and prices were incorrect.", "App"),
    ("Notifications from the app are delayed by over an hour.", "App"),
    ("The search feature in the app never returns any results.", "App"),
    # ---- General ----
    ("Great service overall, nothing to complain about today.", "General"),
    ("Just wanted to say thanks for the quick response from support.", "General"),
    ("Overall experience was fine, no specific issues to report.", "General"),
    ("The restaurant selection available in my area is excellent.", "General"),
    ("Customer support was very helpful and polite on the phone.", "General"),
    ("I have a general question about my loyalty rewards account.", "General"),
    ("Everything was fine this time, no complaints at all.", "General"),
    ("Really appreciate the ongoing discounts and rewards program.", "General"),
]


# ======================================================================
# Classifier bootstrap
# ======================================================================
def build_classifier(training_data, model, label_name):
    """
    Train a TF-IDF + classifier Pipeline on (text, label) pairs.
    Returns (pipeline, ready, message). If any class has fewer than
    MIN_SAMPLES_PER_CLASS samples, training is skipped and ready=False.
    """
    texts, labels = zip(*training_data)
    counts = Counter(labels)
    under_min = {cls: n for cls, n in counts.items() if n < MIN_SAMPLES_PER_CLASS}

    if under_min:
        msg = (
            f"[{label_name} classifier] Not enough training samples for: "
            f"{under_min}. Need at least {MIN_SAMPLES_PER_CLASS} per class."
        )
        return None, False, msg

    processed_texts = [preprocess_review(t) for t in texts]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", model),
    ])
    pipeline.fit(processed_texts, labels)
    return pipeline, True, f"[{label_name} classifier] Trained on {len(texts)} samples."


sentiment_model, sentiment_ready, sentiment_msg = build_classifier(
    SENTIMENT_TRAINING_DATA, LogisticRegression(random_state=42, max_iter=1000), "Sentiment"
)
category_model, category_ready, category_msg = build_classifier(
    CATEGORY_TRAINING_DATA, MultinomialNB(), "Category"
)


# ======================================================================
# In-memory storage
# ======================================================================
# Every review added by staff: {"raw": ..., "processed": ...}
stored_reviews = []

# Every review that has been classified: {"raw", "processed", "sentiment", "category"}
classified_reviews = []


# ======================================================================
# Core actions
# ======================================================================
def add_review():
    raw_text = input("\nEnter the new customer review text: ").strip()
    if not raw_text:
        print("Empty review ignored -- nothing was added.")
        return

    processed = preprocess_review(raw_text)
    stored_reviews.append({"raw": raw_text, "processed": processed})
    print(f"Review added. (Total reviews stored: {len(stored_reviews)})")
    if not processed:
        print("Note: after preprocessing this review contained no content words "
              "(it may have been only stopwords/punctuation).")


def classify_review():
    if not sentiment_ready or not category_ready:
        print("\nCannot classify reviews right now:")
        if not sentiment_ready:
            print(f"  - {sentiment_msg}")
        if not category_ready:
            print(f"  - {category_msg}")
        print("Add more labelled training examples per class and restart the program.")
        return

    if stored_reviews:
        choice = input(
            "\nClassify (n)ew text or an (e)xisting stored review? [n/e]: "
        ).strip().lower()
    else:
        choice = "n"

    if choice == "e" and stored_reviews:
        for i, r in enumerate(stored_reviews):
            print(f"  [{i}] {r['raw']}")
        try:
            idx = int(input("Enter the review number to classify: ").strip())
            entry = stored_reviews[idx]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return
        raw_text, processed = entry["raw"], entry["processed"]
    else:
        raw_text = input("Enter the review text to classify: ").strip()
        if not raw_text:
            print("Empty review -- nothing to classify.")
            return
        processed = preprocess_review(raw_text)
        stored_reviews.append({"raw": raw_text, "processed": processed})

    if not processed:
        print("\nThis review has no usable content words after preprocessing "
              "(e.g. only stopwords/punctuation) -- cannot classify reliably.")
        return

    predicted_sentiment = sentiment_model.predict([processed])[0]
    predicted_category = category_model.predict([processed])[0]

    classified_reviews.append({
        "raw": raw_text,
        "processed": processed,
        "sentiment": predicted_sentiment,
        "category": predicted_category,
    })

    print(f"\nReview: \"{raw_text}\"")
    print(f"  Predicted Sentiment : {predicted_sentiment}")
    print(f"  Predicted Category  : {predicted_category}")


def view_summary():
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    print(f"Total reviews added        : {len(stored_reviews)}")
    print(f"Total reviews classified   : {len(classified_reviews)}")

    if classified_reviews:
        sentiment_counts = Counter(r["sentiment"] for r in classified_reviews)
        category_counts = Counter(r["category"] for r in classified_reviews)

        print("\nSentiment breakdown (classified reviews):")
        for label in ["Positive", "Negative"]:
            print(f"  {label:<10}: {sentiment_counts.get(label, 0)}")

        print("\nIssue category breakdown (classified reviews):")
        for label in ["Delivery", "Food Quality", "App", "General"]:
            print(f"  {label:<14}: {category_counts.get(label, 0)}")
    else:
        print("\nNo reviews have been classified yet, so sentiment/category "
              "breakdowns are unavailable.")

    print("\nTop 5 most frequent content words (across all stored reviews):")
    if stored_reviews:
        all_words = []
        for r in stored_reviews:
            all_words.extend(r["processed"].split())
        if all_words:
            top5 = Counter(all_words).most_common(5)
            for word, count in top5:
                print(f"  {word:<14}: {count}")
        else:
            print("  (no content words found -- all stored reviews were "
                  "stopwords/punctuation only)")
    else:
        print("  (no reviews stored yet)")
    print("=" * 60)


# ======================================================================
# Menu loop
# ======================================================================
MENU = """
Food Delivery Review Intelligence System
-----------------------------------------
1. Add a new review
2. Classify a review
3. View summary report
4. Exit
"""


def main():
    print("Starting up...")
    print(f"  {sentiment_msg}")
    print(f"  {category_msg}")

    while True:
        print(MENU)
        choice = input("Select an option [1-4]: ").strip()

        if choice == "1":
            add_review()
        elif choice == "2":
            classify_review()
        elif choice == "3":
            view_summary()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid option -- please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
