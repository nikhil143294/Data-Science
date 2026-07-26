# Task 3 - Complaint Category Classifier using Multinomial Naive Bayes

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Import preprocessing function from Task 1
from review_preprocessor import preprocess_review

# ----------------------------------------------------
# 1. Labelled Dataset (24 Complaints)
# ----------------------------------------------------

complaints = [

    # Delivery (8)
    ("Order arrived two hours late.", "Delivery"),
    ("Driver delivered to the wrong address.", "Delivery"),
    ("Delivery person never called me.", "Delivery"),
    ("Food was delivered very late.", "Delivery"),
    ("Tracking information was incorrect.", "Delivery"),
    ("Courier cancelled my order.", "Delivery"),
    ("Delivery took too long.", "Delivery"),
    ("Order never arrived.", "Delivery"),

    # Food Quality (8)
    ("Pizza was cold.", "Food Quality"),
    ("Burger was soggy.", "Food Quality"),
    ("Chicken was undercooked.", "Food Quality"),
    ("Soup was tasteless.", "Food Quality"),
    ("Rice was overcooked.", "Food Quality"),
    ("Food smelled bad.", "Food Quality"),
    ("Portion size was very small.", "Food Quality"),
    ("Dessert was stale.", "Food Quality"),

    # App (8)
    ("App crashed during payment.", "App"),
    ("Unable to login.", "App"),
    ("Payment failed.", "App"),
    ("Search feature is not working.", "App"),
    ("App keeps freezing.", "App"),
    ("Discount code failed.", "App"),
    ("Order page is loading forever.", "App"),
    ("App shows an error message.", "App")
]

# ----------------------------------------------------
# 2. Create DataFrame
# ----------------------------------------------------

df = pd.DataFrame(complaints, columns=["Complaint", "Category"])

# Preprocess complaints using Task 1 function
df["Processed"] = df["Complaint"].apply(preprocess_review)

print("="*70)
print("PREPROCESSED DATA")
print("="*70)
print(df)

# ----------------------------------------------------
# 3. Convert to TF-IDF Feature Matrix
# ----------------------------------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["Processed"])
y = df["Category"]

print("\nTF-IDF Matrix Shape:", X.shape)

# ----------------------------------------------------
# 4. Train-Test Split (80% Train, 20% Test)
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# ----------------------------------------------------
# 5. Train Multinomial Naive Bayes
# ----------------------------------------------------

model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ----------------------------------------------------
# 6. Classification Report
# ----------------------------------------------------

print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)

print(classification_report(y_test, y_pred))

# ----------------------------------------------------
# 7. Confusion Matrix
# ----------------------------------------------------

print("="*70)
print("CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual App", "Actual Delivery", "Actual Food Quality"],
    columns=["Predicted App", "Predicted Delivery", "Predicted Food Quality"]
)

print(cm_df)