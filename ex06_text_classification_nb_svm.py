
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

documents = [
    "I love programming in Python",
    "Python is great for machine learning",
    "I enjoy learning data science",
    "I dislike bugs in my code",
    "Debugging is frustrating sometimes",
    "SVM models are powerful classifiers",
    "Naive Bayes is simple and fast",
    "Machine learning can be fun",
    "I hate syntax errors",
    "Text classification is an interesting task"
]

labels = [
    "Positive",
    "Positive",
    "Positive",
    "Negative",
    "Negative",
    "Positive",
    "Positive",
    "Positive",
    "Negative",
    "Positive"
]

X_train, X_test, y_train, y_test = train_test_split(
    documents, labels, test_size=0.3, random_state=42
)

vectorizer   = TfidfVectorizer()
X_train_vec  = vectorizer.fit_transform(X_train)
X_test_vec   = vectorizer.transform(X_test)

# ── Naïve Bayes ───────────────────────────────────────────────────────────────
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
y_pred_nb = nb.predict(X_test_vec)

print("=== NAÏVE BAYES ===")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# ── SVM ───────────────────────────────────────────────────────────────────────
svm = SVC(kernel='linear')
svm.fit(X_train_vec, y_train)
y_pred_svm = svm.predict(X_test_vec)

print("\n=== SVM ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
