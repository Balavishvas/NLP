import nltk
import string
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('punkt')
nltk.download('punkt_tab')

# Documents
documents = [
    "Natural language processing is a branch of artificial intelligence",
    "Natural language processing helps computers understand human language",
    "Machine learning and NLP are closely related fields"
]

print("DOCUMENTS:")
for doc in documents:
    print("-", doc)


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


clean_docs = [preprocess(doc) for doc in documents]

all_tokens = []
for doc in clean_docs:
    tokens = nltk.word_tokenize(doc)
    all_tokens.extend(tokens)

print("\nTOKENS:")
print(all_tokens)

num_words = len(all_tokens)
vocab_size = len(set(all_tokens))
word_freq = Counter(all_tokens)

print("\nBASIC FEATURES:")
print("Total number of words:", num_words)
print("Vocabulary size:", vocab_size)

print("\nWORD FREQUENCY:")
for word, count in word_freq.items():
    print(word, ":", count)

count_vectorizer = CountVectorizer(ngram_range=(1, 2))
bow_matrix = count_vectorizer.fit_transform(clean_docs)

print("\nBAG OF WORDS FEATURES:")
print("Vocabulary:", count_vectorizer.get_feature_names_out())
print("BoW Matrix:\n", bow_matrix.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(clean_docs)

print("\nTF-IDF FEATURES:")
print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_matrix.toarray().round(3))
