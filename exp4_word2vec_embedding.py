import sys
# Uncomment the line below if gensim is not installed
# import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "gensim"])

from gensim.models import Word2Vec

sentences = [
    ["natural", "language", "processing"],
    ["natural", "language", "understanding"],
    ["machine", "learning", "models"],
    ["deep", "learning", "neural", "networks"],
    ["word", "embedding", "using", "word2vec"],
    ["word", "embedding", "techniques"],
    ["artificial", "intelligence", "applications"],
    ["nlp", "uses", "machine", "learning"]
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=4,
    min_count=1,
    sg=1
)

print("VOCABULARY:")
print(model.wv.index_to_key)

print("\nWORD VECTOR FOR 'language':")
print(model.wv["language"])

print("\nWORDS SIMILAR TO 'language':")
for word, score in model.wv.most_similar("language", topn=5):
    print(word, ":", round(score, 3))

print("\nSIMILARITY BETWEEN 'language' AND 'processing':")
print(round(model.wv.similarity("language", "processing"), 3))

print("\nSIMILARITY BETWEEN 'learning' AND 'models':")
print(round(model.wv.similarity("learning", "models"), 3))
