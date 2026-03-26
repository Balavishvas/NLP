import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text = """
Natural Language Processing is an important area of artificial intelligence.
It helps computers understand human language.
POS tagging is a technique used in NLP to identify the grammatical role of each word in a sentence.
"""

# Sentence tokenization
sentences = sent_tokenize(text)

# Store POS tagging results
pos_results = []
for sentence in sentences:
    words = word_tokenize(sentence)
    tags = pos_tag(words)
    for word, tag in tags:
        pos_results.append((word, tag))

# Print POS tagging output
print("POS Tagging Output:\n")
for word, tag in pos_results:
    print(word, "->", tag)
