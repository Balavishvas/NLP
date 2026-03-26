def process_text(text, word_to_replace=None, replacement_word=None):
    import re
    import nltk
    from collections import Counter
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import sent_tokenize, word_tokenize

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    char_count = len(text)
    lines = text.split('\n')
    line_count = len(lines)
    words = text.split()
    word_count = len(words)
    word_frequency = Counter(words)

    if word_to_replace and replacement_word:
        updated_text = text.replace(word_to_replace, replacement_word)
    else:
        updated_text = text

    # NLP PROCESSING
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    normalized_text = text.lower()
    normalized_text = re.sub(r'[^a-z\s]', '', normalized_text)
    normalized_tokens = word_tokenize(normalized_text)

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in normalized_tokens if word not in stop_words]

    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    unique_words = set(words)

    return {
        "character_count": char_count,
        "line_count": line_count,
        "word_count": word_count,
        "sentences": sentences,
        "tokens": tokens,
        "normalized_tokens": normalized_tokens,
        "stopword_removed": filtered_words,
        "stemmed_words": stemmed_words,
        "lemmatized_words": lemmatized_words,
        "unique_words": unique_words,
        "word_frequency": word_frequency,
        "updated_text": updated_text
    }


file_path = "input.txt"  # give input file path here
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

result = process_text(
    text=text_data,
    word_to_replace="automation",
    replacement_word="robotic system"
)

print("Character Count:", result["character_count"])
print("Line Count:", result["line_count"])
print("Word Count:", result["word_count"])
print("\nSentence Segmentation:")
print(result["sentences"])
print("\nTokenization:")
print(result["tokens"])
print("\nNormalization:")
print(result["normalized_tokens"])
print("\nStopword Removal:")
print(result["stopword_removed"])
print("\nStemming:")
print(result["stemmed_words"])
print("\nLemmatization:")
print(result["lemmatized_words"])
print("\nUnique Words:")
print(result["unique_words"])
print("\nWord Frequency:")
print(result["word_frequency"])
print("\nUpdated Text:\n", result["updated_text"])
