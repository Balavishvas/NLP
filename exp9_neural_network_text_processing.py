import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten

nltk.download('punkt')
nltk.download('stopwords')

# Sample dataset
sentences = [
    "Data science helps organizations make better decisions",
    "Neural networks recognize patterns in large datasets",
    "Students practice coding to improve programming skills",
    "Online courses help learners gain new knowledge",
    "Artificial intelligence supports automation in industries",
    "Teachers encourage creativity and critical thinking"
]
labels = np.array([0, 0, 1, 1, 0, 1])

stop_words = set(stopwords.words('english'))
processed_sentences = []
for sentence in sentences:
    words = word_tokenize(sentence.lower())
    filtered = [w for w in words if w.isalpha() and w not in stop_words]
    processed_sentences.append(" ".join(filtered))

print("Processed Sentences:")
for s in processed_sentences:
    print(s)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_sentences)
sequences = tokenizer.texts_to_sequences(processed_sentences)

print("\nVocabulary Index:")
print(tokenizer.word_index)

# Padding sequences
padded = pad_sequences(sequences, padding='post')
print("\nPadded Sequences:")
print(padded)

# Neural Network Model
model = Sequential()
model.add(Embedding(input_dim=100, output_dim=8, input_length=padded.shape[1]))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

print("\nTraining the Model...")
model.fit(padded, labels, epochs=8)

test_text = ["students learn programming through practice"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_pad = pad_sequences(test_seq, maxlen=padded.shape[1])
prediction = model.predict(test_pad)

if prediction > 0.5:
    print("\nPredicted Category: Education Related")
else:
    print("\nPredicted Category: Technology Related")
