# Input text
text = "Robots are smart. Robots work quickly."

# Set n for n-gram (1 = unigram, 2 = bigram, 3 = trigram)
n = 3


# Step 1: Tokenization (manual)
def tokenize(text):
    text = text.lower()
    for p in ".,!?;:":
        text = text.replace(p, " ")
    return text.split()


tokens = tokenize(text)


# Step 2: Create n-grams manually
def generate_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i + n]))
    return ngrams


ngram_list = generate_ngrams(tokens, n)

# Step 3: Count frequency
ngram_freq = {}
for ng in ngram_list:
    if ng in ngram_freq:
        ngram_freq[ng] += 1
    else:
        ngram_freq[ng] = 1

# Step 4: Compute probabilities
total_ngrams = sum(ngram_freq.values())
ngram_prob = {}
for ng, count in ngram_freq.items():
    ngram_prob[ng] = count / total_ngrams

print("\nOriginal text:")
print(text)
print(f"\nTokens:\n{tokens}")
print(f"\n{n}-grams and their frequency:")
for k, v in ngram_freq.items():
    print(k, ":", v)
print(f"\n{n}-gram probabilities:")
for k, v in ngram_prob.items():
    print(k, ":", round(v, 3))
