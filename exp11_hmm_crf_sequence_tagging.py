# Install dependencies if needed:
# pip install nltk sklearn-crfsuite

import nltk
from nltk.tag import hmm
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

data = [
    [("John", "NNP"), ("is", "VBZ"), ("playing", "VBG"), ("football", "NN")],
    [("Mary", "NNP"), ("likes", "VBZ"), ("reading", "VBG"), ("books", "NNS")],
    [("The", "DT"), ("sun", "NN"), ("is", "VBZ"), ("bright", "JJ")],
    [("Children", "NNS"), ("are", "VBP"), ("running", "VBG"), ("fast", "RB")],
    [("He", "PRP"), ("is", "VBZ"), ("eating", "VBG"), ("food", "NN")],
    [("She", "PRP"), ("writes", "VBZ"), ("beautiful", "JJ"), ("poems", "NNS")]
]

# ---- HMM Model ----
trainer = hmm.HiddenMarkovModelTrainer()
hmm_model = trainer.train_supervised(data)


def hmm_accuracy(model, dataset):
    correct, total = 0, 0
    for sent in dataset:
        words = [w for w, t in sent]
        true_tags = [t for w, t in sent]
        pred_tags = [t for w, t in model.tag(words)]
        for t1, t2 in zip(true_tags, pred_tags):
            if t1 == t2:
                correct += 1
            total += 1
    return correct / total


hmm_acc = hmm_accuracy(hmm_model, data)
test_words = ["John", "is", "running", "fast"]
hmm_result = hmm_model.tag(test_words)


# ---- CRF Model ----
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:]
    }
    if i > 0:
        features['prev_word'] = sent[i - 1][0]
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        features['next_word'] = sent[i + 1][0]
    else:
        features['EOS'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


X = [sent2features(s) for s in data]
y = [sent2labels(s) for s in data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

crf = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100)
crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)
crf_acc = metrics.flat_accuracy_score(y_test, y_pred)

test_sent_crf = [(w, "") for w in test_words]
crf_result = crf.predict([sent2features(test_sent_crf)])

print("----- HMM MODEL -----")
print("HMM Accuracy:", hmm_acc)
print("HMM Tagged Sentence:")
print(hmm_result)

print("\n----- CRF MODEL -----")
print("CRF Accuracy:", crf_acc)
print("CRF Tagged Sentence:")
print(list(zip(test_words, crf_result[0])))
