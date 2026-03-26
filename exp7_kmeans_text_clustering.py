import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

documents = [
    "Machine learning is part of artificial intelligence",
    "Deep learning belongs to machine learning",
    "Artificial intelligence is changing technology",
    "Python is a famous programming language",
    "Java and Python are commonly used languages",
    "Software developers write programs using languages",
    "Data science uses machine learning methods",
    "Programming languages help to build software",
    "AI and ML are modern technologies",
    "Developers create applications using Python"
]

data = pd.DataFrame({"Text": documents})
print("Dataset\n")
print(data)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data["Text"])
print("\nTF-IDF Size:", X.shape)

model = KMeans(n_clusters=3, random_state=1)
model.fit(X)
data["Cluster"] = model.labels_

print("\nClustered Documents\n")
print(data)

pca = PCA(n_components=2)
points = pca.fit_transform(X.toarray())

plt.scatter(points[:, 0], points[:, 1], c=data["Cluster"])
plt.title("K-Means Text Clustering")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
