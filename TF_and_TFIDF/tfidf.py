"""
tfidf implementation by numpy and sklearn
"""

import numpy as np
from collections import defaultdict
import math
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDF(object):
    def __init__(self):
        pass
    def compute_tfidf_by_numpy(self, doc):
        # Step 1 & 2: Tokenize and create vocabulary
        vocabulary = set(word for doc in documents for word in doc.split())
        vocab_dict = {word: i for i, word in enumerate(sorted(vocabulary))}

        # Initialize matrices
        tf = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
        idf = np.zeros(len(vocabulary), dtype=np.float64)
        tfidf = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)

        # Step 3: Calculate TF
        for i, doc in enumerate(documents):
            word_count = defaultdict(int)
            for word in doc.split():
                word_index = vocab_dict[word]
                word_count[word] += 1

            total_words = sum(word_count.values())
            for word, count in word_count.items():
                tf[i, vocab_dict[word]] = count / total_words

        # Step 4: Calculate IDF
        doc_count = np.zeros(len(vocabulary), dtype=np.float64)

        for doc in documents:
            for word in set(doc.split()):
                doc_count[vocab_dict[word]] += 1

        idf = np.log(len(documents) / (doc_count + 1)) + 1  # Adding 1 to denominator to avoid division by zero

        # Step 5: Compute TF-IDF
        tfidf = tf * idf

        # Normalize TF-IDF rows to have unit norm
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        tfidf_normalized = tfidf / norms
        return tfidf_normalized
    def compute_tfidf_by_sklearn(self, doc):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(doc)
        return tfidf_matrix.toarray()

if __name__ == "__main__":
    documents = [
        "the quick brown fox",
        "jumped over the lazy dog",
        "the quick dog",
    ]
    tf = TFIDF()
    result_numpy = tf.compute_tfidf_by_sklearn(documents)
    result_sklearn = tf.compute_tfidf_by_numpy(documents)
    print("result for tf numpy:", result_numpy)
    print("result for tf sklearn:", result_sklearn)



