"""
tf implementation by numpy and sklearn
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class TermFrequency(object):
    def __init__(self):
        pass
    def compute_tf_by_numpy(self, doc):
        # Step 1 & 2: Tokenize documents and create vocabulary
        vocabulary = set(word for doc in documents for word in doc.split())
        vocabulary = sorted(list(vocabulary))  # Sort vocabulary for consistency

        # Step 3: Initialize term frequency matrix
        tf_matrix = np.zeros((len(documents), len(vocabulary)))

        # Step 4: Calculate frequencies
        for i, doc in enumerate(documents):
            for word in doc.split():
                tf_matrix[i, vocabulary.index(word)] += 1

        # Step 5: Normalize frequencies
        tf_matrix = tf_matrix / np.sum(tf_matrix, axis=1, keepdims=True)
        return tf_matrix
    def compute_tf_by_sklearn(self, documents):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(documents)
        X= X.toarray()
        tf = X.astype(np.float64)
        tf /= np.sum(tf, axis=1, keepdims=True)
        return tf

if __name__ == "__main__":
    documents = [
        "the quick brown fox",
        "jumped over the lazy dog",
        "the quick dog",
    ]
    tf = TermFrequency()
    result_numpy = tf.compute_tf_by_numpy(documents)
    result_sklearn = tf.compute_tf_by_sklearn(documents)
    print("result for tf numpy:", result_numpy)
    print("result for tf sklearn:", result_sklearn)

