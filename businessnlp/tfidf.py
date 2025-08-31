from collections import Counter
import math
import random

import numpy as np

from businessnlp.normalize import normalize, make_ngrams
import businessnlp.sql as sql
import businessnlp.data as data
from businessnlp.graph import plot_vectors_2d


def term_frequency_map(tokens):
    return Counter(tokens)


def term_frequency(term, tokenized_document):
    document_frequency_map = term_frequency_map(tokenized_document)
    return document_frequency_map.get(term, 0) / len(tokenized_document)


def document_frequency_map(tokenized_corpus):
    document_frequency = Counter()
    for doc in tokenized_corpus:
        document_frequency.update(set(doc))
    return document_frequency


def inverse_document_frequency(term, document_frequency, total_documents):
    return math.log10(1 + total_documents / (document_frequency[term] + 1)) + 1


def tfidf(names, mode="tokens", ngram_size=3):
    """
    Compute TF-IDF vectors for a list of names.
    mode = "tokens" (word-level tokens after normalize)
         or "ngrams" (character-level ngrams of size ngram_size).
    """

    if mode == "tokens":
        # Get normalized list of tokens
        corpus = [normalize(name, return_tokens=True) for name in names]
    elif mode == "ngrams":
        # Character n-grams
        collapsed = [normalize(name) for name in names]
        corpus = [make_ngrams(list(doc), n=ngram_size) for doc in collapsed]

    else:
        raise ValueError(f"Unknown mode {mode}")

    total_documents = len(corpus)

    # Frequency mapping from term to number of docs it appears in
    document_frequency = document_frequency_map(corpus)

    # Vocabulary and term index mapping
    vocabulary = sorted(list(document_frequency.keys()))
    terms = {term: i for i, term in enumerate(vocabulary)}

    # Compute TF-IDF vectors
    vectors = []
    for tokenized_document in corpus:
        vector = np.zeros(len(vocabulary), dtype=np.float64)
        for term in tokenized_document:
            tf = term_frequency(term, tokenized_document)
            idf = inverse_document_frequency(term, document_frequency, total_documents)
            term_idx = terms[term]
            vector[term_idx] += tf * idf

        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        vectors.append(vector)

    return vectors, vocabulary


def tfidf_demo(names):
    """
    Demo TF-IDF using both token-level and n-gram-level normalization.
    Inserts into SQL and shows cosine similarity results.
    """

    # Tokens demo
    print("\n=== TF-IDF demo (tokens) ===\n")
    vectors, vocabulary = tfidf(names, mode="tokens")

    table_name = "tfidf_demo_tokens"
    sql.setup_table(table_name, len(vocabulary))
    for name, vector in zip(names, vectors):
        sql.insert_np_array(table_name, name, vector)

    sample_indices = random.sample(list(range(len(names))), 5)
    for idx in sample_indices:
        name, vector = names[idx], vectors[idx]
        for result in sql.cosine_distance_nearest_vectors(table_name, vector, 3):
            print(f"[tokens cosine] {name} => {result}")

    # N-grams (trigram mode)
    print("\n=== TF-IDF demo (character trigrams) ===\n")
    vectors, vocabulary = tfidf(names, mode="ngrams", ngram_size=3)

    table_name = "tfidf_demo_ngrams"
    sql.setup_table(table_name, len(vocabulary))
    for name, vector in zip(names, vectors):
        sql.insert_np_array(table_name, name, vector)

    for idx in sample_indices:
        name, vector = names[idx], vectors[idx]
        for result in sql.cosine_distance_nearest_vectors(table_name, vector, 3):
            print(f"[ngrams cosine] {name} => {result}")


def demo(sample_size=1000, visualize=False):
    print()
    print("===================================")
    print("Demo company names")
    tfidf_demo(random.sample(data.load_text_file("company_names"), sample_size))
    print()
    print("===================================")
    print("Demo similar names")
    tfidf_demo(random.sample(data.load_text_file("similar_company_names"), sample_size))

    if visualize:
        vectors, names = tfidf(
            random.sample(data.load_text_file("company_names"), 200), mode="tokens"
        )
        plot_vectors_2d(vectors, names, title="TF-IDF Token Vectors")


if __name__ == "__main__":
    print()
    print(
        """
        TF-IDF (Term Frequency–Inverse Document Frequency)
            A simple way to turn text into numbers by counting how often words appear, while removing the importance of very common words like “the” or “and.”
            It highlights words that are important for distinguishing one document from another.

        Example:

            Documents:
                doc1 = "urgent care center"
                doc2 = "medical care clinic"

            Step 1: tokenize and normalize
                doc1 tokens: ['urgent', 'care', 'center']
                doc2 tokens: ['medical', 'care', 'clinic']

            Step 2: compute term frequencies (TF)
                doc1 TF: {'urgent': 1/3, 'care': 1/3, 'center': 1/3}
                doc2 TF: {'medical': 1/3, 'care': 1/3, 'clinic': 1/3}

            Step 3: compute inverse document frequency (IDF)
                'care': appears in 2 docs → low IDF
                'urgent', 'center', 'medical', 'clinic': appear in 1 doc → higher IDF

            Step 4: multiply TF * IDF to get TF-IDF vectors
                doc1 TF-IDF: [urgent: high, care: low, center: high]
                doc2 TF-IDF: [medical: high, care: low, clinic: high]

        Trigrams: measures surface-level string overlap
        Tokens: measures semantic / word-level overlap
"""
    )
    print()
    demo(sample_size=2000)
