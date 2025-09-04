from nltk.corpus.reader import WordListCorpusReader
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import random

from businessnlp.normalize import normalize
import businessnlp.sql as sql
import businessnlp.data as data
from businessnlp.graph import plot_vectors_2d

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # evaluation mode disables gradients

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)


# Return overlapping 3-character sequences from tokens (or characters)
def generate_trigrams(tokens):
    text = "".join(tokens)
    if len(text) < 3:
        return [text]
    return [text[i : i + 3] for i in range(len(text) - 2)]


# Embed a list of strings (tokens or trigrams) in batches.
# Returns list of np.ndarray vectors.
def embed_text_batch(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_emb)
    return embeddings


# Embed a business name.
# mode='tokens' -> word tokens
# mode='ngrams' -> character n-grams
def embed_business_name(name, mode="tokens", ngram_size=3):
    if mode == "tokens":
        tokens = normalize(name, return_tokens=True)
        if not tokens:
            tokens = [normalize(name)]
        # embed each token as-is
        token_vecs = embed_text_batch(tokens)
        return np.mean(token_vecs, axis=0)
    else:  # n-grams
        collapsed = normalize(name)
        trigrams = [
            collapsed[i : i + ngram_size]
            for i in range(len(collapsed) - ngram_size + 1)
        ]
        if not trigrams:
            trigrams = [collapsed]
        trigram_vecs = embed_text_batch(trigrams)
        return np.mean(trigram_vecs, axis=0)


# Embed a list of names into vectors
def generate_embeddings(names, mode="tokens", ngram_size=3):
    embeddings = [
        embed_business_name(name, mode=mode, ngram_size=ngram_size) for name in names
    ]
    return embeddings, names


# Generate embeddings for all names
# Store them in SQL
# Pick sample queries to find nearest neighbors
def embeddings_demo(names, mode="tokens", ngram_size=3, sample_size=5, top_n=10):
    vectors, names = generate_embeddings(names, mode=mode, ngram_size=ngram_size)

    table_name = f"embeddings_demo_{mode}"
    sql.setup_table(table_name, vectors[0].size)
    for name, vector in zip(names, vectors):
        normalized_name = " ".join(normalize(name, return_tokens=True))
        sql.insert_np_array(table_name, normalized_name, vector)

    sample_indices = random.sample(range(len(names)), min(sample_size, len(names)))
    for idx in sample_indices:
        query_name = names[idx]
        normalized_query_name = " ".join(normalize(query_name, return_tokens=True))
        query_vector = vectors[idx]
        print(f"\n[cosine distance] query='{normalized_query_name}'")
        for result in sql.cosine_distance_nearest_vectors(
            table_name, query_vector, top_n
        ):
            print(result)


def demo(sample_size=None, queries=2, top_n=3):
    if sample_size is None:
        words = data.load_text_file("companies")
        overlaps = data.load_text_file("company_overlaps")
        spelling_variations = data.load_text_file("company_variations")
        semantic = data.load_text_file("semantics")
    else:
        words = random.sample(data.load_text_file("companies"), sample_size)
        overlaps = random.sample(data.load_text_file("company_overlaps"), sample_size)
        spelling_variations = random.sample(data.load_text_file("company_variations"), sample_size)
        semantic =  random.sample(data.load_text_file("semantics"), sample_size)

    print()
    print("embeddings normal demo")
    print("--------------------------------")
    print("\nwhole word tokens\n")
    embeddings_demo(words, mode="tokens", sample_size=queries, top_n=top_n)
    print()
    print("--------------------------------")
    print("\ncharacter trigrams\n")
    embeddings_demo(words, mode="ngrams",  ngram_size=3, sample_size=queries, top_n=top_n)
    print()

    print()
    print("embeddings overlaps demo")
    print("--------------------------------")
    print("\nwhole word tokens\n")
    embeddings_demo(overlaps, mode="tokens", sample_size=queries, top_n=top_n)
    print()
    print("--------------------------------")
    print("\ncharacter trigrams\n")
    embeddings_demo(overlaps, mode="ngrams",  ngram_size=3, sample_size=queries, top_n=top_n)
    print()

    print()
    print("embeddings spelling variations demo")
    print("--------------------------------")
    print("\nwhole word tokens\n")
    embeddings_demo(spelling_variations, mode="tokens", sample_size=queries, top_n=top_n)
    print()
    print("--------------------------------")
    print("\ncharacter trigrams\n")
    embeddings_demo(spelling_variations, mode="ngrams",  ngram_size=3, sample_size=queries, top_n=top_n)
    print()

    print()
    print("embeddings semantic demo")
    print("--------------------------------")
    print("\nwhole word tokens\n")
    embeddings_demo(semantic, mode="tokens", sample_size=queries, top_n=top_n)
    print()
    print("--------------------------------")
    print("\ncharacter trigrams\n")
    embeddings_demo(semantic, mode="ngrams",  ngram_size=3, sample_size=queries, top_n=top_n)
    print()


# Embeddings example
# "urgent care" → [embed("urgent"), embed("care")] → mean-pool → embedding for whole name
# "urgent care" → ["urg","rge","gen","ent","car","are"] → embed each → mean-pool → embedding

if __name__ == "__main__":
    print()
    print(
        """
Embeddings....What are they? It is a word people use a lot....

Instead of just counting words, embeddings use deep learning to map words or sentences into high-dimensional vectors that capture meaning.
This lets the model know that “urgent care” is more similar to “medical clinic” than to “car repair,” even if the words don’t overlap.

BERT embeddings

Tokens → capture semantic similarity
Trigrams → capture surface-level or “fuzzy string” similarity
"""
    )
    print()
    demo(queries=2, top_n=3)
