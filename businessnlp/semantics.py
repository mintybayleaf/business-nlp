import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

import random


from businessnlp.normalize import normalize
import businessnlp.sql as sql
import businessnlp.data as data

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs[0] is last hidden state, shape [batch, seq_len, hidden_dim]
        # We average token embeddings -> sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)

    # Convert to numpy
    return embeddings[0].cpu().numpy()


def generate_embeddings(names):
    embeddings = []
    for name in names:
        name = " ".join(normalize(name))
        if name:
            embeddings.append(embed_text(name))

    return embeddings, names


def embeddings_demo(names):
    vectors, names = generate_embeddings(names)

    table_name = "semantics_demo"
    sql.setup_table(table_name, vectors[0].size)
    for name, vector in zip(names, vectors):
        sql.insert_np_array(table_name, name, vector)

    sample_vectors = [vectors[0], vectors[20], vectors[999]]
    sample_names = [
        names[0],
        names[20],
        names[999],
    ]
    for name, vector in zip(sample_names, sample_vectors):
        for result in sql.cosine_distance_nearest_vectors(table_name, vector, 10):
            print(f"[cosine distance] {name} => {result}")


if __name__ == "__main__":
    print()
    print("===================================")
    print()
    print("demo company names")
    embeddings_demo(random.sample(data.load_text_file("company_names"), 1000))
    print()
    print("===================================")
    print()
    print("demo similar names")
    embeddings_demo(random.sample(data.load_text_file("similar_company_names"), 1000))
    print()
    print("===================================")
    print()
    print("demo semantic names")
    embeddings_demo(random.sample(data.load_text_file("semantic_words"), 1000))
