# Buisness Name NLP Experiments

Higher level string manipulation and processing (e.g NLP) for business names.

## What is this?

A collection of python scripts I turned into a module for NLP applications of business names.

### Normalization

Apply unicode normalization and accent folding.

Apply transliteration.

Apply stopword removal, abbreviation fixing and stemming.

### TF-IDF

Implementation of TF-IDF with comparison of vectors using cosine distance via PostgreSQL's pg_vector

### BM25

Implementation of Elasticsearch's BM25 with comparison of vectors using scoring

### Sentence Embeddings

Usage of BERT Model to create embeddings with comparison of vectors using cosine distance via PostgreSQL's pg_vector

### High Level Visualizations

Squish N dimension vector spaces down into a 2D scatter plot via `t-SNE`. Maintains the distance between points so you can clearly see how close things were
without all that 1000-D nonsense.

### Examples

I attached images, docker scripts, etc to help you run the code via the Makefile
