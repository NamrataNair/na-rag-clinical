import pytest
from retrieval.dense_retriever import DenseRetriever

def test_dense_retriever():
    retriever = DenseRetriever()
    corpus = [
        "The patient has a history of diabetes.",
        "He is currently experiencing chest pain.",
        "No evidence of infection was found."
    ]
    corpus_embeddings = retriever.encode(corpus)

    query = "Does the patient have diabetes?"
    results = retriever.retrieve(query, corpus_embeddings, corpus, top_k=2)

    assert len(results) == 2
    assert "history of diabetes" in results[0]["text"].lower()
