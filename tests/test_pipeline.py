import pytest
from indexing.build_index import build_index
from retrieval.dense_retriever import DenseRetriever
from retrieval.assertion_reranker import rerank
from generation.prompt_templates import build_prompt
from generation.generation_guardrails import check_faithfulness
from evaluation.evaluate_retrieval import precision_at_k, negation_fp_rate

@pytest.fixture
def mock_corpus():
    return [
        [
            "The patient is a 45-year-old male presenting with chest pain.",
            "He has a history of diabetes.",
            "He denies any shortness of breath.",
            "Family history is significant for heart disease."
        ],
        [
            "The patient was admitted for suspected appendicitis.",
            "CT scan showed no evidence of appendicitis.",
            "Patient was discharged with instructions to follow up."
        ]
    ]

@pytest.fixture
def setup_pipeline(mock_corpus):
    index = build_index(mock_corpus)
    corpus_texts = [entry["text"] for entry in index]
    retriever = DenseRetriever()
    corpus_embeddings = retriever.encode(corpus_texts)

    text_to_assertion = {entry["text"]: entry["assertion"] for entry in index}

    return retriever, corpus_embeddings, corpus_texts, text_to_assertion

def test_standard_query(setup_pipeline):
    retriever, corpus_embeddings, corpus_texts, text_to_assertion = setup_pipeline
    query = "Does the patient have appendicitis?"

    results = retriever.retrieve(query, corpus_embeddings, corpus_texts, top_k=3)
    for res in results:
        res["assertion"] = text_to_assertion[res["text"]]

    reranked = rerank(results, query=query)

    # Assertions
    # We should retrieve 'suspected appendicitis' or 'chest pain' higher than 'no evidence of appendicitis'
    assert reranked[0]["assertion"] in ["asserted", "hypothetical"]
    assert reranked[-1]["assertion"] == "negated"
    assert reranked[-1]["text"] == "CT scan showed no evidence of appendicitis."

def test_negated_query(setup_pipeline):
    retriever, corpus_embeddings, corpus_texts, text_to_assertion = setup_pipeline
    query = "ruled out appendicitis"

    results = retriever.retrieve(query, corpus_embeddings, corpus_texts, top_k=3)
    for res in results:
        res["assertion"] = text_to_assertion[res["text"]]

    reranked = rerank(results, query=query)

    # We expect 'no evidence of appendicitis' to be promoted because of the negated query intent
    assert reranked[0]["assertion"] == "negated"
    assert "no evidence of appendicitis" in reranked[0]["text"]

def test_guardrails():
    reranked_results = [
        {"text": "The patient is a 45-year-old male presenting with chest pain.", "assertion": "asserted", "score": 0.4299},
        {"text": "The patient was admitted for suspected appendicitis.", "assertion": "hypothetical", "score": 0.3456},
        {"text": "CT scan showed no evidence of appendicitis.", "assertion": "negated", "score": -0.6826}
    ]

    # The generated text hallucinates the negated condition
    mock_generated_text = "CT scan showed no evidence of appendicitis."
    is_faithful = check_faithfulness(mock_generated_text, reranked_results)
    assert is_faithful == False
