import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing.build_index import build_index
from retrieval.dense_retriever import DenseRetriever
from retrieval.assertion_reranker import rerank
from generation.prompt_templates import build_prompt
from generation.generation_guardrails import check_faithfulness
from evaluation.evaluate_retrieval import precision_at_k, negation_fp_rate

def main():
    print("Setting up Mock Corpus...")
    mock_documents = [
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

    print("1. Event Chunking and Assertion Extraction (Building Index)...")
    index = build_index(mock_documents)

    corpus_texts = [entry["text"] for entry in index]

    print("2. Dense Retrieval Setup...")
    retriever = DenseRetriever()
    corpus_embeddings = retriever.encode(corpus_texts)

    query = "Does the patient have appendicitis?"
    print(f"Query: {query}")

    print("3. Retrieve Initial Results...")
    results = retriever.retrieve(query, corpus_embeddings, corpus_texts, top_k=3)

    # Map back assertion metadata to retrieved results
    text_to_assertion = {entry["text"]: entry["assertion"] for entry in index}
    for res in results:
        res["assertion"] = text_to_assertion[res["text"]]

    print("Initial Results:")
    for res in results:
        print(f"  [{res['assertion'].upper()}] {res['text']} (score: {res['score']:.4f})")

    print("4. Assertion-Aware Reranking...")
    reranked_results = rerank(results, query=query)
    print("Reranked Results:")
    for res in reranked_results:
        print(f"  [{res['assertion'].upper()}] {res['text']} (score: {res['score']:.4f})")

    print("5. Prompt Generation...")
    prompt = build_prompt(query, reranked_results)
    print("Generated Prompt:")
    print(prompt)

    print("6. Guardrails Evaluation...")
    # Simulate a generated completion that violates the rule
    mock_generated_text = "The patient has appendicitis."
    print(f"Mock generated answer: {mock_generated_text}")
    is_faithful = check_faithfulness(mock_generated_text, reranked_results)
    print(f"Is generated answer faithful? {is_faithful}")

    print("7. Retrieval Metrics...")
    p_at_1 = precision_at_k(reranked_results, k=1)
    neg_fp = negation_fp_rate(reranked_results)
    print(f"Precision@1: {p_at_1:.2f}")
    print(f"Negation FP Rate: {neg_fp:.2f}")

    print("Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
