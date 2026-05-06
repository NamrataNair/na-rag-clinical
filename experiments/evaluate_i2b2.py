import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing.build_index import build_index
from retrieval.dense_retriever import DenseRetriever
from retrieval.assertion_reranker import rerank
from evaluation.evaluate_retrieval import precision_at_k, negation_fp_rate
from experiments.run_baselines import run_baseline
from experiments.run_na_rag import run_na_rag

def load_i2b2_dataset():
    # Since the i2b2 dataset is not distributed due to licensing,
    # we simulate an evaluation using a robust mock clinical dataset
    # representing various cases of assertions and negations.

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
        ],
        [
            "Patient presents with severe headache.",
            "Rule out meningitis.",
            "No sign of infection."
        ],
        [
            "History of hypertension noted.",
            "Currently complaining of dizziness.",
            "Mother had a stroke at age 60."
        ]
    ]

    queries = [
        {"query": "Does the patient have appendicitis?", "intent": "asserted"},
        {"query": "ruled out meningitis", "intent": "negated"},
        {"query": "what is the family history of heart disease", "intent": "family"},
        {"query": "shortness of breath", "intent": "asserted"},
        {"query": "infection", "intent": "asserted"}
    ]

    return mock_documents, queries

def run_evaluation():
    documents, queries = load_i2b2_dataset()

    # Setup index
    index = build_index(documents)
    corpus_texts = [entry["text"] for entry in index]
    text_to_assertion = {entry["text"]: entry["assertion"] for entry in index}

    # Setup Dense Retriever
    retriever = DenseRetriever()
    corpus_embeddings = retriever.encode(corpus_texts)

    # Metrics
    baseline_p_at_1 = []
    baseline_neg_fp = []

    narag_p_at_1 = []
    narag_neg_fp = []

    for q in queries:
        query_text = q["query"]

        # 1. Baseline
        base_results = run_baseline(query_text, corpus_embeddings, corpus_texts, text_to_assertion, top_k=3)

        # 2. NA-RAG
        # Retrieve first
        initial_results = retriever.retrieve(query_text, corpus_embeddings, corpus_texts, top_k=3)
        for res in initial_results:
            res["assertion"] = text_to_assertion[res["text"]]

        # Rerank using NA-RAG
        narag_results = run_na_rag(initial_results, query=query_text)

        # Calculate metrics for baseline
        # In baseline, precision depends on if the intent matched the assertion retrieved.
        # But evaluate_retrieval.py precision_at_k assumes tp if assertion == "asserted".
        # Let's write a custom evaluator that respects the intent of the query for fairness.

        def calculate_precision(results, intent, k=1):
            top_k = results[:k]
            # If the user asked for a negated condition, a True Positive is retrieving a negated condition.
            if intent == "negated":
                target_assertion = "negated"
            elif intent == "family":
                target_assertion = "family"
            elif intent == "historical":
                target_assertion = "historical"
            else:
                target_assertion = "asserted"

            tp = sum(1 for r in top_k if r["assertion"] == target_assertion)
            return tp / k

        baseline_p_at_1.append(calculate_precision(base_results, q["intent"], k=1))
        baseline_neg_fp.append(negation_fp_rate(base_results))

        narag_p_at_1.append(calculate_precision(narag_results, q["intent"], k=1))
        narag_neg_fp.append(negation_fp_rate(narag_results))

    avg_baseline_p1 = np.mean(baseline_p_at_1)
    avg_baseline_nfp = np.mean(baseline_neg_fp)

    avg_narag_p1 = np.mean(narag_p_at_1)
    avg_narag_nfp = np.mean(narag_neg_fp)

    print(f"--- Evaluation on i2b2 Temporal Dataset (Simulated) ---")
    print(f"Baseline RAG - Precision@1: {avg_baseline_p1:.2f}, Negation FP Rate: {avg_baseline_nfp:.2f}")
    print(f"NA-RAG       - Precision@1: {avg_narag_p1:.2f}, Negation FP Rate: {avg_narag_nfp:.2f}")

    plot_results(avg_baseline_p1, avg_baseline_nfp, avg_narag_p1, avg_narag_nfp)

def plot_results(b_p1, b_nfp, n_p1, n_nfp):
    labels = ['Precision@1 (Higher is Better)', 'Negation False Positive Rate (Lower is Better)']
    baseline_scores = [b_p1, b_nfp]
    narag_scores = [n_p1, n_nfp]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline RAG', color='indianred')
    rects2 = ax.bar(x + width/2, narag_scores, width, label='NA-RAG', color='mediumseagreen')

    ax.set_ylabel('Scores')
    ax.set_title('Baseline RAG vs. NA-RAG Performance on Clinical Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()

    plt.savefig('evaluation_results.png')
    print("Saved plot to 'evaluation_results.png'")

if __name__ == "__main__":
    run_evaluation()
