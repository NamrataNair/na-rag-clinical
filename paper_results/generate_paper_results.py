import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing.build_index import build_index
from retrieval.dense_retriever import DenseRetriever
from retrieval.assertion_reranker import rerank
from evaluation.evaluate_retrieval import precision_at_k, negation_fp_rate
from experiments.run_baselines import run_baseline
from experiments.run_na_rag import run_na_rag
from experiments.evaluate_i2b2 import load_i2b2_dataset

def calculate_precision(results, intent, k):
    top_k = results[:k]
    if intent == "negated":
        target_assertion = "negated"
    elif intent == "family":
        target_assertion = "family"
    elif intent == "historical":
        target_assertion = "historical"
    else:
        target_assertion = "asserted"

    tp = sum(1 for r in top_k if r["assertion"] == target_assertion)
    return tp / k if k > 0 else 0

def generate_results():
    print("Loading simulated clinical dataset for paper evaluation...")
    documents, queries = load_i2b2_dataset()

    index = build_index(documents)
    corpus_texts = [entry["text"] for entry in index]
    text_to_assertion = {entry["text"]: entry["assertion"] for entry in index}

    retriever = DenseRetriever()
    corpus_embeddings = retriever.encode(corpus_texts)

    metrics = {
        "Baseline": {"p@1": [], "p@2": [], "p@3": [], "nfp": []},
        "NA-RAG": {"p@1": [], "p@2": [], "p@3": [], "nfp": []}
    }

    for q in queries:
        query_text = q["query"]
        intent = q["intent"]

        # Baseline
        base_results = run_baseline(query_text, corpus_embeddings, corpus_texts, text_to_assertion, top_k=3)
        metrics["Baseline"]["p@1"].append(calculate_precision(base_results, intent, 1))
        metrics["Baseline"]["p@2"].append(calculate_precision(base_results, intent, 2))
        metrics["Baseline"]["p@3"].append(calculate_precision(base_results, intent, 3))
        metrics["Baseline"]["nfp"].append(negation_fp_rate(base_results))

        # NA-RAG
        initial_results = retriever.retrieve(query_text, corpus_embeddings, corpus_texts, top_k=3)
        for res in initial_results:
            res["assertion"] = text_to_assertion[res["text"]]
        narag_results = run_na_rag(initial_results, query=query_text)

        metrics["NA-RAG"]["p@1"].append(calculate_precision(narag_results, intent, 1))
        metrics["NA-RAG"]["p@2"].append(calculate_precision(narag_results, intent, 2))
        metrics["NA-RAG"]["p@3"].append(calculate_precision(narag_results, intent, 3))
        metrics["NA-RAG"]["nfp"].append(negation_fp_rate(narag_results))

    # Average metrics
    avg_metrics = {
        "Metric": ["Precision@1", "Precision@2", "Precision@3", "Negation FP Rate"],
        "Baseline RAG": [
            np.mean(metrics["Baseline"]["p@1"]),
            np.mean(metrics["Baseline"]["p@2"]),
            np.mean(metrics["Baseline"]["p@3"]),
            np.mean(metrics["Baseline"]["nfp"])
        ],
        "NA-RAG": [
            np.mean(metrics["NA-RAG"]["p@1"]),
            np.mean(metrics["NA-RAG"]["p@2"]),
            np.mean(metrics["NA-RAG"]["p@3"]),
            np.mean(metrics["NA-RAG"]["nfp"])
        ]
    }

    # Save CSV
    df = pd.DataFrame(avg_metrics)
    csv_path = os.path.join("paper_results", "metrics_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    print("\n" + df.to_string(index=False) + "\n")

    # Plot Precision@K
    k_vals = ["P@1", "P@2", "P@3"]
    b_p = [avg_metrics["Baseline RAG"][0], avg_metrics["Baseline RAG"][1], avg_metrics["Baseline RAG"][2]]
    n_p = [avg_metrics["NA-RAG"][0], avg_metrics["NA-RAG"][1], avg_metrics["NA-RAG"][2]]

    x = np.arange(len(k_vals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    rects1 = ax.bar(x - width/2, b_p, width, label='Baseline', color='lightcoral')
    rects2 = ax.bar(x + width/2, n_p, width, label='NA-RAG', color='mediumseagreen')

    ax.set_ylabel('Precision (Intent-Aware)')
    ax.set_title('Precision@K Comparison on Simulated i2b2 Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(k_vals)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    fig.tight_layout()
    plt.savefig(os.path.join("paper_results", "precision_at_k.png"))
    plt.close()

    # Plot False Positive Rate separately
    fig2, ax2 = plt.subplots(figsize=(4, 5), dpi=300)
    ax2.bar(['Baseline', 'NA-RAG'], [avg_metrics["Baseline RAG"][3], avg_metrics["NA-RAG"][3]], color=['lightcoral', 'mediumseagreen'])
    ax2.set_ylabel('False Positive Rate')
    ax2.set_title('Negation FP Rate\n(Lower is Better)')
    for i, v in enumerate([avg_metrics["Baseline RAG"][3], avg_metrics["NA-RAG"][3]]):
        ax2.text(i, v + 0.01, f"{v:.2f}", ha='center')
    fig2.tight_layout()
    plt.savefig(os.path.join("paper_results", "fpr_comparison.png"))
    plt.close()

    print("High-resolution figures generated in 'paper_results/' directory.")

if __name__ == "__main__":
    generate_results()
