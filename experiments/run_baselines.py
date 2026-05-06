"""
Runs Traditional RAG baseline (Dense Retriever without Assertion Reranking).
"""

from retrieval.dense_retriever import DenseRetriever

def run_baseline(query, corpus_embeddings, corpus_texts, text_to_assertion, top_k=3):
    retriever = DenseRetriever()
    # In baseline, we only do dense retrieval without reranking
    results = retriever.retrieve(query, corpus_embeddings, corpus_texts, top_k=top_k)

    # Map back assertion metadata to retrieved results for evaluation purposes only
    for res in results:
        res["assertion"] = text_to_assertion[res["text"]]

    return results

if __name__ == "__main__":
    print("Running baseline retrieval placeholder.")
