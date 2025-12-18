"""
Runs NA-RAG pipeline.
"""

from retrieval.assertion_reranker import rerank


def run_na_rag(results):
    return rerank(results)


if __name__ == "__main__":
    print("Running NA-RAG...")
    # sample_results = [
    #     {"text": "The patient has diabetes.", "assertion": "asserted", "score": 0.9},
    #     {"text": "No evidence of infection.", "assertion": "negated", "score": 0.8},
    #     {"text": "Possible history of hypertension.", "assertion": "hypothetical", "score": 0.7}
    # ]
    # reranked_results = run_na_rag(sample_results)
    # for res in reranked_results:
    #     print(res)  