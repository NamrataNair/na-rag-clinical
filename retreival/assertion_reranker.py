"""
Assertion-aware re-ranking logic.
"""

ASSERTION_WEIGHTS = {
    "asserted": 1.0,
    "hypothetical": 0.4,
    "historical": 0.2,
    "family": -0.5,
    "negated": -1.0
}


def rerank(results):
    for r in results:
        weight = ASSERTION_WEIGHTS.get(r["assertion"], 0)
        r["score"] *= weight
    return sorted(results, key=lambda x: x["score"], reverse=True)
# Example usage
"""
if __name__ == "__main__":  
    sample_results = [
        {"text": "The patient has diabetes.", "assertion": "asserted", "score": 0.9},
        {"text": "No evidence of infection.", "assertion": "negated", "score": 0.8},
        {"text": "Possible history of hypertension.", "assertion": "hypothetical", "score": 0.7}
    ]
    reranked = rerank(sample_results)
    for res in reranked:
        print(res)    
"""