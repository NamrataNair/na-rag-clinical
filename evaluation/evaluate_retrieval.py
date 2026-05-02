def precision_at_k(results, k=5):
    top_k = results[:k]
    tp = sum(1 for r in top_k if r["assertion"] == "asserted")
    return tp / k


def negation_fp_rate(results):
    if not results:
        return 0.0
    fp = sum(1 for r in results if r["assertion"] == "negated")
    return fp / len(results)


# Example usage
"""if __name__ == "__main__":  
    sample_results = [
        {"text": "The patient has diabetes.", "assertion": "asserted", "score": 0.9},
        {"text": "No evidence of infection.", "assertion": "negated", "score": 0.8},
        {"text": "Possible history of hypertension.", "assertion": "hypothetical", "score": 0.7},
        {"text": "The patient denies chest pain.", "assertion": "negated", "score": 0.6},
        {"text": "He is currently experiencing fatigue.", "assertion": "asserted", "score": 0.5}
    ]
    p_at_3 = precision_at_k(sample_results, k=3)
    neg_fp = negation_fp_rate(sample_results)
    print(f"Precision at 3: {p_at_3}")
    print(f"Negation False Positive Rate: {neg_fp}")    
"""