"""
Assertion-aware re-ranking logic.
"""

from preprocessing.assertion_extraction import extract_assertion, Assertion

DEFAULT_ASSERTION_WEIGHTS = {
    "asserted": 1.0,
    "hypothetical": 0.4,
    "historical": 0.2,
    "family": -0.5,
    "negated": -1.0
}


def rerank(results, query=None):
    weights = DEFAULT_ASSERTION_WEIGHTS.copy()

    # Analyze the query for intent and adjust weights
    if query:
        query_assertion = extract_assertion(query).value

        if query_assertion == "negated":
            # If the user specifically asks for what was ruled out or negated
            # e.g., "What conditions were ruled out?"
            weights = {
                "asserted": -1.0,
                "hypothetical": -0.5,
                "historical": -0.5,
                "family": -0.5,
                "negated": 1.0
            }
        elif query_assertion == "family":
            # If the user specifically asks about family history
            weights = {
                "asserted": 0.0,
                "hypothetical": 0.0,
                "historical": 0.0,
                "family": 1.0,
                "negated": 0.0
            }
        elif query_assertion == "historical":
            weights = {
                "asserted": 0.2,
                "hypothetical": 0.0,
                "historical": 1.0,
                "family": 0.0,
                "negated": -0.5
            }

    for r in results:
        weight = weights.get(r["assertion"], 0)
        r["score"] *= weight
    return sorted(results, key=lambda x: x["score"], reverse=True)
