import pytest
from retrieval.assertion_reranker import rerank

@pytest.fixture
def sample_results():
    return [
        {"text": "The patient has diabetes.", "assertion": "asserted", "score": 0.9},
        {"text": "No evidence of infection.", "assertion": "negated", "score": 0.8},
        {"text": "Possible history of hypertension.", "assertion": "hypothetical", "score": 0.7},
        {"text": "Family history of heart disease.", "assertion": "family", "score": 0.6}
    ]

def test_rerank_default(sample_results):
    import copy
    reranked = rerank(copy.deepcopy(sample_results))

    assert reranked[0]["assertion"] == "asserted"
    assert reranked[0]["score"] == 0.9

    assert reranked[1]["assertion"] == "hypothetical"
    assert reranked[1]["score"] == pytest.approx(0.7 * 0.4)

    assert reranked[-1]["assertion"] == "negated"
    assert reranked[-1]["score"] == pytest.approx(0.8 * -1.0)

def test_rerank_negated_query(sample_results):
    import copy
    reranked = rerank(copy.deepcopy(sample_results), query="rule out infection")

    # "rule out infection" is a hypothetical query intent according to HYPOTHETICAL_PATTERNS
    # wait, "rule out" is in both NEGATION_PATTERNS and HYPOTHETICAL_PATTERNS but NEGATION_PATTERNS comes first in evaluation because family > negative > hypothetical
    # Oh wait, let's check `extract_assertion("rule out infection")`. It hits "rule out" in NEGATION_PATTERNS.
    # Therefore, the weights for a negated query are applied.
    # negated query weights: asserted=-1.0, hypothetical=-0.5, historical=-0.5, family=-0.5, negated=1.0

    assert reranked[0]["assertion"] == "negated"
    assert reranked[0]["score"] == pytest.approx(0.8 * 1.0)

    assert reranked[-1]["assertion"] == "asserted"
    assert reranked[-1]["score"] == pytest.approx(0.9 * -1.0)

def test_rerank_family_query(sample_results):
    import copy
    reranked = rerank(copy.deepcopy(sample_results), query="what is the family history")

    # "family history" triggers family intent
    # family query weights: asserted=0.0, hypothetical=0.0, historical=0.0, family=1.0, negated=0.0

    assert reranked[0]["assertion"] == "family"
    assert reranked[0]["score"] == pytest.approx(0.6 * 1.0)

    # All other assertions should score 0.0, so the last element will have score 0.0
    assert reranked[-1]["score"] == pytest.approx(0.0)
