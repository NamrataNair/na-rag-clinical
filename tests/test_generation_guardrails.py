import pytest
from generation.generation_guardrails import check_faithfulness

def test_guardrail_faithful():
    generated_text = "The patient has a history of diabetes."
    sample_evidence = [
        {"text": "The patient has a history of diabetes.", "assertion": "historical"},
        {"text": "No evidence of infection was found.", "assertion": "negated"}
    ]
    assert check_faithfulness(generated_text, sample_evidence) == True

def test_guardrail_unfaithful():
    generated_text = "The patient is experiencing infection."
    sample_evidence = [
        {"text": "The patient has a history of diabetes.", "assertion": "historical"},
        {"text": "infection.", "assertion": "negated"}
    ]
    # The generated text contains "infection.", which is marked as negated in evidence.
    # Therefore, the system should catch this hallucination.
    assert check_faithfulness(generated_text, sample_evidence) == False
