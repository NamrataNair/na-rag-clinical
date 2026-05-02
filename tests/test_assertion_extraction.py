import pytest
from preprocessing.assertion_extraction import extract_assertion, Assertion

def test_extract_assertion_negated():
    assert extract_assertion("The patient denies chest pain.") == Assertion.NEGATED
    assert extract_assertion("There is no history of diabetes.") == Assertion.NEGATED
    assert extract_assertion("CT scan ruled out appendicitis.") == Assertion.NEGATED

def test_extract_assertion_family():
    assert extract_assertion("Family history of heart disease is noted.") == Assertion.FAMILY
    assert extract_assertion("Mother had breast cancer.") == Assertion.FAMILY

def test_extract_assertion_hypothetical():
    assert extract_assertion("Suspected pneumonia.") == Assertion.HYPOTHETICAL
    assert extract_assertion("Consider an MRI if symptoms persist.") == Assertion.HYPOTHETICAL

def test_extract_assertion_historical():
    assert extract_assertion("Patient has a prior history of smoking.") == Assertion.HISTORICAL
    assert extract_assertion("Previous surgery in 2010.") == Assertion.HISTORICAL

def test_extract_assertion_asserted():
    assert extract_assertion("The patient has diabetes.") == Assertion.ASSERTED
    assert extract_assertion("He is presenting with acute abdominal pain.") == Assertion.ASSERTED

def test_precedence_rules():
    # 'no history of' should be NEGATED, not HISTORICAL
    assert extract_assertion("There is no history of diabetes.") == Assertion.NEGATED

    # 'family history of' should be FAMILY, not HISTORICAL
    assert extract_assertion("Family history of diabetes.") == Assertion.FAMILY
