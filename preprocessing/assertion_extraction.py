"""
Rule-based + extensible assertion extraction.
Designed to mirror NegEx / ConText-style logic.
"""

from enum import Enum


class Assertion(Enum):
    ASSERTED = "asserted"
    NEGATED = "negated"
    HYPOTHETICAL = "hypothetical"
    HISTORICAL = "historical"
    FAMILY = "family"


NEGATION_PATTERNS = [
    "no evidence of",
    "denies",
    "negative for",
    "without evidence of", "no sign of", "free of", "rule out", "ruled out", "not present", "absent", "no history of" #add more as needed
]

HYPOTHETICAL_PATTERNS = [
    "possible",
    "suspected",
    "rule out",
    "cannot exclude", "could be",   "may have", "might have",   ,"if ... then", "suggestive of", "consider" #add more as needed and alter 
]

HISTORICAL_PATTERNS = [
    "history of",
    "previous",
    "prior"
]

FAMILY_PATTERNS = [
    "family history of", "mother had", "father had", "sister had", "brother had", :"daughter had", "son had", "parent had", "grandmother had", "grandfather had", "aunt had", "uncle had", "cousin had", #add more as needed.. these are just examples
]


def extract_assertion(sentence: str) -> Assertion:
    s = sentence.lower()

    for p in FAMILY_PATTERNS:
        if p in s:
            return Assertion.FAMILY

    for p in NEGATION_PATTERNS:
        if p in s:
            return Assertion.NEGATED

    for p in HYPOTHETICAL_PATTERNS:
        if p in s:
            return Assertion.HYPOTHETICAL

    for p in HISTORICAL_PATTERNS:
        if p in s:
            return Assertion.HISTORICAL

    return Assertion.ASSERTED
