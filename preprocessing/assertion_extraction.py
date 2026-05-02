"""
Rule-based + extensible assertion extraction.
Designed to mirror NegEx / ConText-style logic.
"""

from enum import Enum
import re

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
    "cannot exclude", "could be",   "may have", "might have",   "if ... then", "suggestive of", "consider" #add more as needed and alter
]

HISTORICAL_PATTERNS = [
    "history of",
    "previous",
    "prior"
]

FAMILY_PATTERNS = [
    "family history", "family history of", "mother had", "father had", "sister had", "brother had", "daughter had", "son had", "parent had", "grandmother had", "grandfather had", "aunt had", "uncle had", "cousin had", #add more as needed.. these are just examples
]

# Order matters: family > negative > hypothetical > historical > asserted
# Also, sort patterns by length descending to match longest phrase first.
def compile_patterns(patterns):
    sorted_patterns = sorted(patterns, key=len, reverse=True)
    # create a regex with word boundaries around each pattern
    pattern_regex = r"\b(" + "|".join(re.escape(p) for p in sorted_patterns) + r")\b"
    return re.compile(pattern_regex)

NEGATION_REGEX = compile_patterns(NEGATION_PATTERNS)
HYPOTHETICAL_REGEX = compile_patterns(HYPOTHETICAL_PATTERNS)
HISTORICAL_REGEX = compile_patterns(HISTORICAL_PATTERNS)
FAMILY_REGEX = compile_patterns(FAMILY_PATTERNS)

def extract_assertion(sentence: str) -> Assertion:
    s = sentence.lower()

    if FAMILY_REGEX.search(s):
        return Assertion.FAMILY

    if NEGATION_REGEX.search(s):
        return Assertion.NEGATED

    if HYPOTHETICAL_REGEX.search(s):
        return Assertion.HYPOTHETICAL

    if HISTORICAL_REGEX.search(s):
        return Assertion.HISTORICAL

    return Assertion.ASSERTED
