"""
Builds assertion-aware index entries.
"""

from preprocessing.assertion_extraction import extract_assertion


def build_index(documents):
    """
    documents: list of list of sentences
    """
    index = []

    for doc_id, sentences in enumerate(documents):
        for sent in sentences:
            assertion = extract_assertion(sent)
            index.append({
                "doc_id": doc_id,
                "text": sent,
                "assertion": assertion.value
            })

    return index


# Example usage
"""
if __name__ == "__main__":  
    docs = [
        [
            "The patient has a history of diabetes.",
            "He is currently experiencing chest pain.",
            "No evidence of infection was found."
        ],
        [
            "Family history of heart disease is noted.",
            "The patient denies any current symptoms."
        ]
    ]
    index = build_index(docs)
    for entry in index:
        print(entry)    
"""