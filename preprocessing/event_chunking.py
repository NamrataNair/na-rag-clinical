"""
Creates event-centric chunks from clinical documents.
"""

def chunk_document(sentences, window=1):
    chunks = []
    for i, sent in enumerate(sentences):
        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        context = " ".join(sentences[start:end])
        chunks.append(context)
    return chunks


# Example usage
"""
if __name__ == "__main__":  
    doc_sentences = [
        "The patient has a history of diabetes.",
        "He is currently experiencing chest pain.",
        "No evidence of infection was found."
    ]
    event_chunks = chunk_document(doc_sentences, window=1)
    for chunk in event_chunks:
        print(chunk)    
"""