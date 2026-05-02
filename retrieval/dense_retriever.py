from sentence_transformers import SentenceTransformer, util


class DenseRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)

    def retrieve(self, query, corpus_embeddings, corpus_texts, top_k=10):
        query_emb = self.encode([query])
        scores = util.cos_sim(query_emb, corpus_embeddings)[0]
        top_idx = scores.topk(k=top_k).indices.tolist()

        return [
            {
                "text": corpus_texts[i],
                "score": scores[i].item()
            }
            for i in top_idx
        ]
# Example usage
"""
if __name__ == "__main__":  
    retriever = DenseRetriever()
    corpus = [
        "The patient has a history of diabetes.",
        "He is currently experiencing chest pain.",
        "No evidence of infection was found."
    ]
    corpus_embeddings = retriever.encode(corpus)

    query = "Does the patient have diabetes?"
    results = retriever.retrieve(query, corpus_embeddings, corpus, top_k=2)
    for res in results:
        print(res)    
"""