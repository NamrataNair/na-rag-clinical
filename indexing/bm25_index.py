from rank_bm25 import BM25Okapi


def build_bm25(corpus):
    tokenized = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized)
