from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def rerank(query, results):
    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)

    scored = list(zip(results, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [r[0] for r in scored]