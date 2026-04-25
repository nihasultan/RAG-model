from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks):
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def retrieve(query, chunks, index, k=5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        if idx < len(chunks):   # safety check
            results.append(chunks[idx])

    return results