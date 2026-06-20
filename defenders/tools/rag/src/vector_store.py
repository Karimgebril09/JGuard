import faiss


class VectorStore:

    def __init__(self, dim):

        self.index = faiss.IndexFlatIP(dim)

        self.chunks = []

    def add(self, chunks, embeddings):

        self.index.add(embeddings)
        for chunk in chunks:
            self.chunks.append(chunk)

    def search(self, query_vector, top_k=5):

        scores, indices = self.index.search( query_vector, top_k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append((self.chunks[index], float(score)) )

        return results