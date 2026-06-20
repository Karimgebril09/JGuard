class ChunkNode:
    def __init__(self, chunk, similarity, is_outlier, no_neighbors):
        self.chunk = chunk
        self.similarity = similarity
        self.is_outlier = is_outlier
        self.no_neighbors = no_neighbors


class ChunkGraphAnalyzer:

    def __init__(self, similarity_threshold=0.30, outlier_threshold=0.15):
        
        self.similarity_threshold = similarity_threshold
        self.outlier_threshold = outlier_threshold

    def cosine_similarity(self, vector1, vector2):
    
        dot_product = 0
        norm1 = 0
        norm2 = 0

        for i in range(len(vector1)):
            dot_product += vector1[i] * vector2[i]
            norm1 += vector1[i] * vector1[i]
            norm2 += vector2[i] * vector2[i]

        norm1 = norm1 ** 0.5
        norm2 = norm2 ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def build_graph(self, chunks, embeddings):

        graph = {}
        node_info = []

        n = len(chunks)

        for i in range(n):
            graph[i] = []

        # Build connections between chunks
        for i in range(n):
            for j in range(i + 1, n):

                similarity = self.cosine_similarity(
                    embeddings[i],
                    embeddings[j]
                )

                if similarity >= self.similarity_threshold:
                    graph[i].append(j)
                    graph[j].append(i)

        # Calculate average similarity
        for i in range(n):

            total_similarity = 0

            for j in range(n):

                if i != j:
                    total_similarity += self.cosine_similarity(
                        embeddings[i],
                        embeddings[j]
                    )

            if n > 1:
                average_similarity = total_similarity / (n - 1)
            else:
                average_similarity = 1

            is_outlier = average_similarity < self.outlier_threshold

            node = ChunkNode(
                chunk=chunks[i],
                similarity=average_similarity,
                is_outlier=is_outlier,
                no_neighbors=len(graph[i])
            )

            node_info.append(node)

        return graph, node_info